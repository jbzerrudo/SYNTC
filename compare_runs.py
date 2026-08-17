"""
Compare two SynTC catalogues and report the climate signal between them.

    python compare_runs.py --control ./run03 --experiment ./run04

run03 is the stationary control and run04 carries the 4% per century intensity
ceiling trend. Neither run means much on its own. The control's 100-year return
level is a validation number, testable against the observed record. The warming
run's cannot be validated that way, because it is a sample from a climate that
has not happened. What the warming run supports is a difference, and measuring
that difference is what this script is for.

How the uncertainty is built
----------------------------
Each ensemble is one independent 100-year realisation, so 20 ensembles give 20
independent estimates of every statistic. The spread across ensembles is the
sampling uncertainty, reported as the standard error of the mean, and the
difference between runs carries a 95% interval built from the two standard
errors. No bootstrap is needed, because the ensembles already are the resamples.

The two runs share Config.seed, so they begin from the same genesis draws and
the comparison is quieter than two independently seeded runs would be. They do
not stay paired: once a storm in one run outlives its counterpart, the random
streams desynchronise. The comparison here is therefore treated as unpaired,
which is the conservative choice. A genuine pairing would narrow these
intervals, not widen them, so nothing below is overstated.

Why intensity classes are counted per storm
-------------------------------------------
A share computed over track points answers "what fraction of six-hourly
positions were at super typhoon strength", which is really a question about how
long storms linger. Counting each storm once, by its strongest wind inside PAR,
answers "what fraction of storms reached super typhoon strength", which is the
question a hazard statement is actually making. This script does the latter and
labels the rows accordingly.

Reading the drift table
-----------------------
The trend is applied linearly from the first simulated year, so an early window
and a late window of the warming run bracket the imposed change. The control
gets the same treatment as a null: whatever early-to-late change appears there
is sampling noise, and it sets the scale for reading the warming run's drift.
A change in the warming run no larger than the control's is not a signal.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

CATEGORIES = (("TD", 22, 34), ("TS", 34, 48), ("STS", 48, 64),
              ("TY", 64, 100), ("STY", 100, 1e9))
COLS = ["SID", "YEAR", "WIND", "IN_PAR", "OVER_LAND"]
# The manuscript's overland ceiling claim is about this threshold.
OVERLAND_THRESHOLD_KT = 106.0

LABELS = [
    ("storms_per_year",                 "PAR storms per year",          "{:8.2f}"),
    ("mean_wind_par",                   "mean wind in PAR (kt)",        "{:8.2f}"),
    ("median_storm_max",                "median storm max (kt)",        "{:8.2f}"),
    ("p90_storm_max",                   "90th pct storm max (kt)",      "{:8.2f}"),
    ("p99_storm_max",                   "99th pct storm max (kt)",      "{:8.2f}"),
    ("max_wind_par",                    "highest wind in PAR (kt)",     "{:8.1f}"),
    (None,                              "",                             None),
    ("pct_TD",                          "storms peaking TD (%)",        "{:8.2f}"),
    ("pct_TS",                          "storms peaking TS (%)",        "{:8.2f}"),
    ("pct_STS",                         "storms peaking STS (%)",       "{:8.2f}"),
    ("pct_TY",                          "storms peaking TY (%)",        "{:8.2f}"),
    ("pct_STY",                         "storms peaking STY (%)",       "{:8.2f}"),
    (None,                              "",                             None),
    ("max_wind_overland",               "highest wind over land (kt)",  "{:8.1f}"),
    ("overland_pts_ge_106",             "overland points >= 106 kt",    "{:8.1f}"),
    ("overland_storms_ge_100_per_year", "overland STY storms per year", "{:8.3f}"),
]
LOOKUP = {k: lab for k, lab, _ in LABELS if k}
DRIFT_KEYS = ("mean_wind_par", "p99_storm_max", "pct_STY", "max_wind_par")


def stats_from(d):
    """Every reported statistic, for one already-loaded slice of a catalogue."""
    if not len(d):
        return None
    n_years = int(d.YEAR.nunique())

    par = d[d.IN_PAR == 1]
    storm_max = par.groupby("SID").WIND.max().to_numpy()
    land = d[d.OVER_LAND == 1]
    land_storm_max = (land.groupby("SID").WIND.max().to_numpy()
                      if len(land) else np.array([]))
    if not len(storm_max):
        return None

    out = {
        "storms_per_year": len(storm_max) / n_years,
        "mean_wind_par": float(par.WIND.mean()),
        "median_storm_max": float(np.median(storm_max)),
        "p90_storm_max": float(np.percentile(storm_max, 90)),
        "p99_storm_max": float(np.percentile(storm_max, 99)),
        "max_wind_par": float(storm_max.max()),
    }
    for name, lo, hi in CATEGORIES:
        out[f"pct_{name}"] = float(
            np.mean((storm_max >= lo) & (storm_max < hi)) * 100)

    out["max_wind_overland"] = float(land.WIND.max()) if len(land) else np.nan
    out["overland_pts_ge_106"] = (
        float((land.WIND >= OVERLAND_THRESHOLD_KT).sum()) if len(land) else 0.0)
    out["overland_storms_ge_100_per_year"] = (
        float(np.sum(land_storm_max >= 100.0) / n_years)
        if len(land_storm_max) else 0.0)
    return out


def run_tables(run, windows=()):
    """Read each ensemble once and return the full-period table plus one table
    per year window. Reading once matters: twenty ensembles at fifteen
    megabytes each is a third of a gigabyte per pass, and the naive version
    made three passes per run."""
    files = sorted(glob.glob(os.path.join(run, "synthetic_storms_ens*.csv")))
    if not files:
        raise SystemExit(f"no ensemble CSVs in {run}")
    rows = {"full": []}
    for i in range(len(windows)):
        rows[i] = []
    for f in files:
        d = pd.read_csv(f, usecols=COLS)
        s = stats_from(d)
        if s:
            rows["full"].append(s)
        for i, (lo, hi) in enumerate(windows):
            s = stats_from(d[d.YEAR.between(lo, hi)])
            if s:
                rows[i].append(s)
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def summarise(df):
    """Mean and standard error of the mean across ensembles."""
    return df.mean(), df.std(ddof=1) / np.sqrt(len(df)), len(df)


def return_level_100(run):
    """The 100-year Weibull level from return_levels.py, if it has been run."""
    path = os.path.join(run, "return_levels.csv")
    if not os.path.exists(path):
        return None, None
    d = pd.read_csv(path)
    d = d[d.source.str.startswith("synthetic") & (d.return_period == 100.0)]
    if not len(d):
        return None, None
    return float(d.Weibull.iloc[0]), float(d.Weibull_ME.iloc[0])


def show(mc, sc, me, se, nc, ne, ctl, exp):
    print(f"\n{'':<32}{ctl + ' (control)':>22}{exp + ' (warming)':>22}"
          f"{'exp - ctl, 95% CI':>26}")
    print("-" * 102)
    for key, label, fmt in LABELS:
        if key is None:
            print()
            continue
        c, e = mc[key], me[key]
        # Independent means, so the variances of the difference add.
        half = 1.96 * float(np.sqrt(sc[key] ** 2 + se[key] ** 2))
        d = e - c
        flag = "" if (d - half) <= 0 <= (d + half) else "  *"
        print(f"{label:<32}{fmt.format(c):>13} +-{sc[key]:>6.2f}"
              f"{fmt.format(e):>13} +-{se[key]:>6.2f}"
              f"{d:>16.2f} +-{half:5.2f}{flag}")
    print("-" * 102)
    print(f"ensembles: {nc} control, {ne} warming.  "
          f"* marks a difference whose 95% interval excludes zero.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True,
                    help="the stationary run, --mpi-trend 0.0")
    ap.add_argument("--experiment", required=True,
                    help="the warming run, --mpi-trend 4.0")
    ap.add_argument("--drift-years", type=int, default=25,
                    help="length of the early and late windows used to show "
                         "the trend inside each run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ctl = os.path.basename(a.control.rstrip(os.sep))
    exp = os.path.basename(a.experiment.rstrip(os.sep))

    first = sorted(glob.glob(os.path.join(
        a.experiment, "synthetic_storms_ens*.csv")))
    if not first:
        raise SystemExit(f"no ensemble CSVs in {a.experiment}")
    yr = pd.read_csv(first[0], usecols=["YEAR"]).YEAR
    y0, y1 = int(yr.min()), int(yr.max())
    w = a.drift_years
    windows = ((y0, y0 + w - 1), (y1 - w + 1, y1))

    tc = run_tables(a.control, windows)
    te = run_tables(a.experiment, windows)

    mc, sc, nc = summarise(tc["full"])
    me, se, ne = summarise(te["full"])
    show(mc, sc, me, se, nc, ne, ctl, exp)

    for run, label in ((a.control, ctl), (a.experiment, exp)):
        v, err = return_level_100(run)
        if v is None:
            print(f"\n{label}: no return_levels.csv yet, run return_levels.py")
        else:
            print(f"\n{label}: 100-year Weibull return level "
                  f"{v:.1f} +- {err:.1f} kt")
    print("\nOnly the control's return level is a validation number. The "
          "warming run's is a projection\nwith no observed counterpart to test "
          "it against.")

    print(f"\n\nDrift inside each run: {windows[0][0]}-{windows[0][1]} against "
          f"{windows[1][0]}-{windows[1][1]}")
    print("-" * 102)
    print(f"{'':<32}{'early':>12}{'late':>12}{'change':>12}    run")
    for tab, label in ((tc, ctl), (te, exp)):
        early = summarise(tab[0])[0]
        late = summarise(tab[1])[0]
        for key in DRIFT_KEYS:
            print(f"{LOOKUP[key]:<32}{early[key]:>12.2f}{late[key]:>12.2f}"
                  f"{late[key] - early[key]:>12.2f}    {label}")
        print()
    print("The control's early-to-late change is the noise floor. Anything the "
          "warming run does\nthat is not clearly larger than that is not a "
          "signal.")

    out = a.out or os.path.join(a.experiment, "run_comparison.csv")
    res = pd.concat([tc["full"].assign(run=ctl, role="control"),
                     te["full"].assign(run=exp, role="experiment")],
                    ignore_index=True)
    res.to_csv(out, index=False)
    print(f"\nper-ensemble values written: {out}")


if __name__ == "__main__":
    main()
