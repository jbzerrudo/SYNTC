"""
Does the Philippine archipelago weaken storms the way the record says it does?

    python filtering_effect.py --ibtracs /path/to/IBTrACS.WP.list.v04r01.points.csv \
        --dtm /path/to/dtm_phil_1km.tif --run ./run03

This is the quantitative form of the filtering effect. The claim the manuscript
has been reaching for, that the archipelago caps intensity, is better made as a
measured loss than as a threshold nothing crosses. A threshold claim is fragile
and, as it happens, false: the observed record contains exactly one point at or
above 106 kt with its centre over Philippine land, Haiyan at 110 kt on 8
November 2013 over Leyte and Samar. A model that produced none would be failing
to reproduce Haiyan, which is worse than producing too many.

What is measured
----------------
A storm counts as a CROSSING if it has at least one point with its centre over
land and its first point back over water lies at least half a degree further
west, so it went through the archipelago rather than clipping a coast and
turning away. For each crossing the entry wind is taken at its first over-land
point and the exit wind at its first point back over water.

A storm counts as DESTROYED if it reaches land and never returns to water. That
is the strongest form of filtering there is and it belongs in the same table.

Both are reported per 100 seasons, so the 47-year record and a 100-year-per-
ensemble catalogue can be compared without either being rescaled by hand.

Why the maximum needs a matched benchmark
-----------------------------------------
The generator prints its highest over-land wind against the observed record
maximum of 110 kt. That comparison is unfair in the model's favour and it looks
unfair against it: the record covers 47 seasons and each ensemble covers 100, so
a higher maximum is expected. The honest benchmark is the distribution of the
largest value in a 100-season draw from the observed over-land annual maxima,
which this script computes and prints alongside.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

import intensity as I
import terrain

BANDS = ((0, 64, "below TY"), (64, 100, "TY 64-99 kt"), (100, 999, "STY >=100 kt"))
WEST_DEG = 0.5


def crossings(df, sid, lat, lon, wind, over_land, west_deg=WEST_DEG):
    """One row per storm that reached land: entry wind, exit wind, fate."""
    rows = []
    for _, s in df.groupby(sid, sort=False):
        s = s.reset_index(drop=True)
        ol = s.index[s[over_land].astype(bool)].tolist()
        if not ol:
            continue
        i, j = ol[0], ol[-1]
        entry = float(s[wind][i])
        if j + 1 >= len(s):
            rows.append((entry, np.nan, len(ol), "destroyed"))
            continue
        exit_ = float(s[wind][j + 1])
        went_west = s[lon][j + 1] < s[lon][i] - west_deg
        rows.append((entry, exit_, len(ol),
                     "crossing" if went_west else "turned away"))
    out = pd.DataFrame(rows, columns=["entry", "exit", "land_steps", "fate"])
    out["loss_pct"] = 100 * (out.entry - out["exit"]) / out.entry
    return out


def observed(ibtracs, dtm, season_max=2023):
    d = I.load_intensity_points(ibtracs, season_max=season_max)
    d = d.sort_values(["SID", "time"])
    _, land = terrain.get(dtm).sample(d.lat.to_numpy(), d.lon.to_numpy())
    d = d.assign(OL=land.astype(int))
    n_seasons = int(d.SEASON.nunique())
    return crossings(d, "SID", "lat", "lon", "vmax_raw", "OL"), n_seasons, d


def synthetic(run):
    files = sorted(glob.glob(os.path.join(run, "synthetic_storms_ens*.csv")))
    if not files:
        raise SystemExit(f"no ensemble CSVs in {run}")
    parts, seasons = [], 0
    for f in files:
        d = pd.read_csv(f, usecols=["SID", "STEP", "LAT", "LON", "WIND",
                                    "OVER_LAND", "YEAR"])
        d = d.sort_values(["SID", "STEP"])
        seasons += int(d.YEAR.nunique())
        parts.append(crossings(d, "SID", "LAT", "LON", "WIND", "OVER_LAND"))
    return pd.concat(parts, ignore_index=True), seasons, len(files)


def report(tag, c, n_seasons):
    per100 = lambda k: 100.0 * k / n_seasons
    cr = c[c.fate == "crossing"]
    print(f"\n--- {tag}  ({n_seasons} seasons) ---")
    print(f"  storms reaching land       {len(c):6d}   {per100(len(c)):7.1f} per 100 seasons")
    print(f"    crossed through          {len(cr):6d}   {per100(len(cr)):7.1f}")
    print(f"    turned away              {int((c.fate=='turned away').sum()):6d}"
          f"   {per100((c.fate=='turned away').sum()):7.1f}")
    print(f"    destroyed over land      {int((c.fate=='destroyed').sum()):6d}"
          f"   {per100((c.fate=='destroyed').sum()):7.1f}")
    if not len(cr):
        return
    print(f"  crossings: median entry {cr.entry.median():.0f} -> exit "
          f"{cr['exit'].median():.0f} kt, median loss {cr.loss_pct.median():.0f}%")
    print(f"  {'entry band':<14}{'n':>6}{'per 100 yr':>12}{'med loss':>10}"
          f"{'med exit':>10}{'max exit':>10}")
    for lo, hi, lab in BANDS:
        b = cr[(cr.entry >= lo) & (cr.entry < hi)]
        if not len(b):
            print(f"  {lab:<14}{0:>6}{0.0:>12.1f}{'-':>10}{'-':>10}{'-':>10}")
            continue
        print(f"  {lab:<14}{len(b):>6}{per100(len(b)):>12.1f}"
              f"{b.loss_pct.median():>9.0f}%{b['exit'].median():>10.0f}"
              f"{b['exit'].max():>10.0f}")


def max_benchmark(obs_points, syn_run, n_draw):
    """Largest over-land wind: what a 100-season draw should produce."""
    am = obs_points[obs_points.OL == 1].groupby("SEASON").vmax_raw.max().to_numpy()
    sh, loc, sc = stats.weibull_min.fit(am, floc=0)
    rng = np.random.default_rng(0)
    sim = stats.weibull_min.rvs(sh, loc, sc, size=(20000, n_draw),
                                random_state=rng).max(axis=1)
    print(f"\n--- highest wind with the centre over land ---")
    print(f"  observed record, {len(am)} seasons                 : {am.max():.0f} kt")
    print(f"  expected largest in a {n_draw}-season draw     : median "
          f"{np.median(sim):.0f}, 90% range {np.percentile(sim,5):.0f}-"
          f"{np.percentile(sim,95):.0f} kt")
    per_ens = []
    for f in sorted(glob.glob(os.path.join(syn_run, "synthetic_storms_ens*.csv"))):
        d = pd.read_csv(f, usecols=["WIND", "OVER_LAND"])
        d = d[d.OVER_LAND == 1]
        if len(d):
            per_ens.append(float(d.WIND.max()))
    if per_ens:
        p = np.array(per_ens)
        print(f"  SynTC, {len(p)} ensembles                        : median "
              f"{np.median(p):.0f}, range {p.min():.0f}-{p.max():.0f} kt")
        print(f"  ensembles above the 95th percentile of the benchmark: "
              f"{int((p > np.percentile(sim,95)).sum())} of {len(p)}  "
              f"(5% expected)")
    return per_ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibtracs", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    terrain.DTM_PATH = a.dtm

    obs, n_obs, obs_points = observed(a.ibtracs, a.dtm)
    syn, n_syn, n_ens = synthetic(a.run)
    report("observed 1977-2023", obs, n_obs)
    report(f"SynTC {os.path.basename(a.run.rstrip(os.sep))} "
           f"({n_ens} ensembles)", syn, n_syn)
    max_benchmark(obs_points, a.run, 100)

    out = a.out or os.path.join(a.run, "filtering_effect.csv")
    pd.concat([obs.assign(source="observed"),
               syn.assign(source="synthetic")], ignore_index=True).to_csv(
        out, index=False)
    print(f"\nper-storm records written: {out}")
    print("\nThe filtering effect is the loss column, not a threshold. Compare "
          "the median loss and the\nexit winds band by band; those are the "
          "numbers the archipelago actually controls.")


if __name__ == "__main__":
    main()
