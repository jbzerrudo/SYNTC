"""
fig_run07_run09_observed.pdf: both catalog configurations against the record.

    python make_run_compare.py --control run07 --experiment run09 \
        --ibtracs IBTrACS.WP.list.v04r01.points.csv --dtm dtm_phil_1km.tif

    python make_run_compare.py --from-csv run07_vs_run09.csv \
        --rl-control rl_run07.csv --rl-experiment rl_run09.csv \
        --ibtracs IBTrACS.WP.list.v04r01.points.csv --dtm dtm_phil_1km.tif

The figure the paper uses to say that a longer memory in the track network buys
nothing measurable. Two panels:

  (a) each catalog statistic as a percentage departure from its observed value,
      with the 95% interval on the observed statistic drawn as a band. A marker
      inside the band is a configuration that cannot be told apart from the
      record on that statistic.

  (b) Weibull return levels for the observed record and both catalogs, with the
      observed bootstrap interval shaded.

Why the intervals differ by statistic
-------------------------------------
An event rate over 47 seasons is a count, so its interval is the exact Poisson
interval on that count. A class share is a proportion of 762 storms, so its
interval is Clopper-Pearson. A percentile or a mean is neither, so its interval
comes from a nonparametric bootstrap that resamples whole storms rather than
track points: points within one storm are not independent, and resampling them
individually would give an interval several times too narrow.

Using one interval type for all twelve would have been simpler and wrong in
different directions for different rows, which matters here because the whole
figure is an argument about which differences are real.

Twelve statistics are computed, eleven are drawn. Overland points at or above
106 kt is omitted from the panel because the observed value is a single track
point, Haiyan over Leyte, and its Poisson interval spans -98% to +457%; drawn to
scale it would compress every other row to a hairline. Its value is printed to
stdout instead.

--from-csv reuses the per-ensemble statistics that compare_runs.py already
wrote, so the figure and the stored table cannot disagree. IBTrACS is still
needed in that mode, because the observed side is not in the CSV.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import terrain

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ibtracs", required=True)
ap.add_argument("--dtm", default="dtm_phil_1km.tif")
ap.add_argument("--control", help="the single-step run folder, e.g. run07")
ap.add_argument("--experiment", help="the extended-memory run folder, e.g. run09")
ap.add_argument("--from-csv", dest="from_csv",
                help="per-ensemble statistics from compare_runs.py, with a "
                     "'run' column naming the two configurations")
ap.add_argument("--rl-control", default=None, help="rl_run07.csv, for panel (b)")
ap.add_argument("--rl-experiment", default=None, help="rl_run09.csv, for panel (b)")
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--out", default=".")
ap.add_argument("--stem", default="fig_run07_run09_observed")
A = ap.parse_args()
terrain.DTM_PATH = A.dtm
if not A.from_csv and not (A.control and A.experiment):
    sys.exit("give either --from-csv, or both --control and --experiment")

import intensity as I                       # noqa: E402  (needs DTM_PATH first)
from syntc_ai import in_par                 # noqa: E402

CTRL = "#1f4fd8"      # run07, single-step
EXPT = "#c81f1f"      # run09, three-step
BAND = "#b8b8b8"
RNG = np.random.default_rng(0)

# key, label, kind. Order is the order of the rows, top to bottom.
STATS = [
    ("storms_per_year",                 "PAR storms per year",       "rate"),
    ("median_storm_max",                "median storm max",          "boot"),
    ("overland_storms_ge_100_per_year", "overland STY storms / yr",  "rate"),
    ("mean_wind_par",                   "mean wind in PAR",          "boot"),
    ("p90_storm_max",                   "90th pct storm max",        "boot"),
    ("p99_storm_max",                   "99th pct storm max",        "boot"),
    ("pct_TD",                          "storms peaking TD",         "share"),
    ("pct_TS",                          "storms peaking TS",         "share"),
    ("pct_STS",                         "storms peaking STS",        "share"),
    ("pct_TY",                          "storms peaking TY",         "share"),
    ("pct_STY",                         "storms peaking STY",        "share"),
]
OMITTED = ("overland_pts_ge_106", "overland points >=106 kt", "rate")


# ------------------------------------------------------------------- observed
def observed(ibtracs):
    p = I.load_intensity_points(ibtracs, season_max=2023)
    p = p[in_par(p.lat.to_numpy(), p.lon.to_numpy())].copy()
    te = terrain.get(A.dtm)
    _, land = te.sample(p.lat.to_numpy(), p.lon.to_numpy())
    p["land"] = land
    ny = p.SEASON.nunique()
    smax = p.groupby("SID").vmax_raw.max()
    ov = p[p.land]
    ovmax = ov.groupby("SID").vmax_raw.max()

    def share(lo, hi=None):
        m = (smax >= lo) if hi is None else ((smax >= lo) & (smax < hi))
        return int(m.sum())

    counts = {                       # k, and the divisor that turns k into the statistic
        "storms_per_year": (len(smax), ny),
        "overland_storms_ge_100_per_year": (int((ovmax >= 100).sum()), ny),
        "overland_pts_ge_106": (int((ov.vmax_raw >= 106).sum()), ny),
    }
    shares = {"pct_TD": share(22, 34), "pct_TS": share(34, 48),
              "pct_STS": share(48, 64), "pct_TY": share(64, 100),
              "pct_STY": share(100)}
    n_storms = len(smax)

    val, lo, hi = {}, {}, {}
    for k, (c, div) in counts.items():
        val[k] = c / div
        a, b = (stats.chi2.ppf(0.025, 2 * c) / 2 if c else 0.0,
                stats.chi2.ppf(0.975, 2 * (c + 1)) / 2)          # exact Poisson
        lo[k], hi[k] = a / div, b / div
    for k, c in shares.items():
        val[k] = 100.0 * c / n_storms
        a, b = stats.beta.ppf(0.025, c, n_storms - c + 1) if c else 0.0, \
               stats.beta.ppf(0.975, c + 1, n_storms - c) if c < n_storms else 1.0
        lo[k], hi[k] = 100.0 * a, 100.0 * b                       # Clopper-Pearson

    # Bootstrap resamples whole storms, not track points.
    sid = p.SID.to_numpy()
    order = np.argsort(sid, kind="stable")
    ids, starts = np.unique(sid[order], return_index=True)
    groups = np.split(p.vmax_raw.to_numpy()[order], starts[1:])
    sm = smax.to_numpy()
    val["mean_wind_par"] = p.vmax_raw.mean()
    val["median_storm_max"] = float(np.median(sm))
    val["p90_storm_max"] = float(np.percentile(sm, 90))
    val["p99_storm_max"] = float(np.percentile(sm, 99))
    B = {k: np.empty(A.n_boot) for k in
         ("mean_wind_par", "median_storm_max", "p90_storm_max", "p99_storm_max")}
    n = len(groups)
    for b in range(A.n_boot):
        idx = RNG.integers(0, n, n)
        pts = np.concatenate([groups[i] for i in idx])
        s = sm[idx]
        B["mean_wind_par"][b] = pts.mean()
        B["median_storm_max"][b] = np.median(s)
        B["p90_storm_max"][b] = np.percentile(s, 90)
        B["p99_storm_max"][b] = np.percentile(s, 99)
    for k, arr in B.items():
        lo[k], hi[k] = np.percentile(arr, [2.5, 97.5])

    print(f"observed: {n_storms} PAR storms over {ny} seasons, "
          f"{len(ov)} overland points")
    return val, lo, hi


# ------------------------------------------------------------------ synthetic
def per_ensemble(run):
    """The same twelve statistics, one row per ensemble."""
    te = terrain.get(A.dtm)
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "synthetic_storms_ens*.csv"))):
        d = pd.read_csv(f, usecols=["STORM_ID", "YEAR", "LAT", "LON", "WIND", "IN_PAR"])
        d = d[d.IN_PAR == 1]
        ny = d.YEAR.nunique()
        sm = d.groupby("STORM_ID").WIND.max()
        _, land = te.sample(d.LAT.to_numpy(), d.LON.to_numpy())
        ov = d[land]
        ovm = ov.groupby("STORM_ID").WIND.max()
        n = len(sm)
        rows.append(dict(
            storms_per_year=n / ny, mean_wind_par=d.WIND.mean(),
            median_storm_max=np.median(sm), p90_storm_max=np.percentile(sm, 90),
            p99_storm_max=np.percentile(sm, 99),
            pct_TD=100 * ((sm >= 22) & (sm < 34)).mean(),
            pct_TS=100 * ((sm >= 34) & (sm < 48)).mean(),
            pct_STS=100 * ((sm >= 48) & (sm < 64)).mean(),
            pct_TY=100 * ((sm >= 64) & (sm < 100)).mean(),
            pct_STY=100 * (sm >= 100).mean(),
            overland_pts_ge_106=(ov.WIND >= 106).sum() / ny,
            overland_storms_ge_100_per_year=(ovm >= 100).sum() / ny))
    return pd.DataFrame(rows)


if A.from_csv:
    tab = pd.read_csv(A.from_csv)
    runs = list(dict.fromkeys(tab.run))
    if len(runs) != 2:
        sys.exit(f"expected two runs in {A.from_csv}, found {runs}")
    ctrl_name, expt_name = runs
    ctrl = tab[tab.run == ctrl_name]
    expt = tab[tab.run == expt_name]
    if "overland_pts_ge_106" in tab and tab.overland_pts_ge_106.max() > 5:
        # stored as a count per ensemble-century, not a rate
        ctrl = ctrl.assign(overland_pts_ge_106=ctrl.overland_pts_ge_106 / 100.0)
        expt = expt.assign(overland_pts_ge_106=expt.overland_pts_ge_106 / 100.0)
else:
    ctrl_name = os.path.basename(A.control.rstrip(os.sep))
    expt_name = os.path.basename(A.experiment.rstrip(os.sep))
    ctrl, expt = per_ensemble(A.control), per_ensemble(A.experiment)

print(f"{ctrl_name}: {len(ctrl)} ensembles | {expt_name}: {len(expt)} ensembles")
obs, lo, hi = observed(A.ibtracs)

# ------------------------------------------------------------------- reporting
print(f"\n{'statistic':<34}{'observed':>10}{'  [95% interval]':>22}"
      f"{ctrl_name:>10}{expt_name:>10}   inside?")
inside = {ctrl_name: 0, expt_name: 0}
for key, label, kind in STATS + [OMITTED]:
    o, l, h = obs[key], lo[key], hi[key]
    c, e = ctrl[key].mean(), expt[key].mean()
    ic, ie = l <= c <= h, l <= e <= h
    inside[ctrl_name] += ic
    inside[expt_name] += ie
    tag = "".join(("C" if ic else "-", "E" if ie else "-"))
    star = "   (omitted from the panel)" if key == OMITTED[0] else ""
    print(f"{key:<34}{o:>10.3f}  [{l:>8.3f},{h:>8.3f}]{c:>10.3f}{e:>10.3f}   {tag}{star}")
print(f"\ninside the observed interval: {ctrl_name} {inside[ctrl_name]} of 12, "
      f"{expt_name} {inside[expt_name]} of 12")

# --------------------------------------------------------------------- figure
fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 5.2), facecolor="white",
                             gridspec_kw={"width_ratios": [1.30, 1.0]})

y = np.arange(len(STATS))[::-1]
for i, (key, label, kind) in enumerate(STATS):
    o, l, h = obs[key], lo[key], hi[key]
    ax.barh(y[i], 100 * (h - l) / o, left=100 * (l - o) / o, height=0.62,
            color=BAND, zorder=1)
    ax.plot(100 * (ctrl[key].mean() - o) / o, y[i], "o", ms=5.5, color=CTRL, zorder=4)
    ax.plot(100 * (expt[key].mean() - o) / o, y[i], "s", ms=5.0, color=EXPT, zorder=4)

n_in = sum(1 for k, _, _ in STATS if lo[k] <= ctrl[k].mean() <= hi[k])
ax.axhline(y[n_in] + 0.5, color="#333333", lw=0.9, ls=(0, (4, 3)), zorder=3)
ax.axvline(0, color="#111111", lw=1.4, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([lab for _, lab, _ in STATS], fontsize=9.0)
ax.set_xlabel("departure from observed record (%)", fontsize=10.5)
ax.set_ylim(-0.7, len(STATS) - 0.3)
ax.grid(axis="x", color="#c8c8c8", ls="--", lw=0.6, alpha=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("(a)", fontsize=11, loc="center")
xr = ax.get_xlim()[1]
ax.text(xr, y[0] + 0.15, "inside the\nobserved interval", fontsize=8, style="italic",
        color="#555555", ha="right", va="top")
ax.text(xr, y[n_in] - 0.3, "outside the\nobserved interval", fontsize=8,
        style="italic", color="#555555", ha="right", va="top")


def weibull_curve(path, which):
    d = pd.read_csv(path)
    src = [s for s in d.source.unique() if which in s]
    if not src:
        return None
    d = d[d.source == src[0]]
    return (d.return_period.to_numpy(), d.Weibull.to_numpy(),
            d.Weibull_lo.to_numpy(), d.Weibull_hi.to_numpy())


rlc = weibull_curve(A.rl_control, "observed") if A.rl_control else None
if rlc:
    T, w, l, h = rlc
    bx.fill_between(T, l, h, color=BAND, alpha=0.75, zorder=1,
                    label="observed 95% interval")
    bx.semilogx(T, w, "-", lw=1.8, color="#111111", zorder=4,
                label="observed 1977$-$2023")
for path, name, col, ls in ((A.rl_control, ctrl_name, CTRL, "-"),
                            (A.rl_experiment, expt_name, EXPT, (0, (5, 2)))):
    if not path:
        continue
    c = weibull_curve(path, "synthetic")
    if c:
        bx.semilogx(c[0], c[1], ls=ls, lw=1.6, color=col, zorder=5, label=name)
bx.set_xlabel("return period (yr)", fontsize=10.5)
bx.set_ylabel("10-min sustained wind (kt)", fontsize=10.5)
bx.grid(color="#c8c8c8", ls="--", lw=0.6, alpha=0.8, zorder=0)
bx.set_axisbelow(True)
for s in ("top", "right"):
    bx.spines[s].set_visible(False)
bx.set_title("(b)", fontsize=11, loc="center")
bx.legend(fontsize=8.5, framealpha=0.95, edgecolor="#999999", loc="lower right")

fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=5.5, color=CTRL),
                    Line2D([], [], ls="none", marker="s", ms=5.0, color=EXPT),
                    Patch(facecolor=BAND)],
           labels=[f"{ctrl_name}, single-step memory",
                   f"{expt_name}, three-step memory",
                   "observed 95% interval"],
           ncol=3, frameon=False, fontsize=9.0, loc="lower center",
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 1])
for ext in ("pdf", "png"):
    p = os.path.join(A.out, f"{A.stem}.{ext}")
    fig.savefig(p, dpi=250, bbox_inches="tight", facecolor="white")
    print("written:", p)
plt.close(fig)
