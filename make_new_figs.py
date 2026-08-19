"""Two figures for the revised manuscript, in the house style of the release."""
import argparse, glob, sys
import numpy as np, pandas as pd
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import terrain
import intensity as I
from syntc_ai import in_par

GRID = "#c8c8c8"; OBS = "#000000"; SYN = "#2ca02c"; WARN = "#d62728"; ACC = "#1f77b4"
_ap = argparse.ArgumentParser(description="Annual-maximum and saturation-exponent figures.")
_ap.add_argument("--ibtracs", required=True)
_ap.add_argument("--run", required=True, help="the stationary catalogue folder, e.g. run07")
_ap.add_argument("--dtm", default="dtm_phil_1km.tif")
_ap.add_argument("--scout", default=".", help="folder holding scoutk_06 .. scoutk_12")
A = _ap.parse_args()
IB, RUN = A.ibtracs, A.run
terrain.DTM_PATH = A.dtm

def grid(ax):
    ax.grid(which="major", ls="--", lw=0.6, color=GRID, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)

o = I.load_intensity_points(IB, season_max=2023)
o = o[in_par(o.lat.to_numpy(), o.lon.to_numpy())]
obs = np.sort(o.groupby("SEASON").vmax_raw.max().to_numpy().astype(float))
syn = np.sort(np.concatenate([pd.read_csv(f, usecols=["YEAR","WIND","IN_PAR"])
        .query("IN_PAR == 1").groupby("YEAR").WIND.max().to_numpy()
        for f in sorted(glob.glob(RUN + "/synthetic_storms_ens*.csv"))]).astype(float))

# ---------------------------------------------------------------- figure A
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.1), facecolor="white")

bins = np.arange(65, 146, 5)
a1.hist(obs, bins=bins, density=True, color=OBS, alpha=0.22, label="observed 1977-2023 (n = 47)")
a1.hist(syn, bins=bins, density=True, histtype="step", lw=1.8, color=SYN,
        label="SynTC, 20 ensembles (n = 2,000)")
for v, c, ls in ((np.median(obs), OBS, "--"), (np.median(syn), SYN, "--")):
    a1.axvline(v, color=c, lw=1.1, ls=ls, zorder=5)
a1.annotate("medians 105.5 / 110.0 kt", xy=(np.median(obs), a1.get_ylim()[1]*0.52),
            xytext=(9, 0), textcoords="offset points", ha="left", fontsize=8.5, color="#444444")
a1.set_xlabel("PAR annual maximum wind (kt)", fontsize=10)
a1.set_ylabel("Density", fontsize=10)
a1.legend(fontsize=8.2, framealpha=0.95, edgecolor="#999999", loc="upper left")
a1.text(0.012, 1.02, "(a)", transform=a1.transAxes, fontsize=11, fontweight="bold", va="bottom")
grid(a1)

def emp(x):
    n = len(x); i = np.arange(1, n + 1)
    return (n + 1) / (n + 1 - i), x            # Weibull plotting position
Tobs, yobs = emp(obs); Tsyn, ysyn = emp(syn)
a2.semilogx(Tobs, yobs, "o", ms=4.5, mfc="white", mec=OBS, mew=1.0, zorder=6,
            label="observed, empirical")
a2.semilogx(Tsyn, ysyn, "-", lw=1.8, color=SYN, zorder=5, label="SynTC, empirical")
T = np.logspace(0.05, 2.4, 200); P = 1 - 1/T
for x, c, lab in ((obs, OBS, "observed, Weibull"), (syn, SYN, "SynTC, Weibull")):
    pr = stats.weibull_min.fit(x, floc=0)
    a2.semilogx(T, stats.weibull_min.ppf(P, *pr), ls=(0, (5, 2)), lw=1.4, color=c,
                alpha=0.85, zorder=4, label=lab)
a2.axvline(100, color="#555555", lw=0.8, ls=":", zorder=3)
a2.annotate("100-yr: 126.4 obs, 125.1 SynTC", xy=(100, 74), xytext=(-6, 0),
            textcoords="offset points", fontsize=8.5, color="#444444", va="center", ha="right")
a2.set_xlabel("Return period (years)", fontsize=10)
a2.set_ylabel("Annual maximum wind (kt)", fontsize=10)
a2.set_xlim(1.05, 260); a2.set_ylim(70, 145)
a2.legend(fontsize=8.2, framealpha=0.95, edgecolor="#999999", loc="upper left")
a2.text(0.012, 1.02, "(b)", transform=a2.transAxes, fontsize=11, fontweight="bold", va="bottom")
grid(a2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"fig_annual_maxima.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- figure B
ks = np.array([6, 8, 10, 12], float)
att, olm = [], []
for k in (6, 8, 10, 12):
    sm = np.concatenate([np.load(os.path.join(A.scout, "scoutk_%02d" % k, "e%02d.npz" % e))["stormmax"] for e in range(1, 6)])
    att.append(100 * np.mean(sm >= 100))
    olm.append([float(np.load(os.path.join(A.scout, "scoutk_%02d" % k, "e%02d.npz" % e))["olmax"][0]) for e in range(1, 6)])
att = np.array(att); olm = np.array(olm)

fig, (b1, b2) = plt.subplots(1, 2, figsize=(9.6, 3.9), facecolor="white")
b1.plot(ks, att, "-o", lw=1.8, color=ACC, ms=6, mfc="white", mec=ACC, mew=1.4, zorder=6)
b1.axhline(16.27, color=OBS, lw=1.2, ls="--", zorder=4)
b1.annotate("observed 16.3%", xy=(12.1, 16.27), xytext=(-4, 5), textcoords="offset points",
            ha="right", fontsize=8.5, color="#444444")
b1.set_xlabel(r"saturation exponent $k$", fontsize=10)
b1.set_ylabel("storms peaking $\\geq$100 kt in PAR (%)", fontsize=10)
b1.set_xticks(ks); b1.set_ylim(7.5, 17.5)
b1.text(0.012, 1.02, "(a)", transform=b1.transAxes, fontsize=11, fontweight="bold", va="bottom")
grid(b1)

for i, k in enumerate(ks):
    b2.plot([k]*5, olm[i], "o", ms=4.5, color=WARN, alpha=0.45, zorder=5)
b2.plot(ks, olm.mean(axis=1), "-s", lw=1.8, color=WARN, ms=6, mfc="white", mec=WARN, mew=1.4, zorder=6)
b2.axhline(112, color=OBS, lw=1.2, ls="--", zorder=4)
b2.axhline(120, color="#555555", lw=1.0, ls=":", zorder=4)
b2.annotate("benchmark median 112 kt", xy=(12.1, 112), xytext=(-4, -11),
            textcoords="offset points", ha="right", fontsize=8.5, color="#444444")
b2.annotate("benchmark 95th pct 120 kt", xy=(12.1, 120), xytext=(-4, 4),
            textcoords="offset points", ha="right", fontsize=8.5, color="#555555")
b2.set_xlabel(r"saturation exponent $k$", fontsize=10)
b2.set_ylabel("highest wind over land, per ensemble (kt)", fontsize=10)
b2.set_xticks(ks); b2.set_ylim(103, 130)
b2.text(0.012, 1.02, "(b)", transform=b2.transAxes, fontsize=11, fontweight="bold", va="bottom")
grid(b2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"fig_saturation_tradeoff.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote fig_annual_maxima and fig_saturation_tradeoff (png + pdf)")
