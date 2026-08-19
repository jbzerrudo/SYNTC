"""Landfall by Philippine island group: SynTC against the record, and the
observed trend. Island groups are latitude bands at 14.5 and 9.5 degrees N,
the same convention genesis_forecast.py uses, so the figure and the tool
report the same quantity.
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import terrain, intensity as I
from syntc_ai import in_par

ap = argparse.ArgumentParser()
ap.add_argument("--ibtracs", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--dtm", default="dtm_phil_1km.tif")
ap.add_argument("--out", default="fig_island_landfall")
A = ap.parse_args(); terrain.DTM_PATH = A.dtm

GROUPS = ("Luzon", "Visayas", "Mindanao")
COL = {"Luzon": "#1f77b4", "Visayas": "#ff7f0e", "Mindanao": "#2ca02c"}
GRID = "#c8c8c8"
grp = lambda lat: np.where(lat >= 14.5, "Luzon", np.where(lat >= 9.5, "Visayas", "Mindanao"))

def axes(ax):
    ax.grid(which="major", ls="--", lw=0.6, color=GRID, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---------------- observed ----------------
o = I.load_intensity_points(A.ibtracs, season_max=2023).sort_values(["SID", "time"])
_, l = terrain.get().sample(o.lat.to_numpy(), o.lon.to_numpy())
o = o.assign(OL=l.astype(int))
par = set(o[in_par(o.lat.to_numpy(), o.lon.to_numpy())].SID.unique())
first = o[(o.OL == 1) & (o.SID.isin(par))].groupby("SID").first().reset_index()
first["grp"] = grp(first.lat.to_numpy())
YR = np.arange(1977, 2024)
obs_share = {g: 100 * (first.grp == g).mean() for g in GROUPS}
counts = {g: first[first.grp == g].groupby("SEASON").size().reindex(YR, fill_value=0) for g in GROUPS}
total = first.groupby("SEASON").size().reindex(YR, fill_value=0)

# ---------------- synthetic ----------------
rows = []
for f in sorted(glob.glob(os.path.join(A.run, "synthetic_storms_ens*.csv"))):
    d = pd.read_csv(f, usecols=["SID","STEP","YEAR","LAT","IN_PAR","OVER_LAND"]).sort_values(["SID","STEP"])
    p = set(d[d.IN_PAR == 1].SID.unique())
    fi = d[(d.OVER_LAND == 1) & (d.SID.isin(p))].groupby("SID").first()
    g = grp(fi.LAT.to_numpy())
    rows.append([100 * np.mean(g == x) for x in GROUPS])
S = np.array(rows)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2), facecolor="white")

x = np.arange(3); w = 0.36
a1.bar(x - w/2, [obs_share[g] for g in GROUPS], w, color="#4d4d4d",
       label=f"observed 1977-2023 (n = {len(first)})", zorder=3)
a1.bar(x + w/2, S.mean(0), w, yerr=S.std(0, ddof=1), capsize=4, color="#2ca02c",
       error_kw=dict(lw=1.1, ecolor="#173d17"), label="SynTC, 20 ensembles", zorder=3)
for i, g in enumerate(GROUPS):
    a1.text(i - w/2, obs_share[g] + 1.4, f"{obs_share[g]:.1f}", ha="center", fontsize=8.5)
    a1.text(i + w/2, S.mean(0)[i] + S.std(0, ddof=1)[i] + 1.4, f"{S.mean(0)[i]:.1f}", ha="center", fontsize=8.5)
a1.set_xticks(x); a1.set_xticklabels(GROUPS); a1.set_ylim(0, 62)
a1.set_ylabel("share of land-crossing storms (%)", fontsize=10)
a1.legend(fontsize=8.4, framealpha=0.95, edgecolor="#999999", loc="upper right")
a1.text(0.012, 1.02, "(a)", transform=a1.transAxes, fontsize=11, fontweight="bold", va="bottom")
axes(a1)

for g in GROUPS:
    a2.plot(YR, counts[g].values, lw=1.0, color=COL[g], alpha=0.55, marker="o", ms=2.8, zorder=4)
    b = stats.linregress(YR, counts[g].values)
    a2.plot(YR, b.intercept + b.slope*YR, lw=1.6, color=COL[g], zorder=5,
            label=f"{g}  {b.slope*47:+.1f} over 47 yr  (p = {b.pvalue:.2f})")
bt = stats.linregress(YR, total.values); tau, pk = stats.kendalltau(YR, total.values)
a2.plot(YR, total.values, lw=1.0, color="#111111", alpha=0.45, marker="o", ms=3.2, zorder=4)
a2.plot(YR, bt.intercept + bt.slope*YR, lw=2.0, color="#111111", zorder=6,
        label=f"all  {bt.slope*47:+.1f} over 47 yr  (p = {bt.pvalue:.3f})")
a2.set_xlabel("season", fontsize=10)
a2.set_ylabel("landfalling storms per season", fontsize=10)
a2.set_xlim(1976, 2024)
a2.legend(fontsize=8.0, framealpha=0.95, edgecolor="#999999", loc="upper right", ncol=1)
a2.text(0.012, 1.02, "(b)", transform=a2.transAxes, fontsize=11, fontweight="bold", va="bottom")
axes(a2)

fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"{A.out}.{e}", dpi=200, bbox_inches="tight", facecolor="white")
print("observed share  :", {g: round(obs_share[g],1) for g in GROUPS})
print("SynTC share     :", {g: f"{S.mean(0)[i]:.1f} +- {S.std(0,ddof=1)[i]:.1f}" for i,g in enumerate(GROUPS)})
print(f"all-group trend : {bt.slope:+.4f}/yr  p = {bt.pvalue:.3f}  Kendall p = {pk:.3f}")
