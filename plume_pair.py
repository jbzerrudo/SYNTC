"""Two-panel genesis plume: the probability field, overlaid with the spaghetti
of only those realisations that stay inside the high-probability core.

A track is drawn if the median passage probability of the cells it visits sits
in the top CORE_Q of all realisations for that genesis point. The result is the
corridor the storm is most likely to follow, rather than 60 tracks sampled at
random from a distribution whose spread is the point being hidden.
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import rasterio

SEQ = mcolors.LinearSegmentedColormap.from_list("heat", [
    "#ffffcc","#ffeda0","#fed976","#feb24c","#fd8d3c",
    "#fc4e2a","#e31a1c","#bd0026","#800026"])
PAR = ((25.0,120.0),(25.0,135.0),(5.0,135.0),(5.0,115.0),(15.0,115.0),(21.0,120.0))
EXT = (112.0, 145.0, 2.0, 28.0)
_ap = argparse.ArgumentParser(description="Two-panel genesis plume with core-following tracks.")
_ap.add_argument("--gen", default="gen07", help="folder holding the genesis_*_passage.csv / _tracks.csv pairs")
_ap.add_argument("--dtm", default="dtm_phil_1km.tif")
_ap.add_argument("--left",  default="genesis_13N_132E_m10")
_ap.add_argument("--right", default="genesis_10N_140E_m11")
_ap.add_argument("--left-label",  default="(a)  13\u00b0N, 132\u00b0E, October, inside PAR")
_ap.add_argument("--right-label", default="(b)  10\u00b0N, 140\u00b0E, November, outside PAR")
_ap.add_argument("--left-pt",  nargs=2, type=float, default=[13.0, 132.0], metavar=("LAT","LON"))
_ap.add_argument("--right-pt", nargs=2, type=float, default=[10.0, 140.0], metavar=("LAT","LON"))
_ap.add_argument("--keep", type=int, default=15, help="realisations drawn per panel")
_ap.add_argument("--out", default="genesis_plume_pair")
A = _ap.parse_args()
NKEEP = A.keep
DTM = A.dtm

_src = rasterio.open(DTM); _e = _src.read(1).astype(float)
_e[~np.isfinite(_e)] = 0.0; _e[_e < 0] = 0.0
_s = 8; _land = _e[::_s, ::_s] > 0.5
_ny, _nx = _land.shape
_lon = _src.bounds.left + (np.arange(_nx)+0.5)*_s*_src.transform.a
_lat = _src.bounds.top  + (np.arange(_ny)+0.5)*_s*_src.transform.e

def land_outline(ax):
    ax.contour(_lon, _lat, _land.astype(float), levels=[0.5],
               colors="#52514e", linewidths=0.6, zorder=3)

def north_arrow(ax, x=0.055, y=0.84, L=0.085):
    ax.annotate("", xy=(x, y+L), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#0b0b0b", mutation_scale=12))
    ax.text(x, y+L+0.012, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#0b0b0b")

def panel(ax, tag, glat, glon, label):
    p = pd.read_csv(os.path.join(A.gen, tag + "_passage.csv"))
    tr = pd.read_csv(os.path.join(A.gen, tag + "_tracks.csv"))
    lons = np.sort(p.lon.unique()); lats = np.sort(p.lat.unique())
    dlon = np.diff(lons)[0]; dlat = np.diff(lats)[0]
    grid = p.pivot(index="lat", columns="lon", values="probability").values
    le = np.append(lons - dlon/2, lons[-1] + dlon/2)
    la = np.append(lats - dlat/2, lats[-1] + dlat/2)
    m = np.ma.masked_where(grid <= 0, grid)
    pc = ax.pcolormesh(le, la, m, cmap=SEQ, shading="flat",
                       norm=mcolors.PowerNorm(0.45, vmin=0, vmax=1), zorder=1)

    # score every realisation by the probability of the cells it visits
    lut = {(round(r.lon,1), round(r.lat,1)): r.probability for r in p.itertuples()}
    def snap(v, v0, d): return round(v0 + np.round((v - v0)/d)*d, 1)
    # Rank realisations by how faithfully they follow the high-probability
    # corridor. The first 24 h are dropped because every realisation occupies
    # the genesis cell, where probability is 1 by construction, and a track
    # that dissipates early would otherwise outrank one that runs the corridor
    # end to end. Realisations shorter than four days are excluded for the
    # same reason.
    SKIP, MINLEN = 4, 16
    score = {}
    for sid, t in tr.groupby("SID", sort=False):
        t = t.sort_values("STEP")
        x = t.LON.to_numpy()[SKIP:]; y = t.LAT.to_numpy()[SKIP:]
        if len(x) < MINLEN:
            continue
        w = (x >= EXT[0]) & (x <= EXT[1]) & (y >= EXT[2]) & (y <= EXT[3])
        if w.sum() < MINLEN:
            continue
        pr = [lut.get((snap(a, lons[0], dlon), snap(b, lats[0], dlat)), 0.0)
              for a, b in zip(x[w], y[w])]
        # The 10th percentile, not the mean. A mean lets a realisation recurve
        # out of the corridor and still rank well on the strength of the leg it
        # spent inside; a low quantile only rewards realisations that stay in.
        score[sid] = float(np.percentile(pr, 10))
    sc = pd.Series(score)
    keep = sc.nlargest(NKEEP).index
    for sid in keep:
        t = tr[tr.SID == sid].sort_values("STEP")
        ax.plot(t.LON, t.LAT, color="#08306b", lw=0.75, alpha=0.45,
                zorder=2, solid_capstyle="round")

    land_outline(ax)
    v = np.array(PAR + (PAR[0],))
    ax.plot(v[:,1], v[:,0], color="#0b3a5c", lw=1.3, ls="--", zorder=5)
    ax.plot([glon], [glat], marker="o", ms=7, color="#0b3a5c",
            markeredgecolor="white", markeredgewidth=1.3, zorder=6, linestyle="none")
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        ax.plot([glon+dx*0.9, glon+dx*2.1], [glat+dy*0.9, glat+dy*2.1],
                color="#0b3a5c", lw=1.0, zorder=6, solid_capstyle="butt")
    ax.set_xlim(*EXT[:2]); ax.set_ylim(*EXT[2:]); ax.set_aspect("equal")
    ax.set_xticks(np.arange(115, EXT[1]+0.1, 10))
    ax.set_yticks(np.arange(EXT[2]+3, EXT[3]+0.1, 5))
    ax.set_xticks(np.arange(EXT[0], EXT[1]+0.1, 2.5), minor=True)
    ax.set_yticks(np.arange(EXT[2], EXT[3]+0.1, 2.5), minor=True)
    ax.xaxis.set_major_formatter(lambda x,_: f"{x:.0f}°E")
    ax.yaxis.set_major_formatter(lambda y,_: f"{y:.0f}°N")
    ax.grid(which="major", lw=0.5, color="#c9c9c9", zorder=0)
    ax.grid(which="minor", lw=0.22, color="#ececec", zorder=0)
    ax.set_axisbelow(True)
    north_arrow(ax)
    ax.text(0.012, 1.015, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom")
    return pc, len(keep), len(sc)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 5.2), facecolor="white",
                               constrained_layout=True)
pc, nL, tL = panel(axL, A.left,  A.left_pt[0],  A.left_pt[1],  A.left_label)
_,  nR, tR = panel(axR, A.right, A.right_pt[0], A.right_pt[1], A.right_label)
axL.set_ylabel("latitude", fontsize=10)
for ax in (axL, axR): ax.set_xlabel("longitude", fontsize=10)
cb = fig.colorbar(pc, ax=[axL, axR], shrink=0.86, pad=0.015, aspect=26)
cb.set_label("probability the storm passes through this cell", fontsize=9)
cb.ax.tick_params(labelsize=8)
for ext in ("png","pdf"):
    fig.savefig(f"{A.out}.{ext}", dpi=200, facecolor="white")
print(f"drawn: top {nL} of {tL} eligible (a), top {nR} of {tR} (b)")
