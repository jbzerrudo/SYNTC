"""Redraw the genesis plume on a PAR-focused window, with a north arrow and a
visible graticule. Reads the CSVs the tool already wrote, so the figure comes
from the same 2,000 realisations as before and nothing is re-simulated.

    python replot_plume.py --run tool --lat 13 --lon 132 --month 10
"""
import argparse, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio

SEQ = mcolors.LinearSegmentedColormap.from_list("heat", [
    "#ffffcc","#ffeda0","#fed976","#feb24c","#fd8d3c",
    "#fc4e2a","#e31a1c","#bd0026","#800026"])
PAR = ((25.0,120.0),(25.0,135.0),(5.0,135.0),(5.0,115.0),(15.0,115.0),(21.0,120.0))
EXT = (112.0, 140.0, 2.0, 28.0)          # PAR plus a 5 degree margin
NTRACK = 60                              # track lines drawn (300 suits --style spaghetti)

def land_outline(ax, dtm, zorder=3):
    src = rasterio.open(dtm); e = src.read(1).astype(float)
    e[~np.isfinite(e)] = 0.0; e[e < 0] = 0.0
    s = 8; land = e[::s, ::s] > 0.5
    ny, nx = land.shape
    lon = src.bounds.left + (np.arange(nx)+0.5)*s*src.transform.a
    lat = src.bounds.top  + (np.arange(ny)+0.5)*s*src.transform.e
    ax.contour(lon, lat, land.astype(float), levels=[0.5],
               colors="#52514e", linewidths=0.6, zorder=zorder)

def north_arrow(ax, x=0.055, y=0.86, L=0.085):
    ax.annotate("", xy=(x, y+L), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#0b0b0b",
                                mutation_scale=13))
    ax.text(x, y+L+0.012, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=10.5, fontweight="bold", color="#0b0b0b")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="tool")
    ap.add_argument("--lat", type=float, default=13.0)
    ap.add_argument("--lon", type=float, default=132.0)
    ap.add_argument("--month", type=int, default=10)
    ap.add_argument("--dtm", default="syntc-ai/release/syntc/dtm_phil_1km.tif")
    ap.add_argument("--out", default="genesis_plume_13N_132E_oct_PAR")
    ap.add_argument("--style", choices=("spaghetti","field"), default="spaghetti")
    a = ap.parse_args()
    stem = f"{a.run}/genesis_{int(a.lat)}N_{int(a.lon)}E_m{a.month:02d}"
    p  = pd.read_csv(stem + "_passage.csv")
    tr = pd.read_csv(stem + "_tracks.csv")

    lons = np.sort(p.lon.unique()); lats = np.sort(p.lat.unique())
    dlon = np.diff(lons)[0]; dlat = np.diff(lats)[0]
    grid = p.pivot(index="lat", columns="lon", values="probability").values
    le = np.append(lons - dlon/2, lons[-1] + dlon/2)
    la = np.append(lats - dlat/2, lats[-1] + dlat/2)

    fig, ax = plt.subplots(figsize=(7.4, 6.4), facecolor="white")
    pc = None
    if a.style == "field":
        m = np.ma.masked_where(grid <= 0, grid)
        pc = ax.pcolormesh(le, la, m, cmap=SEQ, shading="flat",
                           norm=mcolors.PowerNorm(0.45, vmin=0, vmax=1), zorder=1)
    ids = tr.SID.unique()
    step = max(1, len(ids)//NTRACK)
    for k, sid in enumerate(ids[::step]):
        t = tr[tr.SID == sid].sort_values("STEP")
        ax.plot(t.LON, t.LAT,
                color="#0b0b0b" if a.style == "field" else "#c00000",
                lw=0.30 if a.style == "field" else 0.6,
                alpha=0.13, zorder=2 if a.style == "field" else 4,
                solid_capstyle="round",
                label="SynTC realisations" if k == 0 else None)
    land_outline(ax, a.dtm)
    v = np.array(PAR + (PAR[0],))
    ax.plot(v[:,1], v[:,0], color="#0b3a5c", lw=1.3, ls="--", zorder=5, label="PAR")
    ax.plot([a.lon], [a.lat], marker="o", ms=6.5, color="#0b3a5c",
            markeredgecolor="white", markeredgewidth=1.2, zorder=6,
            linestyle="none", label="genesis")
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        ax.plot([a.lon+dx*0.9, a.lon+dx*2.1], [a.lat+dy*0.9, a.lat+dy*2.1],
                color="#0b3a5c", lw=1.0, zorder=6, solid_capstyle="butt")

    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_xticks(np.arange(EXT[0], EXT[1]+0.1, 5))
    ax.set_yticks(np.arange(EXT[2], EXT[3]+0.1, 5))
    ax.set_xticks(np.arange(EXT[0], EXT[1]+0.1, 1), minor=True)
    ax.set_yticks(np.arange(EXT[2], EXT[3]+0.1, 1), minor=True)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}°E")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}°N")
    ax.grid(which="major", lw=0.55, color="#c9c9c9", zorder=0)
    ax.grid(which="minor", lw=0.25, color="#ececec", zorder=0)
    ax.set_axisbelow(True); ax.set_aspect("equal")
    north_arrow(ax)
    if pc is not None:
        cb = fig.colorbar(pc, ax=ax, shrink=0.80, pad=0.02)
        cb.set_label("probability the storm passes through this cell\n"
                     "(colour scale is power-stretched, values are unclipped)", fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
    ax.set_xlabel("longitude", fontsize=10); ax.set_ylabel("latitude", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right", frameon=True, edgecolor="#52514e")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(a.out + ".pdf", bbox_inches="tight", facecolor="white")
    inside = p[(p.lon.between(EXT[0],EXT[1])) & (p.lat.between(EXT[2],EXT[3]))]
    print("window %g-%gE %g-%gN keeps %.1f%% of the passage probability mass"
          % (*EXT, 100*inside.probability.sum()/p.probability.sum()))

if __name__ == "__main__":
    main()