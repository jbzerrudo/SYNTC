"""
fig_area_of_interest.pdf: Figure 1, the analysis domain.

    python make_aoi.py --dtm dtm_phil_1km.tif

Replaces AOI_v2.png, an 18.6 MB ArcGIS raster, with a vector PDF reproducible
from one input the paper already ships.

Four things, and nothing else
-----------------------------
  the Philippine map          from the terrain model's own land mask
  its elevation               the same 1 km array the decay relation reads
  the PAR boundary            dashed, PAGASA's six-sided polygon
  the six vertices            ringed and labelled with their coordinates

No storm tracks. The earlier version drew all 1,188 observed tracks as white
lines; at that density no single storm can be followed and no density can be
read off, so they obscured the terrain and the boundary without adding a
quantity. Track density is reported properly, with a colour scale and a skill
score against a bootstrap noise floor, in the hotspot figures.

No land outside the Philippines. The terrain model covers the Philippines only,
so drawing Taiwan or Borneo would imply a terrain treatment that does not
exist. The limitations section says storms crossing Taiwan are not attenuated
for exactly that reason, and the figure should not contradict it.

The coastline is the boundary of the terrain model's valid-data mask, so the
outline and the shading cannot disagree with each other, or with the array the
generator reads at run time.

Aspect is about 1.6, so the figure fits upright at \\textwidth with room for
text on the same page. The version it replaces was 1.29 and could only be set
as a full-page sideways float, which is what pushed an orphan page into the
compiled manuscript.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource, LinearSegmentedColormap

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dtm", default="dtm_phil_1km.tif")
ap.add_argument("--out", default=".")
ap.add_argument("--stem", default="fig_area_of_interest")
A = ap.parse_args()

# PAGASA's PAR hexagon as (lat, lon), matching syntc_ai.CONFIG.par_vertices and
# the coordinates quoted in the caption.
PAR = [(25.0, 120.0), (25.0, 135.0), (5.0, 135.0),
       (5.0, 115.0), (15.0, 115.0), (21.0, 120.0)]
CENTRE = (12.375, 121.5)                       # lat, lon
EXTENT = (113.0, 137.5, 3.0, 27.5)             # W, E, S, N

SEA = "#eaf0f4"
COAST = "#1a1a1a"
TERRAIN = LinearSegmentedColormap.from_list(
    "phl_terrain",
    ["#e8dcc0", "#d8c39a", "#c0a173", "#a07c4f", "#7a5733", "#54371e"])

# ----------------------------------------------------------------------- data
r = rasterio.open(A.dtm)
dem = np.ma.masked_less_equal(r.read(1, masked=True), 0.5)   # SynTC land test: elevation > 0.5 m, dem = np.ma.masked_less(r.read(1, masked=True), 0)
b = r.bounds
print(f"terrain model {r.width} x {r.height}, {b.left:.2f}-{b.right:.2f}E "
      f"{b.bottom:.2f}-{b.top:.2f}N, max {dem.max():.0f} m")

# One border ring of sea, so a coastline reaching the raster edge closes
# instead of tracing the rectangle.
land = np.pad((~np.ma.getmaskarray(dem)).astype(float), 1, constant_values=0.0)
lon = b.left + (np.arange(r.width) + 0.5) * r.res[0] #lon = np.linspace(b.left, b.right, r.width)
lat = b.top - (np.arange(r.height) + 0.5) * r.res[1] #lat = np.linspace(b.top, b.bottom, r.height)
dx, dy = lon[1] - lon[0], lat[0] - lat[1]
lon_p = np.concatenate(([lon[0] - dx], lon, [lon[-1] + dx]))
lat_p = np.concatenate(([lat[0] + dy], lat, [lat[-1] - dy]))

# --------------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(8.6, 5.4), facecolor="white")
ax.set_facecolor(SEA)
ax.set_xlim(EXTENT[0], EXTENT[1])
ax.set_ylim(EXTENT[2], EXTENT[3])
ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(EXTENT[2:]))))

ls = LightSource(azdeg=315, altdeg=45)
shade = ls.shade(np.ma.filled(dem, 0).astype(float), cmap=TERRAIN,
                 blend_mode="soft", vert_exag=60, dx=1000, dy=1000,
                 vmin=0, vmax=2840)
shade[..., 3] = np.where(np.ma.getmaskarray(dem), 0.0, 1.0)
ax.imshow(shade, extent=(b.left, b.right, b.bottom, b.top), origin="upper",
          zorder=3, interpolation="bilinear")
ax.contour(lon_p, lat_p, land, levels=[0.5], colors=COAST, linewidths=0.5,
           zorder=4)

# PAR boundary and its vertices
ring = np.array([(lo, la) for la, lo in PAR] + [(PAR[0][1], PAR[0][0])])
ax.plot(ring[:, 0], ring[:, 1], color="#111111", lw=1.6, ls=(0, (6, 3)),
        zorder=6)
ax.plot(ring[:-1, 0], ring[:-1, 1], "o", ms=6.5, mfc="white", mec="#111111",
        mew=1.4, zorder=7)
LABEL = {(25.0, 120.0): (0.6, -0.8, "left", "top"),
         (25.0, 135.0): (-0.6, -0.8, "right", "top"),
         (5.0, 135.0): (-0.6, 0.8, "right", "bottom"),
         (5.0, 115.0): (0.6, 0.8, "left", "bottom"),
         (15.0, 115.0): (0.7, 0.0, "left", "center"),
         (21.0, 120.0): (0.7, 0.0, "left", "center")}
for la, lo in PAR:
    ddx, ddy, ha, va = LABEL[(la, lo)]
    ax.text(lo + ddx, la + ddy, f"{lo:.0f}°E, {la:.0f}°N",
            fontsize=9, ha=ha, va=va, color="#111111", zorder=8,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.6))
ax.text(133.4, 15.0, "PAR", fontsize=11, color="#111111", fontweight="bold",
        ha="right", va="center", zorder=8,
        bbox=dict(fc="white", ec="none", alpha=0.8, pad=2.0))

# geographic centre
ax.plot(CENTRE[1], CENTRE[0], marker="P", ms=11, color="#c81f1f",
        mec="white", mew=1.0, zorder=9)
ax.annotate(f"{CENTRE[1]}°E, {CENTRE[0]}°N", (CENTRE[1], CENTRE[0]),
            xytext=(0, -14), textcoords="offset points", fontsize=9,
            ha="center", va="top", color="#c81f1f", fontweight="bold",
            zorder=9, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.6))

xt = np.arange(115, 136, 5)
yt = np.arange(5, 26, 5)
ax.set_xticks(xt)
ax.set_yticks(yt)
ax.set_xticklabels([f"{v}°E" for v in xt])
ax.set_yticklabels([f"{v}°N" for v in yt])
ax.grid(color="#b9c4cc", alpha=0.55, lw=0.5, zorder=1)
ax.tick_params(labelsize=9.5, length=3, width=0.8)
for s in ax.spines.values():
    s.set_color("#333333")
    s.set_linewidth(0.9)

sm = plt.cm.ScalarMappable(cmap=TERRAIN, norm=plt.Normalize(vmin=0, vmax=2840))
cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.015, aspect=25)
cb.set_label("elevation (m)", fontsize=9.5)
cb.ax.tick_params(labelsize=8.5)
cb.outline.set_linewidth(0.7)

fig.tight_layout()
for ext in ("pdf", "png"):
    p = os.path.join(A.out, f"{A.stem}.{ext}")
    fig.savefig(p, dpi=250, bbox_inches="tight", facecolor="white")
    print("written:", p, "%.2f MB" % (os.path.getsize(p) / 1e6))
plt.close(fig)
