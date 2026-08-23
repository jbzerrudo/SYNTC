"""
A storm forms here, in this month. Where does it probably go?

    python genesis_forecast.py --model ./run03/model.pkl \
        --dtm /path/to/dtm_phil_1km.tif --lat 13 --lon 132 --month 10

This is the tool form of SynTC. The catalogue answers "what does a century of
storms look like"; this answers the operational question instead, which is what
follows from one genesis event at a known place and time of year.

It works by seeding the same trained propagator that built the catalogue with a
fixed position and month, then drawing n independent realisations. Nothing is
refitted: the model is loaded from the pickle the generation run wrote, in about
a hundredth of a second, so the paper and the tool are provably using one fitted
model rather than two fits that happened to share a seed.

What comes out
--------------
  <stem>_tracks.csv     every simulated track, for your own analysis
  <stem>_passage.csv    probability of passage per grid cell
  <stem>.png            the plume, with the PAR hexagon and the tracks behind it

and a printed summary: the chance of entering PAR, of a Philippine landfall and
where, how long until PAR entry, and the peak intensity distribution.

Probability of passage
----------------------
For each grid cell, the fraction of realisations whose track passes through it.
Tracks are densified to roughly 25 km before gridding, because a storm moving at
25 km/h covers about 150 km in one 6-hourly step and would otherwise skip cells
it plainly travelled through, punching holes in the field that are an artefact
of the output cadence rather than anything physical.

The number is a probability CONDITIONAL on a storm forming at that point in that
month. It is not the probability that such a storm forms. Multiply by your own
genesis rate if you want an absolute risk.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import terrain
import figstyle as FS

# Rough island-group split by latitude. Crude, but stated: the alternative is
# shipping administrative polygons the rest of the code does not need.
GROUPS = (("Luzon", 14.5, 90.0), ("Visayas", 9.5, 14.5), ("Mindanao", -90.0, 9.5))
MONTHS = ("", "January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")
# ColorBrewer YlOrRd, the same ramp as the hotspot maps and the ArcGIS
# products, so every density field in the paper is read the same way. White at
# the bottom so a cell no storm reached is background rather than pale yellow.
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#ffffff", "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
            "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"])
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#dcdbd6"


def densify(lat, lon, step_km=25.0):
    """Insert points along each leg so gridding cannot leave holes."""
    out_lat, out_lon = [lat[0]], [lon[0]]
    for i in range(1, len(lat)):
        dy = (lat[i] - lat[i-1]) * 111.32
        dx = (lon[i] - lon[i-1]) * 111.32 * np.cos(np.radians(lat[i-1]))
        n = max(1, int(np.hypot(dx, dy) / step_km))
        for k in range(1, n + 1):
            out_lat.append(lat[i-1] + (lat[i] - lat[i-1]) * k / n)
            out_lon.append(lon[i-1] + (lon[i] - lon[i-1]) * k / n)
    return np.array(out_lat), np.array(out_lon)


def passage_probability(df, lon_edges, lat_edges):
    """Fraction of realisations whose track passes through each cell.

    Counted once per realisation per cell. A storm that loiters does not get to
    inflate its own probability, which is the difference between a probability
    of passage and a track density.
    """
    nlon, nlat = len(lon_edges) - 1, len(lat_edges) - 1
    hit = np.zeros((nlon, nlat), dtype=np.int32)
    n = 0
    for _, t in df.groupby("SID", sort=False):
        t = t.sort_values("STEP")
        la, lo = densify(t.LAT.to_numpy(), t.LON.to_numpy())
        h, _, _ = np.histogram2d(lo, la, bins=[lon_edges, lat_edges])
        hit += (h > 0).astype(np.int32)
        n += 1
    return hit / max(n, 1), n


def summarise(df, seen_par, crossed, lat0, lon0, month, dtm):
    n = len(seen_par)
    peak = df.groupby("SID").WIND.max()
    print(f"\nA storm forming at {lat0:.1f}N {lon0:.1f}E in {MONTHS[month]}, "
          f"{n:,} realisations\n")
    print(f"  enters PAR                     {100*seen_par.mean():5.1f}%")
    print(f"  centre crosses Philippine land {100*crossed.mean():5.1f}%")

    par = df[df.IN_PAR == 1]
    if len(par):
        first = par.sort_values("STEP").groupby("SID").head(1)
        h = first.STEP.to_numpy() * 6.0
        print(f"  time from genesis to PAR entry {np.median(h):5.0f} h "
              f"(10-90%: {np.percentile(h,10):.0f}-{np.percentile(h,90):.0f} h)")
        pk = par.groupby("SID").WIND.max()
        print(f"  peak wind while inside PAR     {pk.median():5.0f} kt "
              f"(10-90%: {np.percentile(pk,10):.0f}-{np.percentile(pk,90):.0f} kt)")

    print(f"\n  peak wind, whole life          {peak.median():5.0f} kt "
          f"(10-90%: {np.percentile(peak,10):.0f}-{np.percentile(peak,90):.0f} kt)")
    for name, lo, hi in (("reaches TY (>=64 kt)", 64, 1e9),
                         ("reaches STY (>=100 kt)", 100, 1e9)):
        print(f"  {name:<30} {100*np.mean((peak>=lo)&(peak<hi)):5.1f}%")

    land = df[df.OVER_LAND == 1]
    if len(land):
        firstland = land.sort_values("STEP").groupby("SID").head(1)
        print("\n  landfall by island group (share of all realisations)")
        for name, lo, hi in GROUPS:
            k = ((firstland.LAT >= lo) & (firstland.LAT < hi)).sum()
            print(f"    {name:<10} {100*k/n:5.1f}%")
        w = land.groupby("SID").WIND.max()
        print(f"  strongest wind while over land  {w.median():5.0f} kt median, "
              f"{w.max():.0f} kt worst case")


def land_outline(ax, dtm, zorder=3):
    """Draw the Philippine coastline from the DTM.

    Without it the plume floats on an empty grid and a reader cannot see what
    the storms are hitting. The DTM covers only the Philippines, which is
    exactly the coastline this figure needs. Subsampled because a 1 km raster
    contoured at full resolution costs seconds and renders identically at this
    scale.
    """
    try:
        tx = terrain.get(dtm)
    except Exception:
        return
    s = 10
    land = tx.is_land[::s, ::s]
    ny, nx = land.shape
    lon = tx.left + (np.arange(nx) + 0.5) * s * tx.transform.a
    lat = tx.top + (np.arange(ny) + 0.5) * s * tx.transform.e
    ax.contour(lon, lat, land.astype(float), levels=[0.5], colors="#52514e",
               linewidths=0.6, zorder=zorder)


def central_path(df, mode="median", min_frac=0.5):
    """A single line through the middle of the plume, or (None, None).

    median is a per-step statistic over every surviving realisation, so it is
    derived from the plume itself rather than from one selected track. That is
    what makes it the middle of the field, and also what makes it dangerous:
    where the plume splits into a westward branch and a recurving one, the
    median position sits between them, in water no realisation ever entered.
    Off by default for that reason.

    medoid returns a real realisation instead, the one whose own path stays
    closest to that median path, so it is always a track a storm could follow.

    The line stops once fewer than min_frac of the realisations are still
    alive, since a median over the surviving few is a median over the
    longest-lived storms rather than over the ensemble.
    """
    if mode == "off":
        return None, None
    n0 = df.SID.nunique()
    steps, mla, mlo = [], [], []
    for step, g in df.groupby("STEP"):
        if len(g) < max(20, min_frac * n0):
            continue
        steps.append(step)
        mla.append(float(g.LAT.median()))
        mlo.append(float(g.LON.median()))
    if len(steps) < 4:
        return None, None
    mla, mlo = np.array(mla), np.array(mlo)
    if mode == "median":
        return mlo, mla
    ref = dict(zip(steps, zip(mla, mlo)))
    best, best_d = None, np.inf
    for sid, t in df.groupby("SID", sort=False):
        t = t[t.STEP.isin(ref)]
        if len(t) < 0.8 * len(steps):
            continue
        a = np.array([ref[st] for st in t.STEP])
        dy = (t.LAT.to_numpy() - a[:, 0]) * 111.32
        dx = (t.LON.to_numpy() - a[:, 1]) * 111.32 * np.cos(np.radians(a[:, 0]))
        d = float(np.hypot(dx, dy).mean())
        if d < best_d:
            best, best_d = sid, d
    if best is None:
        return mlo, mla
    t = df[df.SID == best].sort_values("STEP")
    return t.LON.to_numpy(), t.LAT.to_numpy()


def figure(prob, lon_edges, lat_edges, df, lat0, lon0, month, n, path, verts,
           dtm, central="off", tracks=60):
    fig, ax = plt.subplots(figsize=(7.6, 6.6), facecolor="white")
    m = np.ma.masked_where(prob.T <= 0, prob.T)
    # The genesis cell is 1.0 by construction and a linear ramp spends the whole
    # colour range on it, leaving the plume nearly white. A power norm keeps the
    # axis honest, 0 to 1 with nothing clipped, while giving the low
    # probabilities that carry the actual information somewhere to live.
    pc = ax.pcolormesh(lon_edges, lat_edges, m, cmap=SEQ, shading="flat",
                       norm=matplotlib.colors.PowerNorm(0.45, vmin=0, vmax=1),
                       zorder=1)
    if tracks > 0:
        ids = df.SID.unique()
        for sid in ids[:: max(1, len(ids) // tracks)]:
            t = df[df.SID == sid].sort_values("STEP")
            ax.plot(t.LON, t.LAT, color="#0b0b0b", lw=0.3, alpha=0.13,
                    zorder=2)
    land_outline(ax, dtm)
    v = np.array(verts + (verts[0],))
    ax.plot(v[:, 1], v[:, 0], color="#0b3a5c", lw=1.3, ls="--", zorder=5,
            label="PAR")
    cx, cy = central_path(df, central)
    if cx is not None:
        # Same 0.75 weight as the realisations. The white stroke is what makes
        # it readable over the dark cells, not extra line width.
        ax.plot(cx, cy, color="#08306b", lw=0.75, zorder=5.5,
                solid_capstyle="round",
                path_effects=[pe.withStroke(linewidth=2.05, foreground="white",
                                            alpha=0.85)],
                label="middle of the plume" if central == "median"
                      else "most representative track")

    # Genesis marker: a small ringed dot with a crosshair, not a star. The
    # point being marked is a coordinate, and a 17-point star covers about two
    # grid cells of the field it is sitting on, hiding the highest-probability
    # cells in the figure.
    ax.plot([lon0], [lat0], marker=FS.tc_marker(), ms=26, mfc="none",
            mec="#0b3a5c", mew=1.35, zorder=6, linestyle="none",
            label="genesis")
    ax.plot([lon0], [lat0], marker="o", ms=2.6, color="#0b3a5c", zorder=6.1,
            linestyle="none")
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ax.plot([lon0 + dx * 1.1, lon0 + dx * 2.6],
                [lat0 + dy * 1.1, lat0 + dy * 2.6],
                color="#0b3a5c", lw=1.0, zorder=6, solid_capstyle="butt")
    cb = fig.colorbar(pc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("probability the storm passes through this cell\n"
             "(colour scale is power-stretched, values are unclipped)",
             fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel("longitude (E)", fontsize=10)
    ax.set_ylabel("latitude (N)", fontsize=10)
    if FS.TITLES:
        ax.set_title(f"Genesis at {lat0:.1f}N {lon0:.1f}E in {MONTHS[month]}\n"
                     f"{n:,} SynTC realisations", fontsize=12, color=INK,
                     loc="left")
    ax.legend(fontsize=8.5, loc="upper right", frameon=True, edgecolor=LINE)
    ax.grid(lw=0.4, color="#eeeeee", zorder=0)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model.pkl from a run folder")
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--month", type=int, required=True, help="1-12")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--wind", type=float, default=None,
                    help="genesis intensity in kt; omit to draw it from the "
                         "genesis model, which is the climatological choice")
    ap.add_argument("--grid", type=float, default=1.0, help="cell size, degrees")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=".")
    ap.add_argument("--tracks", type=int, default=60,
                    help="how many realisations to draw over the field. "
                         "0 shows the probability field alone, which is the "
                         "honest object: the field carries the uncertainty, "
                         "the drawn lines are only a sample of it.")
    ap.add_argument("--central", default="off",
                    choices=("off", "median", "medoid"),
                    help="draw a blue line through the middle of the plume. "
                         "median is a per-step statistic over all surviving "
                         "realisations; medoid is the single realisation "
                         "closest to it. Off by default because a median "
                         "position is misleading where the plume is bimodal.")
    ap.add_argument("--titles", action="store_true",
                    help="draw the title into the image; off by default so the "
                         "LaTeX caption is the only caption")
    a = ap.parse_args()
    FS.TITLES = a.titles
    if not 1 <= a.month <= 12:
        raise SystemExit("--month must be 1-12")

    terrain.DTM_PATH = a.dtm
    from syntc_ai import load_model, CONFIG
    model = load_model(a.model)

    df, seen_par, crossed = model.simulate_from_genesis(
        lat0=a.lat, lon0=a.lon, month=a.month, n=a.n, wind0=a.wind,
        seed=a.seed, dtm=a.dtm)
    if not len(df):
        raise SystemExit("no storms survived the first step; check the seed")

    summarise(df, seen_par, crossed, a.lat, a.lon, a.month, a.dtm)

    cfg = model.cfg
    lon_edges = np.arange(cfg.lon_min, cfg.lon_max + a.grid, a.grid)
    lat_edges = np.arange(cfg.lat_min, cfg.lat_max + a.grid, a.grid)
    prob, n_used = passage_probability(df, lon_edges, lat_edges)

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"genesis_{a.lat:g}N_{a.lon:g}E_m{a.month:02d}")
    df.to_csv(f"{stem}_tracks.csv", index=False)
    ii, jj = np.nonzero(prob)
    pd.DataFrame({"lon": lon_edges[ii] + a.grid / 2,
                  "lat": lat_edges[jj] + a.grid / 2,
                  "probability": prob[ii, jj]}).to_csv(
        f"{stem}_passage.csv", index=False)
    figure(prob, lon_edges, lat_edges, df, a.lat, a.lon, a.month, n_used,
           f"{stem}.png", cfg.par_vertices, a.dtm,
           central=a.central, tracks=a.tracks)
    print(f"\nwritten:\n  {stem}.png\n  {stem}.pdf\n  {stem}_tracks.csv"
          f"\n  {stem}_passage.csv")
    print("\nThese are probabilities GIVEN a storm forms at that point in that "
          "month.\nThey are not the probability that one forms.")


if __name__ == "__main__":
    main()
