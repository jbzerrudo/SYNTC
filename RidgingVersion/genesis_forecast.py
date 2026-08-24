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
import sys

# This file lives in SYNTC/RidgingVersion/, a separate folder kept apart from
# the paper's method files at the repo root. terrain.py, figstyle.py and the
# model live one level up, so put the parent on the path when running from
# source. A frozen SynTC.exe bundles those modules alongside this script, so it
# must NOT repoint the path outward.
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
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


def _in_par(lon, lat, verts):
    """Point in the PAR polygon. verts are (lat, lon) pairs, as cfg stores them."""
    inside = False
    n = len(verts)
    for i in range(n):
        y1, x1 = verts[i]
        y2, x2 = verts[(i + 1) % n]
        if (x1 > lon) != (x2 > lon):
            if lat < y1 + (lon - x1) * (y2 - y1) / (x2 - x1):
                inside = not inside
    return inside


def summarise(df, seen_par, crossed, lat0, lon0, month, dtm,
              ridge_xy=None, verts=None):
    n = len(seen_par)
    peak = df.groupby("SID").WIND.max()
    print(f"\nA storm forming at {lat0:.1f}N {lon0:.1f}E in {MONTHS[month]}, "
          f"{n:,} realisations\n")
    print(f"  enters PAR                     {100*seen_par.mean():5.1f}%")
    # Whether the ridge enters PAR is a property of one line, so it is a yes or
    # a no, not a percentage. Printing it as "100%" beside the figure above
    # would put two different kinds of quantity in the same column and invite
    # a reader to treat the crest as a forecast probability. The entry and exit
    # coordinates are given so the claim can be checked against the map.
    if ridge_xy is not None and ridge_xy[0] is not None and verts is not None:
        rx = np.asarray(ridge_xy[0]); ry = np.asarray(ridge_xy[1])
        ins = [i for i in range(len(rx)) if _in_par(rx[i], ry[i], verts)]
        if ins:
            print(f"  ridge (most likely path) enters PAR: yes, "
                  f"{rx[ins[0]]:.1f}E {ry[ins[0]]:.1f}N to "
                  f"{rx[ins[-1]]:.1f}E {ry[ins[-1]]:.1f}N")
        else:
            print(f"  ridge (most likely path) enters PAR: no")
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


def ridge_path(prob, lon_edges, lat_edges, lon0, lat0, ring=110.0,
               reach=3500.0, q=0.90, floor=0.02, leash=2.5, smooth=3,
               anchor=True):
    """The crest of the passage field, traced outward from genesis.

    Walk out from the genesis point in rings of great-circle distance. Within
    each ring take the cells in the top decile of passage probability and
    average their coordinates, weighted by probability. That is where the
    realisations actually concentrate at that range.

    This is what a per-step median cannot do. The median takes the median
    latitude and the median longitude at each 6-hourly step, so once a majority
    of realisations recurve it sits in the recurving branch, even where the
    westward branch is the denser corridor. A narrow, well populated westward
    channel loses to a broad recurving fan on a per-step head count while
    beating it comfortably on probability per cell. The ridge follows the
    probability; the median follows the head count.

    Rings are indexed by distance alone, so nothing in that construction stops
    the crest of ring k+1 from lying in a different branch of the plume from
    the crest of ring k. Unconstrained it jumps: on a 13N 132E October field it
    steps 2,585 km across the archipelago from the westward branch to the
    recurving one. The leash fixes that. Each ring is restricted to cells
    within leash * ring of the previous point, so the line follows the branch
    it started in, and stops rather than jumping when that branch runs out.

    The line is anchored at the genesis point. Without that it starts a couple
    of hundred kilometres away and appears to float: on a 1 degree grid the
    innermost ring holds only three cells with any probability in them, one
    fewer than the four the ring rule needs, so it is skipped, and the 3-point
    smoothing then pulls the surviving first point further out still. Measured
    on the 17.3N 147.3E August field the gap was 192 km before smoothing and
    215 km after. Anchoring costs nothing in honesty: the storm starts at the
    genesis point by construction, which is known exactly, better than any
    cell average near it. Pass anchor=False to see the raw ring crests.

    Returns (lon, lat, why), or (None, None, None) if the field is unusable.
    why says what ended the line: "thin" when the field fell below the floor,
    "split" when the branch had no continuation inside the leash, "reach" when
    it simply ran out of range. The figure states it, because a line that
    stops halfway across the map otherwise reads as a bug.
    """
    G = np.nan_to_num(np.asarray(prob, dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    lon_c = 0.5 * (np.asarray(lon_edges, float)[:-1]
                   + np.asarray(lon_edges, float)[1:])
    lat_c = 0.5 * (np.asarray(lat_edges, float)[:-1]
                   + np.asarray(lat_edges, float)[1:])
    nlo, nla = lon_c.size, lat_c.size
    # accept the field either way round: (nlon, nlat) as passage_probability
    # returns it, or (nlat, nlon) as a lat-indexed pivot returns it.
    if G.shape == (nlo, nla):
        LO, LA = np.meshgrid(lon_c, lat_c, indexing="ij")
    elif G.shape == (nla, nlo):
        LO, LA = np.meshgrid(lon_c, lat_c, indexing="xy")
    else:
        return None, None, None
    if G.max() <= 0:
        return None, None, None

    def gc(lo1, la1, lo2, la2):
        p1, p2 = np.radians(la1), np.radians(la2)
        h = (np.sin((p2 - p1) / 2.0) ** 2
             + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2.0) ** 2)
        return 2.0 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

    D = gc(lon0, lat0, LO, LA)
    stop = floor * float(G.max())
    xs, ys, why = [], [], "reach"
    for d0 in np.arange(0.0, reach, ring):
        m = (D >= d0) & (D < d0 + ring) & (G > 0)
        if m.sum() < 4:
            continue
        if float(G[m].max()) < stop:
            why = "thin"     # the field has thinned to noise; the ridge ends
            break
        sel = m & (G >= np.quantile(G[m], q))
        if xs:
            sel = sel & (gc(xs[-1], ys[-1], LO, LA) <= leash * ring)
            if not sel.any():
                why = "split"   # this branch has no continuation here
                break
        w = G[sel]
        xs.append(float(np.average(LO[sel], weights=w)))
        ys.append(float(np.average(LA[sel], weights=w)))

    if len(xs) < 4:
        return None, None, None
    xs, ys = np.array(xs), np.array(ys)
    if smooth and smooth > 1 and xs.size >= smooth:
        k = np.ones(int(smooth)) / float(smooth)
        pad = int(smooth) // 2
        xs = np.convolve(np.pad(xs, pad, mode="edge"), k, mode="valid")
        ys = np.convolve(np.pad(ys, pad, mode="edge"), k, mode="valid")
    if anchor:
        # after smoothing, so the anchor stays exactly on the genesis point
        xs = np.concatenate(([float(lon0)], xs))
        ys = np.concatenate(([float(lat0)], ys))
    return xs, ys, why


def central_path(df, mode="median", min_frac=0.5, field=None):
    """A single line through the middle of the plume, or (None, None).

    ridge is the crest of the passage field and needs field=(prob, lon_edges,
    lat_edges, lon0, lat0). It is the only mode here that follows probability
    rather than head count, so it is the one that stays in a narrow westward
    corridor when the recurving branch is more numerous but more spread out.

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
        return None, None, None
    if mode == "ridge":
        return (None, None, None) if field is None else ridge_path(*field)
    n0 = df.SID.nunique()
    steps, mla, mlo = [], [], []
    for step, g in df.groupby("STEP"):
        if len(g) < max(20, min_frac * n0):
            continue
        steps.append(step)
        mla.append(float(g.LAT.median()))
        mlo.append(float(g.LON.median()))
    if len(steps) < 4:
        return None, None, None
    mla, mlo = np.array(mla), np.array(mlo)
    if mode == "median":
        return mlo, mla, None
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
        return mlo, mla, None
    t = df[df.SID == best].sort_values("STEP")
    return t.LON.to_numpy(), t.LAT.to_numpy(), None



def along_ridge(df, rx, ry, lon0, lat0, corridor=150.0, min_frac=0.8,
                skip=4, min_steps=8):
    """Every realisation that follows the ridge, not the best N of them.

    Selecting a fixed count gives a bundle whose density carries no
    information: 50 lines look the same whether 50 realisations or 5,000 take
    the corridor. This applies a fixed criterion instead and reports how many
    met it, so the figure states what fraction of the ensemble is in the
    corridor and the drawn lines are a sample of a named population.

    A realisation is in if at least min_frac of its positions lie within
    corridor km of the ridge. Only positions inside the ridge's own range from
    genesis are scored, so a realisation that keeps going after the ridge ends
    is not punished for the part of its track the ridge never covered. That
    matters here: the ridge stops where the corridor does, and the storms that
    carry on west past that point are the ones worth seeing.

    Distance is measured to the nearest ridge vertex rather than to the
    polyline, which overstates it by at most half the ring spacing, about
    55 km against a corridor of 300.

    Returns (ids, n_total). Falls back to every realisation if the ridge is
    unusable, since drawing nothing is worse than drawing everything.
    """
    n_total = int(df.SID.nunique())
    if rx is None or len(rx) < 2:
        return list(df.SID.unique()), n_total
    R = 6371.0

    def gc(lo1, la1, lo2, la2):
        p1, p2 = np.radians(la1), np.radians(la2)
        h = (np.sin((p2 - p1) / 2.0) ** 2
             + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2.0) ** 2)
        return 2.0 * R * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

    d = df.sort_values(["SID", "STEP"])
    d = d[d.STEP >= skip]
    if d.empty:
        return list(df.SID.unique()), n_total
    x = d.LON.to_numpy(dtype=float); y = d.LAT.to_numpy(dtype=float)
    reach = float(gc(lon0, lat0, np.asarray(rx), np.asarray(ry)).max())
    inrange = gc(lon0, lat0, x, y) <= reach + 0.5 * corridor
    # nearest ridge vertex, all points at once
    near = gc(x[:, None], y[:, None],
              np.asarray(rx)[None, :], np.asarray(ry)[None, :]).min(axis=1)
    ok = pd.DataFrame({"SID": d.SID.to_numpy(),
                       "scored": inrange,
                       "close": inrange & (near <= corridor)})
    g = ok.groupby("SID", sort=False).sum()
    g = g[g.scored >= min_steps]
    keep = g.index[(g.close / g.scored) >= min_frac]
    if not len(keep):
        return list(df.SID.unique()), n_total
    return list(keep), n_total


def closest_to_ridge(df, rx, ry, lon0, lat0, n, skip=4, restrict=None):
    """The n realisations whose tracks run closest to the ridge.

    Ranked by the mean distance from the realisation to the nearest ridge
    vertex, over the steps that fall within the ridge's own range from genesis.
    Returns a list of SIDs, closest first, taken from `restrict` if given.

    This draws a TIGHTER bundle than the corridor on purpose: "the ten closest
    tracks" is a tighter object than "every track within 150 km". It is only
    ever an option, never the default, and it is meant to sit over the
    full-ensemble underlay, so the spread it leaves out stays on the page.
    """
    if rx is None or len(rx) < 2:
        return list(df.SID.unique())[:n]
    R = 6371.0

    def gc(lo1, la1, lo2, la2):
        p1, p2 = np.radians(la1), np.radians(la2)
        h = (np.sin((p2 - p1) / 2.0) ** 2
             + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2.0) ** 2)
        return 2.0 * R * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

    d = df.sort_values(["SID", "STEP"])
    d = d[d.STEP >= skip]
    if restrict is not None:
        d = d[d.SID.isin(set(restrict))]
    if d.empty:
        return (list(restrict)[:n] if restrict is not None
                else list(df.SID.unique())[:n])
    x = d.LON.to_numpy(float); y = d.LAT.to_numpy(float)
    reach = float(gc(lon0, lat0, np.asarray(rx), np.asarray(ry)).max())
    inrange = gc(lon0, lat0, x, y) <= reach
    near = gc(x[:, None], y[:, None],
              np.asarray(rx)[None, :], np.asarray(ry)[None, :]).min(axis=1)
    t = pd.DataFrame({"SID": d.SID.to_numpy(), "near": near, "in": inrange})
    t = t[t["in"]]
    if t.empty:
        return (list(restrict)[:n] if restrict is not None
                else list(df.SID.unique())[:n])
    order = t.groupby("SID", sort=False).near.mean().sort_values()
    return list(order.index[:n])


def fit_extent(prob, lon_edges, lat_edges, lon0, lat0, verts,
               tail=0.05, pad=1.5, min_span=14.0):
    """Crop the frame to the plume instead of drawing the whole model domain.

    The field is generated over the full basin, about 100-180E and 0-45N, but
    one genesis point fills a small part of it. At domain size the plume is a
    smudge in a corner and the drawn realisations read as scratch rather than
    as a corridor: same line width, same alpha, twice the frame. Cropping is
    the whole difference.

    The crop is a marginal one. Passage probability is summed along each axis
    and the outer `tail` of that sum is dropped from each end, which trims the
    thin fringe without cutting into the body of the plume. At the default it
    keeps about 90 per cent of the passage probability. A bounding box over
    cells above a threshold does not work here: with ten thousand realisations
    a 0.4 per cent contour still reaches both edges of the basin.

    The PAR and the genesis point are always inside the result, so the frame
    still answers the question the tool is for. Pass --ext to override.
    """
    G = np.nan_to_num(np.asarray(prob, dtype=float), nan=0.0,
                      posinf=0.0, neginf=0.0)
    lon_c = 0.5 * (np.asarray(lon_edges, float)[:-1]
                   + np.asarray(lon_edges, float)[1:])
    lat_c = 0.5 * (np.asarray(lat_edges, float)[:-1]
                   + np.asarray(lat_edges, float)[1:])
    if G.shape == (lon_c.size, lat_c.size):
        mlon, mlat = G.sum(axis=1), G.sum(axis=0)
    elif G.shape == (lat_c.size, lon_c.size):
        mlon, mlat = G.sum(axis=0), G.sum(axis=1)
    else:
        return None
    if G.sum() <= 0:
        return None

    def span(centres, mass):
        c = np.cumsum(mass) / mass.sum()
        lo = centres[min(int(np.searchsorted(c, tail)), centres.size - 1)]
        hi = centres[min(int(np.searchsorted(c, 1.0 - tail)), centres.size - 1)]
        return float(min(lo, hi)), float(max(lo, hi))

    x0, x1 = span(lon_c, mlon)
    y0, y1 = span(lat_c, mlat)
    xs = [x0, x1, lon0] + [v[1] for v in verts]
    ys = [y0, y1, lat0] + [v[0] for v in verts]
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    if x1 - x0 < min_span:
        c = 0.5 * (x0 + x1); x0, x1 = c - min_span / 2, c + min_span / 2
    if y1 - y0 < min_span:
        c = 0.5 * (y0 + y1); y0, y1 = c - min_span / 2, c + min_span / 2
    # never past the field itself; there is nothing drawn out there
    x0 = max(x0, float(lon_edges[0]));  x1 = min(x1, float(lon_edges[-1]))
    y0 = max(y0, float(lat_edges[0]));  y1 = min(y1, float(lat_edges[-1]))
    return x0, x1, y0, y1


def figure(prob, lon_edges, lat_edges, df, lat0, lon0, month, n, path, verts,
           dtm, central="ridge", tracks=60, ext=None, pick="ridge",
           track_color="#d00000", track_lw=0.2, track_alpha=0.3,
           corridor=150.0, top=None, along_ids=None, n_all=None,
           ridge_xy=None, conditional=False, underlay=False,
           under_color="#d00000", under_lw=0.10, under_alpha=0.05,
           show_field=True):
    n_all = int(df.SID.nunique()) if n_all is None else int(n_all)
    frame = ext or fit_extent(prob, lon_edges, lat_edges, lon0, lat0, verts)
    if frame is None:
        frame = (float(lon_edges[0]), float(lon_edges[-1]),
                 float(lat_edges[0]), float(lat_edges[-1]))
    x0, x1, y0, y1 = frame
    if ext is None:
        _g = np.nan_to_num(np.asarray(prob, float))
        _lo = 0.5 * (np.asarray(lon_edges, float)[:-1] + np.asarray(lon_edges, float)[1:])
        _la = 0.5 * (np.asarray(lat_edges, float)[:-1] + np.asarray(lat_edges, float)[1:])
        _LO, _LA = (np.meshgrid(_lo, _la, indexing="ij")
                    if _g.shape == (_lo.size, _la.size)
                    else np.meshgrid(_lo, _la, indexing="xy"))
        _in = (_LO >= x0) & (_LO <= x1) & (_LA >= y0) & (_LA <= y1)
        if _g.sum() > 0:
            print(f"frame {x0:.1f}-{x1:.1f}E {y0:.1f}-{y1:.1f}N holds "
                  f"{100 * _g[_in].sum() / _g.sum():.1f}% of the passage "
                  f"probability; --ext overrides it")
    latm = 0.5 * (y0 + y1)
    # aspect is the y unit over the x unit, so 1/cos(lat) is the one that keeps
    # a degree of longitude the right length against a degree of latitude.
    asp = 1.0 / max(0.2, np.cos(np.radians(latm)))
    wh = (x1 - x0) / ((y1 - y0) * asp)
    fig, ax = plt.subplots(figsize=(float(np.clip(6.6 * wh, 5.4, 9.8)) + 1.4,
                                    6.6), facecolor="white")
    m = np.ma.masked_where(prob.T <= 0, prob.T)
    # The genesis cell is 1.0 by construction and a linear ramp spends the whole
    # colour range on it, leaving the plume nearly white. A power norm keeps the
    # axis honest, 0 to 1 with nothing clipped, while giving the low
    # probabilities that carry the actual information somewhere to live.
    pc = None
    if show_field:
        pc = ax.pcolormesh(lon_edges, lat_edges, m, cmap=SEQ, shading="flat",
                           norm=matplotlib.colors.PowerNorm(0.45, vmin=0,
                                                            vmax=1),
                           zorder=1)
    tlabel = None
    if pick == "ridge":
        # ids_all is the whole corridor population: every realisation that
        # stays within `corridor` km of the ridge. It is drawn WHOLE by
        # default, so the red set IS that population, not a sample of it: no
        # ordering, no seed, nothing hidden. --top N instead draws the N that
        # run closest to the ridge, a tighter bundle, over the same underlay.
        ids_all = list(along_ids) if along_ids is not None else []
        share = 100.0 * len(ids_all) / max(n_all, 1)
        # The width is in the label. The share is a property of the tube, not
        # of the storm: on one 17.3N 147.3E field it runs from 2.6% at 100 km
        # to 26% at 400 km. A reader who cannot see the width would read the
        # share as a forecast probability, which it is not.
        if top and ridge_xy is not None and 0 < top < len(ids_all):
            ids = closest_to_ridge(df, ridge_xy[0], ridge_xy[1], lon0, lat0,
                                   top, restrict=ids_all)
            tlabel = (f"{len(ids)} realisations closest to the ridge "
                      f"(of {len(ids_all):,} within {corridor:g} km, "
                      f"{share:.1f}% of {n_all:,})")
        else:
            ids = ids_all
            tlabel = (f"{len(ids_all):,} realisations within {corridor:g} km "
                      f"of the ridge ({share:.1f}% of {n_all:,})")
    elif tracks > 0:
        u = df.SID.unique()
        ids = list(u[:: max(1, len(u) // tracks)])
    else:
        ids = []
    if underlay:
        # Every realisation, faint, underneath. This is the only way the drawn
        # ink is proportional to the population: 50 lines look like 50 lines
        # whether they stand for 257 realisations or 1,877, but 257 red lines
        # against 10,000 grey ones look like 2.6%, which is what they are.
        # Note what it duplicates: the coloured field IS the density of all
        # 10,000, so the underlay says the same thing in ink. That is the
        # point, not a fault. It puts the sample and the population in the
        # same visual units so the eye can compare them.
        allsub = df.sort_values(["SID", "STEP"])
        allsegs = [g[["LON", "LAT"]].to_numpy()
                   for _, g in allsub.groupby("SID", sort=False) if len(g) > 1]
        if allsegs:
            ax.add_collection(LineCollection(
                allsegs, colors=under_color, linewidths=under_lw,
                alpha=under_alpha, zorder=1.5, capstyle="round",
                label=f"all {n_all:,} realisations"))
    if ids:
        # A handful of tracks needs a bold line to read over the dark field; a
        # few hundred needs a thin one or it becomes a block. Scale the weight
        # to the count drawn, not to the storm, so nothing is hand-set per map.
        k = len(ids)
        lw = (0.7 if k <= 15 else 0.38 if k <= 60 else
              0.20 if k <= 200 else 0.10)
        al = (0.85 if k <= 15 else 0.55 if k <= 60 else
              0.38 if k <= 200 else 0.22)
        # One LineCollection rather than a few thousand plot calls.
        sub = df[df.SID.isin(set(ids))].sort_values(["SID", "STEP"])
        segs = [g[["LON", "LAT"]].to_numpy()
                for _, g in sub.groupby("SID", sort=False) if len(g) > 1]
        if segs:
            ax.add_collection(LineCollection(
                segs, colors=track_color, linewidths=lw,
                alpha=al, zorder=2, capstyle="round", label=tlabel))
    land_outline(ax, dtm)
    v = np.array(verts + (verts[0],))
    ax.plot(v[:, 1], v[:, 0], color="#0b3a5c", lw=1.3, ls="--", zorder=5,
            label="PAR")
    if central == "ridge" and ridge_xy is not None:
        # Never re-derive the ridge from a field that was conditioned on it.
        # That closes a loop: select the storms near the ridge, rebuild the
        # field from only those storms, then read a ridge off that field, which
        # will of course sit where you put it. The ridge stays the one measured
        # on the unconditional field.
        cx, cy, why = ridge_xy
    else:
        cx, cy, why = central_path(
            df, central, field=(prob, lon_edges, lat_edges, lon0, lat0))
    if cx is not None:
        # Six times the weight of a realisation, because it is the one line in
        # the figure making a statement. The white stroke is what keeps it
        # readable over the dark cells.
        lab = {"median": "middle of the plume",
               "ridge": "ridge of the plume"}.get(
                   central, "most representative track")
        # A line that stops in open water reads as a bug unless the figure
        # says otherwise, so the reason goes in the legend and the end of the
        # line is marked. It is a real terminus, not a clipped one.
        ax.plot(cx, cy, color="#1a5fd0", lw=1.0, zorder=5.5,
                solid_capstyle="round", label=lab)
        end = {"thin": "field thins out here",
               "split": "corridor ends here"}.get(why)
        if end:
            ax.plot([cx[-1]], [cy[-1]], marker="o", ms=4.4, mfc="white",
                    mec="#08306b", mew=1.1, zorder=5.6, linestyle="none",
                    label=end)

    # Genesis marker: a point, because the thing being marked is a coordinate.
    # Anything larger covers the highest-probability cells in the figure, which
    # are exactly the ones sitting under it. Matches plume_pair.py.
    ax.plot([lon0], [lat0], marker="o", ms=7, color="#0b3a5c",
            markeredgecolor="white", markeredgewidth=1.3, zorder=6,
            linestyle="none", label="genesis")
    cb = fig.colorbar(pc, ax=ax, shrink=0.82, pad=0.02) if pc else None
    if cb is not None:
        cb.set_label(("probability the storm passes through this cell,\n"
                  "GIVEN it follows the corridor"
                  if conditional else
                  "probability the storm passes through this cell") +
                 "\n(colour scale is power-stretched, values are unclipped)",
                     fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
    ax.set_xlabel("longitude (E)", fontsize=10)
    ax.set_ylabel("latitude (N)", fontsize=10)
    if FS.TITLES:
        ax.set_title(f"Genesis at {lat0:.1f}N {lon0:.1f}E in {MONTHS[month]}\n"
                     f"{n:,} SynTC realisations", fontsize=12, color=INK,
                     loc="left")
    ax.legend(fontsize=8.5, loc="upper right", frameon=True, edgecolor=LINE)
    ax.grid(which="major", lw=0.5, color="#c9c9c9", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(asp)
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
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for storm generation. The drawn track selection "
                         "is deterministic and does not use it.")
    ap.add_argument("--out", default=".")
    ap.add_argument("--tracks", type=int, default=60,
                    help="how many realisations to draw over the field. "
                         "0 shows the probability field alone, which is the "
                         "honest object: the field carries the uncertainty, "
                         "the drawn lines are only a sample of it.")
    ap.add_argument("--central", default="ridge",
                    choices=("ridge", "off", "median", "medoid"),
                    help="draw a blue line through the middle of the plume. "
                         "ridge is the crest of the probability field and is "
                         "the one to use: it follows the densest corridor even "
                         "where a more numerous but more spread out branch "
                         "pulls the other two away. median is a per-step "
                         "statistic over all surviving realisations and sits "
                         "between the branches of a bimodal plume; medoid is "
                         "the single realisation closest to that median. Off "
                         "by default so the manuscript figures are unchanged.")
    ap.add_argument("--track-color", default="#d00000",
                    help="colour of the drawn realisations. Red reads well "
                         "over the pale cells that cover most of the frame, "
                         "and loses contrast only in the small dark band "
                         "around genesis, where the field itself is red. Use "
                         "#0b0b0b for near-black if that band matters.")
    ap.add_argument("--track-lw", type=float, default=0.2,
                    help="line width of the drawn realisations")
    ap.add_argument("--track-alpha", type=float, default=0.3,
                    help="opacity of the drawn realisations. A thinner line "
                         "needs more of this to stay visible.")
    ap.add_argument("--corridor", type=float, default=150.0,
                    help="half-width in km of the corridor. 150 km is the "
                         "default: it is about the radius of an average "
                         "tropical cyclone (a 200-500 km diameter is a "
                         "100-250 km radius) and it is within the plume's own "
                         "measured spread of 50-185 km, so two independent "
                         "rulers agree on it. It is NOT calibrated against "
                         "observations; the share it produces depends on it "
                         "and is reported in the legend for that reason.")
    ap.add_argument("--underlay", action="store_true",
                    help="overlay every realisation faintly on top of the "
                         "field. OFF by default, so the map is the field plus "
                         "the corridor. Turn it ON to show all 10,000 tracks "
                         "over the field: that is the proof the ridge follows "
                         "the real density and is not a drawn-in line.")
    ap.add_argument("--under-alpha", type=float, default=0.05,
                    help="opacity of the underlay. At 10,000 lines this has to "
                         "be very low or the page goes solid.")
    ap.add_argument("--under-color", default="#d00000",
                    help="colour of the underlay (all realisations). Red by "
                         "default, so the whole ensemble reads as one soft red "
                         "spaghetti with the corridor a touch denser on top. "
                         "Use a grey such as #4a4a4a to set the corridor red "
                         "apart from the rest.")
    ap.add_argument("--top", type=int, default=None, metavar="N",
                    help="draw only the N realisations closest to the ridge "
                         "(5, 10, 15 ... up to 100), instead of the whole "
                         "corridor. A tighter bundle by design; it still sits "
                         "over the full-ensemble underlay, so the spread it "
                         "leaves out stays visible. Default draws every "
                         "realisation in the corridor.")
    ap.add_argument("--field", default="all",
                    choices=("all", "corridor", "none"),
                    help="which passage field is shown. all is the DEFAULT and "
                         "is the honest one: every cell any realisation "
                         "reaches keeps its colour, so the recurving branch "
                         "stays visible even though no drawn track goes there. "
                         "corridor rebuilds the field from the realisations "
                         "near the ridge only, which makes the coloured core "
                         "sit on the blue line but hides where the other four "
                         "fifths of the ensemble went, and changes the "
                         "quantity to a conditional probability. The ridge is "
                         "measured on the unconditional field either way, so "
                         "the conditioning can never feed itself. none draws "
                         "no field at all, which is the right pairing with "
                         "--underlay: 10,000 drawn lines ARE the density, so "
                         "colouring it underneath as well says the same thing "
                         "twice and hazes the colormap doing it.")
    ap.add_argument("--pick", default="ridge", choices=("ridge", "even"),
                    help="which realisations get drawn. ridge is the DEFAULT: "
                         "it draws every realisation that stays within "
                         "--corridor km of the ridge, stating the count in the "
                         "legend, so the red set IS the corridor population "
                         "rather than a sample of it. Narrow it with --top N. "
                         "even is a blind stride through the whole ensemble "
                         "and shows the unconditional spread. The old core "
                         "mode is gone: it ranked by the probability of the "
                         "cells a track visited, which in a bimodal plume "
                         "rewards the broad recurving branch over the "
                         "narrow corridor. Measured on a 17.3N 147.3E August "
                         "field it put 5 of its 60 picks in the corridor, "
                         "against 7 for a blind stride.")
    ap.add_argument("--ext", nargs=4, type=float, default=None,
                    metavar=("LON0", "LON1", "LAT0", "LAT1"),
                    help="map frame. Default fits the plume, the PAR and the "
                         "genesis point, because drawing the whole model "
                         "domain leaves the plume a smudge in one corner.")
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

    cfg = model.cfg
    lon_edges = np.arange(cfg.lon_min, cfg.lon_max + a.grid, a.grid)
    lat_edges = np.arange(cfg.lat_min, cfg.lat_max + a.grid, a.grid)
    prob, n_used = passage_probability(df, lon_edges, lat_edges)

    # The ridge and the corridor membership are both measured on the
    # unconditional field. Only what is DRAWN may be conditioned on them.
    ridge_xy = ridge_path(prob, lon_edges, lat_edges, a.lon, a.lat)
    along_ids, n_all = along_ridge(df, ridge_xy[0], ridge_xy[1], a.lon, a.lat,
                                   corridor=a.corridor)

    # after the ridge, so the summary can report whether it enters PAR
    summarise(df, seen_par, crossed, a.lat, a.lon, a.month, a.dtm,
              ridge_xy=ridge_xy, verts=cfg.par_vertices)

    conditional = (a.field == "corridor" and a.pick == "ridge"
                   and 0 < len(along_ids) < n_all)
    prob_show = prob
    if conditional:
        prob_show, _ = passage_probability(
            df[df.SID.isin(set(along_ids))], lon_edges, lat_edges)
        print(f"corridor: {len(along_ids):,} of {n_all:,} realisations "
              f"({100.0 * len(along_ids) / n_all:.1f}%) stay within "
              f"{a.corridor:g} km of the ridge; the field shown is "
              f"conditional on that")

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"genesis_{a.lat:g}N_{a.lon:g}E_m{a.month:02d}")
    df.to_csv(f"{stem}_tracks.csv", index=False)
    ii, jj = np.nonzero(prob)
    pd.DataFrame({"lon": lon_edges[ii] + a.grid / 2,
                  "lat": lat_edges[jj] + a.grid / 2,
                  "probability": prob[ii, jj]}).to_csv(
        f"{stem}_passage.csv", index=False)
    if conditional:
        ii2, jj2 = np.nonzero(prob_show)
        pd.DataFrame({"lon": lon_edges[ii2] + a.grid / 2,
                      "lat": lat_edges[jj2] + a.grid / 2,
                      "probability": prob_show[ii2, jj2]}).to_csv(
            f"{stem}_passage_corridor.csv", index=False)
    figure(prob_show, lon_edges, lat_edges, df, a.lat, a.lon, a.month, n_used,
           f"{stem}.png", cfg.par_vertices, a.dtm,
           central=a.central, tracks=a.tracks, ext=a.ext, pick=a.pick,
           track_color=a.track_color, track_lw=a.track_lw,
           track_alpha=a.track_alpha, corridor=a.corridor,
           top=a.top, along_ids=along_ids, n_all=n_all,
           ridge_xy=ridge_xy, conditional=conditional,
           underlay=a.underlay, under_alpha=a.under_alpha,
           under_color=a.under_color, show_field=(a.field != "none"))
    print(f"\nwritten:\n  {stem}.png\n  {stem}.pdf\n  {stem}_tracks.csv"
          f"\n  {stem}_passage.csv")
    print("\nThese are probabilities GIVEN a storm forms at that point in that "
          "month.\nThey are not the probability that one forms.")


if __name__ == "__main__":
    main()
