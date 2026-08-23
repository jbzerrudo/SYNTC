"""Several genesis runs on one map, one colour per origin.

    python plume_multi.py --gen forecast --dtm dtm_phil_1km.tif \
        --run genesis_17.3N_147.3E_m08 17.3 147.3 "Saudel" \
        --run genesis_9.8N_136.4E_m08   9.8 136.4 "LPA east of Mindanao" \
        --keep 15 --pick spread --out two_systems

For two or more systems live at the same time. Each --run is a separate
genesis_forecast.py output already sitting in --gen.

Why the probability fields are not added
---------------------------------------
Each field is a probability CONDITIONAL on a storm existing at that origin.
Two such fields describe different conditioning events, so summing them is not
a probability of anything. Adding them would also double-count the water both
systems can reach, which is exactly the water a forecaster cares about.

This script therefore draws the tracks of every origin together, which is
legitimate because tracks are just paths, and shows at most ONE field as
background, chosen with --field. If you want a genuinely combined field you
have to supply the relative likelihood of each origin yourself, and that is a
judgement about the present synoptic situation that this model does not carry.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import rasterio

SEQ = mcolors.LinearSegmentedColormap.from_list("heat", [
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
    "#fc4e2a", "#e31a1c", "#bd0026", "#800026"])
PAR = ((25.0, 120.0), (25.0, 135.0), (5.0, 135.0), (5.0, 115.0),
       (15.0, 115.0), (21.0, 120.0))
# Dark2. Six origins is already more than a map can carry legibly.
# The genesis marker carries the system's PAGASA class, using the same colours
# as the intensity figures in the manuscript, with LPA added below TD.
CAT = (("LPA",   0,  21, "#1a9850"),
       ("TD",   22,  33, "#2166ac"),
       ("TS",   34,  47, "#f0c000"),
       ("STS",  48,  63, "#f57c00"),
       ("TY",   64,  99, "#d7191c"),
       ("STY", 100, 999, "#762a83"))
CATC = {c[0]: c[3] for c in CAT}
# Tracks stay the single dark blue plume_pair uses. Identity is carried by the
# marker colour and by which origin the lines leave from, not by the lines.
TRACK = "#e31a1c"          # realisations, red
CENTRAL = "#08306b"        # the middle track, blue


def classify(kt):
    for name, lo, hi, _ in CAT:
        if lo <= kt <= hi:
            return name
    return "LPA"

ap = argparse.ArgumentParser(description="Several genesis runs on one map.")
ap.add_argument("--gen", default="forecast")
ap.add_argument("--dtm", default="dtm_phil_1km.tif")
ap.add_argument("--run", nargs="+", action="append", required=True,
                metavar="STEM LAT LON LABEL [CLASS]",
                help="repeat once per origin. CLASS is one of LPA TD TS STS "
                     "TY STY and sets the marker colour. Omit it and the class "
                     "is read from the run's own genesis wind, which is wrong "
                     "for an LPA, because the model draws a genesis intensity "
                     "for it while PAGASA has not classified it.")
ap.add_argument("--keep", type=int, default=15, help="tracks drawn per origin")
ap.add_argument("--pick", default="spread", choices=("random", "spread"),
                help="core is deliberately unavailable here: it selects "
                     "against divergence, which defeats the purpose of "
                     "comparing where two systems could go")
ap.add_argument("--min-steps", type=int, default=8)
ap.add_argument("--field", default="",
                help="stem whose probability field is drawn as background. "
                     "Omit for no field, which is the honest default when the "
                     "origins are not comparable in likelihood.")
ap.add_argument("--ext", nargs=4, type=float, default=[112.0, 155.0, 2.0, 28.0],
                metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--note", default="", help="stats block above the map, \\n for breaks")
ap.add_argument("--title", default="")
ap.add_argument("--legend-loc", default="below",
                choices=("below", "right", "lower left", "lower right",
                         "upper left", "upper right", "off"),
                help="where the origin legend goes. below and right put it "
                     "outside the map, which is the default because a legend "
                     "inside covers the archipelago.")
ap.add_argument("--focus", default="",
                help="STEM of the one system to feature. Its full field is "
                     "drawn as cells and its tracks are shown; every other "
                     "system is reduced to its marker and its 50%% core "
                     "outline, as context. NOTE: this is a display choice "
                     "only. SynTC propagates every storm conditioned on its "
                     "own state alone, so the focal system's realisations were "
                     "drawn in a world where the others do not exist. No "
                     "interaction between systems is modelled anywhere.")
ap.add_argument("--context", default="outline",
                choices=("outline", "marker", "off"),
                help="with --focus, how much of the other systems to draw. "
                     "outline gives each a marker and its 50%% core, marker "
                     "gives the dot only, off draws nothing at all, leaving a "
                     "single genesis point on the map.")
ap.add_argument("--central", default="medoid",
                 choices=("medoid", "mean", "median", "off"),
                 help="the thick line through the middle of the plume. medoid "
                      "(default) is a real realisation, the one that stays "
                      "closest to the ensemble median path, so it is always "
                      "physically possible. mean and median are per-step "
                      "statistics and can run between the branches of a "
                      "bimodal plume, through water no storm visited. off "
                      "draws none.")
ap.add_argument("--no-cores", action="store_true",
                help="draw tracks only, without each origin's probability core")
ap.add_argument("--core-alpha", type=float, default=0.80,
                help="fill opacity of the cores")
ap.add_argument("--core-style", default="ylord", choices=("ylord", "bysystem"),
                help="ylord (default) fills every core with the same "
                     "yellow-to-red ramp used everywhere else in this release, "
                     "so overlapping cores simply darken; bysystem tints each "
                     "core with its own track colour instead.")
ap.add_argument("--out", default="genesis_multi")
A = ap.parse_args()
EXT = tuple(A.ext)

src = rasterio.open(A.dtm)
_e = src.read(1).astype(float)
_e[~np.isfinite(_e)] = 0.0
_e[_e < 0] = 0.0
_s = 8
_land = _e[::_s, ::_s] > 0.5
_ny, _nx = _land.shape
_lon = src.bounds.left + (np.arange(_nx) + 0.5) * _s * src.transform.a
_lat = src.bounds.top + (np.arange(_ny) + 0.5) * _s * src.transform.e


def core(ax, gen, stem, tr, colour, fill_alpha, cells=True,
         full=False):
    """This origin's own probability core, from its own conditional field.

    Every realisation is scored by the LOWEST passage probability along its
    own path. The region where that origin's field exceeds the median of those
    scores then contains half of its realisations along their entire path; the
    10th percentile gives ninety percent.

    Each origin gets its own contour from its own field. Nothing is summed,
    because two fields conditional on different events do not add, but every
    origin does get a core, which is what a forecaster looking at two or three
    live systems actually needs to see.
    """
    f = os.path.join(gen, stem + "_passage.csv")
    if not os.path.exists(f):
        print("  no core for %s: %s_passage.csv is missing from %s"
              % (stem, stem, gen))
        return None
    p = pd.read_csv(f)
    lons = np.sort(p.lon.unique())
    lats = np.sort(p.lat.unique())
    dlon, dlat = np.diff(lons)[0], np.diff(lats)[0]
    g = np.nan_to_num(
        p.pivot(index="lat", columns="lon", values="probability").values, nan=0.0)
    lut = {(round(r.lon, 1), round(r.lat, 1)): r.probability
           for r in p.itertuples()}
    snap = lambda v, v0, d: round(v0 + np.round((v - v0) / d) * d, 1)
    mins = []
    for _, t in tr.groupby("SID", sort=False):
        t = t.sort_values("STEP")
        x, y = t.LON.to_numpy(), t.LAT.to_numpy()
        w = (x >= EXT[0]) & (x <= EXT[1]) & (y >= EXT[2]) & (y <= EXT[3])
        if w.sum() < 2:
            continue
        mins.append(min(lut.get((snap(a, lons[0], dlon),
                                 snap(b, lats[0], dlat)), 0.0)
                        for a, b in zip(x[w], y[w])))
    if len(mins) < 20:
        print("  no core for %s: only %d realisations inside the frame"
              % (stem, len(mins)))
        return None
    mins = np.asarray(mins)
    top = float(np.nanmax(g))
    lev50 = float(np.percentile(mins, 50))
    lev90 = float(np.percentile(mins, 10))
    if not (0 < lev90 < lev50 < top):
        print("  no core for %s: levels degenerate (90%%=%.4f, 50%%=%.4f)"
              % (stem, lev90, lev50))
        return None
    # pcolormesh, not contourf. The 1-degree cell is the unit the probability
    # is defined on, and smoothing it into a blob hides the resolution of the
    # field. Same ramp, same power stretch, same shading as the single-panel
    # plume; only the mask is new, clipping each system to its own 90% region
    # so three fields can share one map.
    mesh = None
    if cells:
        le = np.append(lons - dlon / 2, lons[-1] + dlon / 2)
        laa = np.append(lats - dlat / 2, lats[-1] + dlat / 2)
        # A focal system shows its whole field, exactly as the single-panel
        # plume does. A system sharing the map is clipped to its 90% core so
        # three fields do not paint over each other.
        m = np.ma.masked_where(g <= 0 if full else g < lev90, g)
        mesh = ax.pcolormesh(le, laa, m, cmap=SEQ, shading="flat",
                             norm=mcolors.PowerNorm(0.45, vmin=0.0, vmax=1.0),
                             alpha=fill_alpha, zorder=1.5)
    rings = ((lev90, 0.8, (0, (1.4, 2.4)), 0.75),
             (lev50, 0.9, (0, (5.0, 2.4)), 0.95))
    for lev, lw, ls, al in (rings if cells else rings[1:]):
        cs = ax.contour(lons, lats, g, levels=[lev], colors=colour,
                        linewidths=lw, linestyles=[ls], zorder=4, alpha=al)
        halo = [pe.withStroke(linewidth=lw + 2.0, foreground="white",
                              alpha=0.85)]
        try:
            cs.set_path_effects(halo)
        except AttributeError:
            for c in cs.collections:
                c.set_path_effects(halo)
    return mesh


def central_track(tr, mode="medoid", min_frac=0.5):
    """The single line drawn thick through the middle of the plume.

    medoid (default) is a REAL realisation: the one whose own path stays
    closest to the ensemble's median path. It is always physically possible,
    because a storm actually followed it.

    mean and median are per-step statistics. They are what most people mean by
    "the average track" and they are dangerous here: when the plume is bimodal,
    one branch running west and one recurving, the mean position at every step
    sits between the two branches, in water no realisation ever entered. Offered
    because they are asked for, defaulted away from because of that.

    Either way the line stops once fewer than min_frac of the realisations are
    still alive, since a mean over the surviving quarter is a mean over the
    longest-lived storms, not over the ensemble.
    """
    if mode == "off":
        return None, None
    n0 = tr.SID.nunique()
    steps, mla, mlo = [], [], []
    for step, g in tr.groupby("STEP"):
        if len(g) < max(20, min_frac * n0):
            continue
        steps.append(step)
        mla.append(float(g.LAT.median()))
        mlo.append(float(g.LON.median()))
    if len(steps) < 4:
        return None, None
    steps = np.array(steps)
    mla, mlo = np.array(mla), np.array(mlo)

    if mode in ("mean", "median"):
        if mode == "mean":
            mla = np.array([float(tr[tr.STEP == s].LAT.mean()) for s in steps])
            mlo = np.array([float(tr[tr.STEP == s].LON.mean()) for s in steps])
        return mlo, mla

    # medoid: the realisation with the smallest mean distance to that path
    ref = dict(zip(steps, zip(mla, mlo)))
    best, best_d = None, np.inf
    for sid, t in tr.groupby("SID", sort=False):
        t = t[t.STEP.isin(ref)]
        if len(t) < 0.8 * len(steps):
            continue
        a = np.array([ref[s] for s in t.STEP])
        dy = (t.LAT.to_numpy() - a[:, 0]) * 111.32
        dx = (t.LON.to_numpy() - a[:, 1]) * 111.32 * np.cos(np.radians(a[:, 0]))
        d = float(np.hypot(dx, dy).mean())
        if d < best_d:
            best, best_d = sid, d
    if best is None:
        return mlo, mla
    t = tr[tr.SID == best].sort_values("STEP")
    return t.LON.to_numpy(), t.LAT.to_numpy()


def draw_central(ax, x, y, colour="#08306b", lw=0.75, z=5.5):
    if x is None or len(x) < 2:
        return
    ax.plot(x, y, color=colour, lw=lw, zorder=z, solid_capstyle="round",
            path_effects=[pe.withStroke(linewidth=lw + 1.3, foreground="white",
                                        alpha=0.9)])


def north_arrow(ax, x=0.055, y=0.84, L=0.085):
    ax.annotate("", xy=(x, y + L), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#0b0b0b",
                                mutation_scale=12))
    ax.text(x, y + L + 0.012, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=10, fontweight="bold", color="#0b0b0b")


def endpoint(tr, sids, step=20):
    sub = tr[tr.SID.isin(set(sids))]
    last = (sub.sort_values("STEP").groupby("SID").tail(1)
               .set_index("SID")[["LON", "LAT"]])
    at = sub[sub.STEP == step].set_index("SID")[["LON", "LAT"]]
    out = last.copy()
    common = out.index.intersection(at.index)
    out.loc[common] = at.loc[common]
    return out


def choose(tr, n, seed):
    """Eligible realisations, then n of them spread across the outcomes."""
    ok = []
    for sid, t in tr.groupby("SID", sort=False):
        x = t.LON.to_numpy()
        y = t.LAT.to_numpy()
        w = (x >= EXT[0]) & (x <= EXT[1]) & (y >= EXT[2]) & (y <= EXT[3])
        if len(x) >= A.min_steps and w.sum() >= A.min_steps:
            ok.append(sid)
    ok = pd.Index(ok)
    n = min(n, len(ok))
    rng = np.random.default_rng(seed)
    if n <= 0:
        return ok[:0]
    if A.pick == "random":
        return pd.Index(rng.choice(ok.to_numpy(), n, replace=False))
    X = endpoint(tr, ok).reindex(ok).to_numpy(dtype=float)
    try:
        from scipy.cluster.vq import kmeans2
        cen, lab = kmeans2(X, n, minit="++", seed=seed, missing="warn")
    except Exception:
        return pd.Index(rng.choice(ok.to_numpy(), n, replace=False))
    picked = []
    for k in range(len(cen)):
        m = np.nonzero(lab == k)[0]
        if len(m):
            d = np.hypot(*(X[m] - cen[k]).T)
            picked.append(ok[m[int(np.argmin(d))]])
    if len(picked) < n:
        rest = [s for s in ok if s not in set(picked)]
        picked += list(rng.choice(rest, min(n - len(picked), len(rest)),
                                  replace=False))
    return pd.Index(picked)


fig, ax = plt.subplots(figsize=(8.6, 6.4), facecolor="white",
                       constrained_layout=True)

if A.field:
    p = pd.read_csv(os.path.join(A.gen, A.field + "_passage.csv"))
    lons = np.sort(p.lon.unique())
    lats = np.sort(p.lat.unique())
    dlon, dlat = np.diff(lons)[0], np.diff(lats)[0]
    grid = p.pivot(index="lat", columns="lon", values="probability").values
    le = np.append(lons - dlon / 2, lons[-1] + dlon / 2)
    la = np.append(lats - dlat / 2, lats[-1] + dlat / 2)
    m = np.ma.masked_where(np.nan_to_num(grid) <= 0, grid)
    pc = ax.pcolormesh(le, la, m, cmap=SEQ, shading="flat",
                       norm=mcolors.PowerNorm(0.45, vmin=0, vmax=1), zorder=1)
    cb = fig.colorbar(pc, ax=ax, shrink=0.86, pad=0.015, aspect=26)
    cb.set_label("probability the storm passes through this cell,\n"
                 "for %s only" % A.field, fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

drawn = []
handles = []
mesh = None
for i, spec in enumerate(A.run):
    if len(spec) not in (4, 5):
        raise SystemExit("--run takes STEM LAT LON LABEL [CLASS], got %d values"
                         % len(spec))
    stem, slat, slon, label = spec[:4]
    tr = pd.read_csv(os.path.join(A.gen, stem + "_tracks.csv"))
    if len(spec) == 5:
        cls = spec[4].upper()
        if cls not in CATC:
            raise SystemExit("unknown class %r; use one of %s"
                             % (cls, " ".join(CATC)))
    else:
        z = tr[tr.STEP == 0].WIND
        cls = classify(float(z.median()) if len(z) else 0.0)
    c = CATC[cls]
    focal = (not A.focus) or (stem == A.focus)
    if not focal and A.context == "off":
        drawn.append((label, "--", 0, tr.SID.nunique(), False))
        continue
    if not A.no_cores and not (not focal and A.context == "marker"):
        got = core(ax, A.gen, stem, tr, c, A.core_alpha,
                   cells=focal, full=bool(A.focus) and focal)
        mesh = got if got is not None else mesh
    keep = choose(tr, A.keep, A.seed + i) if focal else []
    for sid in keep:
        t = tr[tr.SID == sid].sort_values("STEP")
        ax.plot(t.LON, t.LAT, color=TRACK, lw=0.75, alpha=0.55, zorder=2 + i,
                solid_capstyle="round")
    if focal:
        draw_central(ax, *central_track(tr, A.central), colour=CENTRAL)
    # Ringed dot with a crosshair, as in plume_pair. A marker large enough to
    # see would otherwise cover the highest-probability cells on the map.
    la0, lo0 = float(slat), float(slon)
    h, = ax.plot([lo0], [la0], marker="o", ms=6.5, color=c,
                 markeredgecolor="white", markeredgewidth=1.2, zorder=7,
                 linestyle="none",
                 label="%s  %s%s" % (label, cls, "" if focal else "  (context)"))
    handles.append(h)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ax.plot([lo0 + dx * 0.9, lo0 + dx * 2.1],
                [la0 + dy * 0.9, la0 + dy * 2.1],
                color=c, lw=1.0, zorder=7, solid_capstyle="butt")
    drawn.append((label, cls, len(keep), tr.SID.nunique(), focal))

ax.contour(_lon, _lat, _land.astype(float), levels=[0.5], colors="#52514e",
           linewidths=0.6, zorder=3)
v = np.array(PAR + (PAR[0],))
ax.plot(v[:, 1], v[:, 0], color="#0b3a5c", lw=1.3, ls="--", zorder=5,
        label="PAR")

ax.set_xlim(*EXT[:2])
ax.set_ylim(*EXT[2:])
ax.set_aspect("equal")
ax.set_xticks(np.arange(115, EXT[1] + 0.1, 10))
ax.set_yticks(np.arange(EXT[2] + 3, EXT[3] + 0.1, 5))
ax.set_xticks(np.arange(EXT[0], EXT[1] + 0.1, 2.5), minor=True)
ax.set_yticks(np.arange(EXT[2], EXT[3] + 0.1, 2.5), minor=True)
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}°E")
ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}°N")
ax.grid(which="major", lw=0.5, color="#c9c9c9", zorder=0)
ax.grid(which="minor", lw=0.22, color="#ececec", zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel("longitude", fontsize=10, labelpad=6)
ax.set_ylabel("latitude", fontsize=10)
north_arrow(ax)
if mesh is not None and not A.field:
    # One bar for all three: the ramp and the stretch are identical, so a
    # single scale reads every core.
    cb = fig.colorbar(mesh, ax=ax, shrink=0.86, pad=0.015, aspect=26)
    cb.set_label("probability the storm passes through this cell", fontsize=9)
    cb.ax.tick_params(labelsize=8)
if A.legend_loc == "below":
    try:
        fig.legend(handles, [x.get_label() for x in handles],
                   loc="outside lower center",
                   ncol=min(4, len(handles)),
                   fontsize=8.5, frameon=True, edgecolor="#dcdbd6")
    except (ValueError, TypeError):          # matplotlib < 3.7
        ax.legend(handles, [x.get_label() for x in handles],
                  loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=min(4, len(handles)), fontsize=8.5, frameon=True,
                  edgecolor="#dcdbd6")
elif A.legend_loc == "right":
    try:
        fig.legend(handles, [x.get_label() for x in handles],
                   loc="outside right upper", fontsize=8.5,
                   frameon=True, edgecolor="#dcdbd6")
    except (ValueError, TypeError):
        ax.legend(handles, [x.get_label() for x in handles],
                  loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=8.5, frameon=True, edgecolor="#dcdbd6")
elif A.legend_loc != "off":
    ax.legend(fontsize=8.5, loc=A.legend_loc, frameon=True,
              edgecolor="#dcdbd6", framealpha=0.92)
if A.title:
    ax.set_title(A.title, loc="left", fontsize=11, fontweight="bold", pad=8)
if A.note:
    fig.suptitle(A.note.replace("\\n", "\n"), fontsize=8.4, family="monospace",
                 color="#1b1b1b", x=0.012, ha="left", va="top")

for ext in ("png", "pdf"):
    fig.savefig(f"{A.out}.{ext}", dpi=200, facecolor="white")
for label, cls, k, n, focal in drawn:
    tag = "" if focal else "   context (%s)" % A.context
    print(f"drawn: {k:>3} of {n:,} realisations   {cls:<4} {label}{tag}")
if A.focus:
    print("Focal display only. SynTC propagates each storm conditioned on its\n"
          "own state alone, so no interaction between these systems is modelled.")
print("Tracks from different origins are overlaid; the fields are not summed.")
