"""Two-panel genesis plume: the probability field, overlaid with the spaghetti
of the realisations that run closest to the ridge.

RidgingVersion copy. The repo-root plume_pair.py, which draws manuscript
Figure 13, is unchanged and still uses the published core-following selection.

What changed here. The old rule ranked a realisation by the passage
probability of the cells it visited. In a bimodal plume that rewards the broad
recurving branch over the narrow westward corridor, because a wide fan covers
more moderately-probable cells than a tight channel does. Measured on a
17.3N 147.3E August field it put 5 of its 60 picks inside the corridor,
against 7 for a blind stride through the ensemble: no better than chance at
the job its own docstring claimed.

This version traces the ridge, the crest of the panel's own passage field, and
draws the N realisations that stay closest to it. Same helpers as the genesis
map, imported rather than copied so the two cannot drift apart.
"""
import argparse, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import rasterio

# ridge_path, along_ridge and closest_to_ridge live in the ridging genesis
# tool. Frozen, it sits beside this file inside the bundle; from source it sits
# beside it in RidgingVersion. Either way, this directory has to be importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import genesis_forecast as GF

SEQ = mcolors.LinearSegmentedColormap.from_list("heat", [
    "#ffffcc","#ffeda0","#fed976","#feb24c","#fd8d3c",
    "#fc4e2a","#e31a1c","#bd0026","#800026"])
TRACK_COLOR   = "#333333"   # realisations, dark grey. A red line is
                            # unreadable against the red end of the field.
CENTRAL_COLOR = "#1a5fd0"   # the ridge, one clean blue, no casing
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
_ap.add_argument("--keep-left",  type=int, default=None, help="overrides --keep for panel (a)")
_ap.add_argument("--keep-right", type=int, default=None, help="overrides --keep for panel (b)")
_ap.add_argument("--note-left",  default="", help="stats box for panel (a); use \\n for line breaks")
_ap.add_argument("--note-right", default="", help="stats box for panel (b)")
_ap.add_argument("--note-loc", default="lr", choices=("ll","lr","ul","ur"),
                 help="corner for the per-panel stats box: ll lr ul ur (default lr)")
_ap.add_argument("--note", default="",
                 help="figure-level stats, drawn once above both panels. Use this "
                      "when both panels are the same genesis run and the numbers "
                      "describe all realisations, not the drawn subset. \\n for breaks")
_ap.add_argument("--out", default="genesis_plume_pair")
_ap.add_argument("--pick", default="core", choices=("core", "random", "spread"),
                 help="how the drawn realisations are chosen. core (default) "
                      "ranks by the 10th percentile of passage probability "
                      "along the path, which selects tracks that never leave "
                      "the corridor and therefore understates the spread. "
                      "random draws an unbiased sample. spread clusters the "
                      "realisations by where they end up and draws one "
                      "representative per cluster, so the panel shows distinct "
                      "scenarios rather than one corridor.")
_ap.add_argument("--min-steps", type=int, default=16,
                 help="shortest realisation eligible to be drawn, in 6-hourly "
                      "steps. The default of 16 is four days and matches the "
                      "manuscript figure; lower it to let short-lived storms "
                      "into the sample.")
_ap.add_argument("--corridor", type=float, default=150.0,
                 help="half-width in km defining 'near the ridge'. 150 km is "
                      "about the radius of an average tropical cyclone and is "
                      "inside the plume's own measured spread. It is a chosen "
                      "scale, not calibrated against observed tracks.")
_ap.add_argument("--central", default="ridge",
                 choices=("ridge", "medoid", "mean", "median", "off"),
                 help="the thick line through the middle of the plume. medoid "
                      "(default) is a real realisation, the one that stays "
                      "closest to the ensemble median path, so it is always "
                      "physically possible. mean and median are per-step "
                      "statistics and can run between the branches of a "
                      "bimodal plume, through water no storm visited. off "
                      "is the DEFAULT here, because this script draws the "
                      "manuscript figure and its output must not change.")
_ap.add_argument("--cone", action="store_true",
                 help="overlay 50%% and 90%% position containment computed from "
                      "ALL realisations, not the drawn subset. This is the "
                      "quantity that cannot narrow with lead time.")
_ap.add_argument("--cone-color", default="#800026",
                 help="colour of the containment curves (default red)")
_ap.add_argument("--cone-alpha", type=float, default=0.75,
                 help="opacity of the containment curves (default 0.75)")
_ap.add_argument("--seed", type=int, default=0,
                 help="seed for --pick random and --pick spread")
_ap.add_argument("--ext", nargs=4, type=float, default=None,
                 metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                 help="map window. Defaults to the 112-145E, 2-28N frame the "
                      "manuscript figure uses. Widen it for a genesis point "
                      "outside that frame, otherwise every track point beyond "
                      "the frame is masked and the panels come out empty.")
A = _ap.parse_args()
if A.ext:
    EXT = tuple(A.ext)
NKEEP = A.keep
KEEP_L = A.keep_left  if A.keep_left  is not None else A.keep
KEEP_R = A.keep_right if A.keep_right is not None else A.keep
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

def endpoint(tr, sids, step=20):
    """Where each realisation is at `step`, or its last position if it died.

    Done in two passes over the frame rather than one filter per realisation,
    which matters at 2,000 tracks.
    """
    sub = tr[tr.SID.isin(set(sids))]
    last = (sub.sort_values("STEP").groupby("SID").tail(1)
               .set_index("SID")[["LON", "LAT"]])
    at = sub[sub.STEP == step].set_index("SID")[["LON", "LAT"]]
    out = last.copy()
    common = out.index.intersection(at.index)
    out.loc[common] = at.loc[common]
    return out


def choose(sc, tr, n):
    """Which realisations get drawn.

    `core` is the original rule and stays the default so the manuscript figure
    is unchanged. It ranks by the 10th percentile of passage probability along
    the path, which by construction excludes any realisation that diverges,
    because diverging means entering low-probability cells. The drawn spread is
    therefore much smaller than the ensemble's own, and can even contract with
    lead time, which no real ensemble does.

    `random` samples without replacement and is unbiased.

    `spread` clusters the realisations by where they are at day five and draws
    the member closest to each cluster centre, so the panel carries one track
    per distinct scenario instead of `n` versions of the same one.
    """
    n = min(n, len(sc))
    if n <= 0:
        return pd.Index([])
    if A.pick == "core":
        return sc.nlargest(n).index
    rng = np.random.default_rng(A.seed)
    if A.pick == "random":
        return pd.Index(rng.choice(sc.index.to_numpy(), n, replace=False))
    pts = endpoint(tr, sc.index)
    X = pts.reindex(sc.index).to_numpy(dtype=float)
    try:
        from scipy.cluster.vq import kmeans2
        cen, lab = kmeans2(X, n, minit="++", seed=A.seed, missing="warn")
    except Exception:
        return pd.Index(rng.choice(sc.index.to_numpy(), n, replace=False))
    picked = []
    for k in range(len(cen)):
        m = np.nonzero(lab == k)[0]
        if not len(m):
            continue
        d = np.hypot(*(X[m] - cen[k]).T)
        picked.append(sc.index[m[int(np.argmin(d))]])
    # An empty cluster leaves the panel short; top up at random from the rest.
    if len(picked) < n:
        rest = [s for s in sc.index if s not in set(picked)]
        picked += list(rng.choice(rest, min(n - len(picked), len(rest)),
                                  replace=False))
    return pd.Index(picked)


def cone(ax, tr, lons, lats, grid, lut, snap, dlon, dlat):
    """Envelopes that contain whole realisations, drawn from the full ensemble.

    A disc around the median position is the usual way to draw a cone, and it
    is wrong here: this ensemble is bimodal, one branch running west and one
    recurving, so a disc wide enough to hold both covers water no storm visits.

    Instead, score every realisation by the LOWEST passage probability along
    its own path. The region where the field exceeds the 50th percentile of
    those scores then contains half of all realisations along their entire
    path, by construction, and the 10th percentile gives ninety percent. The
    contour follows the probability field, so it splits when the field splits.

    Computed from every realisation, not the drawn subset, so it cannot narrow
    with lead time.
    """
    mins = []
    for _, t in tr.groupby("SID", sort=False):
        t = t.sort_values("STEP")
        x, y = t.LON.to_numpy(), t.LAT.to_numpy()
        w = (x >= EXT[0]) & (x <= EXT[1]) & (y >= EXT[2]) & (y <= EXT[3])
        if w.sum() < 2:
            continue
        pr = [lut.get((snap(a, lons[0], dlon), snap(b, lats[0], dlat)), 0.0)
              for a, b in zip(x[w], y[w])]
        mins.append(min(pr))
    if len(mins) < 20:
        return
    mins = np.asarray(mins)
    g = np.nan_to_num(grid, nan=0.0)
    # A plain red line vanishes against the red end of the YlOrRd field, so
    # every curve is drawn over a white stroke of its own. Both curves are
    # dashed and share one colour; width and dash length separate them.
    for frac, lw, ls, al in ((0.90, 0.8, (0, (1.4, 2.4)), 0.85),
                             (0.50, 0.9, (0, (5.0, 2.4)), 1.00)):
        lev = float(np.percentile(mins, 100.0 * (1.0 - frac)))
        if not np.isfinite(lev) or lev <= 0 or lev >= np.nanmax(g):
            continue
        cs = ax.contour(lons, lats, g, levels=[lev], colors=A.cone_color,
                        linewidths=lw, linestyles=[ls], zorder=4,
                        alpha=A.cone_alpha * al)
        halo = [pe.withStroke(linewidth=lw + 1.4, foreground="white",
                              alpha=0.85)]
        try:
            cs.set_path_effects(halo)          # matplotlib >= 3.8
        except AttributeError:
            for c in cs.collections:           # older matplotlib
                c.set_path_effects(halo)


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


def draw_central(ax, x, y, colour=CENTRAL_COLOR, lw=1.0, z=5.5):
    if x is None or len(x) < 2:
        return
    # One tone, no white casing. It thins where it crosses the darkest cells
    # near genesis; that is the price of dropping the halo and it was asked for.
    ax.plot(x, y, color=colour, lw=lw, zorder=z, solid_capstyle="round")


def panel(ax, tag, glat, glon, label, nkeep=None, note=""):
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
    # Ridge-following selection. The ridge is the crest of THIS panel's own
    # passage field, so nothing is carried in from elsewhere. The drawn tracks
    # are the nkeep realisations that run closest to it, taken from the ones
    # that stay within --corridor km. nkeep of 0 draws the field alone.
    nk = NKEEP if nkeep is None else nkeep
    rx, ry, why = GF.ridge_path(grid, le, la, glon, glat)
    keep, n_corr, n_tot = [], 0, int(tr.SID.nunique())
    if rx is not None:
        ids_all, n_tot = GF.along_ridge(tr, rx, ry, glon, glat,
                                        corridor=A.corridor)
        n_corr = len(ids_all)
        if nk > 0:
            keep = GF.closest_to_ridge(tr, rx, ry, glon, glat, nk,
                                       restrict=ids_all)
    if keep:
        sub = tr[tr.SID.isin(set(keep))].sort_values(["SID", "STEP"])
        for _sid, t in sub.groupby("SID", sort=False):
            ax.plot(t.LON, t.LAT, color=TRACK_COLOR, lw=0.75, alpha=0.55,
                    zorder=2, solid_capstyle="round")

    if A.cone:
        cone(ax, tr, lons, lats, grid, lut, snap, dlon, dlat)
    if A.central == "ridge":
        # the ridge already measured above; never recompute it from anything
        # that was itself selected using it
        draw_central(ax, rx, ry)
    else:
        draw_central(ax, *central_track(tr, A.central))
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
    if note:
        _x, _y, _ha, _va = {"ll": (0.022, 0.030, "left",  "bottom"),
                            "lr": (0.978, 0.030, "right", "bottom"),
                            "ul": (0.022, 0.970, "left",  "top"),
                            "ur": (0.978, 0.970, "right", "top")}[A.note_loc]
        ax.text(_x, _y, note.replace("\\n", "\n"), transform=ax.transAxes,
                fontsize=7.4, va=_va, ha=_ha, family="monospace",
                color="#1b1b1b", zorder=6,
                bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                          edgecolor="#8A6F4E", linewidth=0.7, alpha=0.90))
    return pc, len(keep), n_corr

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 5.2), facecolor="white",
                               constrained_layout=True)
pc, nL, tL = panel(axL, A.left,  A.left_pt[0],  A.left_pt[1],  A.left_label,
                   nkeep=KEEP_L, note=A.note_left)
_,  nR, tR = panel(axR, A.right, A.right_pt[0], A.right_pt[1], A.right_label,
                   nkeep=KEEP_R, note=A.note_right)
axL.set_ylabel("latitude", fontsize=10)
for ax in (axL, axR): ax.set_xlabel("longitude", fontsize=10)
cb = fig.colorbar(pc, ax=[axL, axR], shrink=0.86, pad=0.015, aspect=26)
cb.set_label("probability the storm passes through this cell", fontsize=9)
cb.ax.tick_params(labelsize=8)
if A.note:
    fig.suptitle(A.note.replace("\\n", "\n"), fontsize=8.4, family="monospace",
                 color="#1b1b1b", x=0.012, ha="left", va="top")
for ext in ("png","pdf"):
    fig.savefig(f"{A.out}.{ext}", dpi=200, facecolor="white")
print(f"drawn: {nL} of {tL} within {A.corridor:g} km of the ridge (a), "
      f"{nR} of {tR} (b); selection = closest to ridge")
