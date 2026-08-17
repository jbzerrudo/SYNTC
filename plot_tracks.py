"""
Track geometry: observed against synthetic, and how it moves with the season.

    python plot_tracks.py --run ./run03 \
        --ibtracs /path/to/IBTrACS.WP.list.v04r01.points.csv \
        --dtm /path/to/dtm_phil_1km.tif

Writes two figures next to the run:

  tracks_observed_vs_syntc.png/.pdf   the same number of tracks, side by side
  seasonal_shift.png/.pdf             median track latitude by month

Why the track counts are matched
--------------------------------
The catalogue holds forty times as many storms as the record. Drawing all of
them beside 47 seasons of observations produces a synthetic panel that is solid
ink and an observed panel that is sparse, and the reader learns the ratio of the
two sample sizes rather than anything about the tracks. This script draws a
random sample of synthetic tracks of exactly the observed size, from a fixed
seed, so the two panels are the same kind of picture and can be compared by eye.
The sample is stated in the caption and the seed is a command-line argument, so
a reader who suspects a flattering draw can change it.

The seasonal shift figure
-------------------------
For each calendar month, the median latitude of track points in each 5 degree
longitude band. This is the quantity that carries the seasonal migration of the
subtropical ridge: storms run west near 10N in the late season and near 20N at
the height of it. SynTC is never told this. Month enters the propagator only as
a sine-cosine pair among its inputs, so the migration either emerges from the
fitted conditional density or it does not, and the figure is the test.

Bands holding fewer than ten observed points are dropped rather than plotted,
so the observed curves stop where the record stops supporting them.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import intensity as I
import terrain
import figstyle as FS
from syntc_ai import in_par

INK, MUTED, LINE, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
# Cyclic ramp, because month 12 is adjacent to month 1. A sequential ramp would
# put December and January at opposite ends of the colour bar and invent a
# discontinuity that the calendar does not have.
CYC = LinearSegmentedColormap.from_list(
    "cyc", ["#184f95", "#3987e5", "#7fb069", "#e0a458", "#b3402a",
            "#8c4a7d", "#184f95"])
# PAR longitudes in 2.5 degree bands. The seasonal-shift panel is computed on
# track points INSIDE PAR only, which is the domain the paper is about and the
# same subset the hotspot figures use.
#
# An earlier version of this script used every point of any storm that entered
# PAR, out to 150E, and reported a corridor 6 to 10 degrees too far north. That
# was the script, not the model: restricted to PAR the monthly medians agree
# within 0.4 degrees from May to December. The discrepancy came from what
# happens outside PAR, where 36.4% of synthetic points lie north of 25N against
# 14.7% observed, because SynTC has no extratropical transition and keeps
# recurved storms alive too long. That is a real defect and is reported as one,
# but it is a track-termination problem, not a shift in the corridor, and
# folding it into this figure would have mislabelled it.
LON_BANDS = np.arange(115.0, 135.1, 2.5)
MIN_PTS = 10


def coastline(ax, dtm, zorder=3):
    try:
        tx = terrain.get(dtm)
    except Exception:
        return
    s = 10
    land = tx.is_land[::s, ::s]
    ny, nx = land.shape
    ax.contour(tx.left + (np.arange(nx) + 0.5) * s * tx.transform.a,
               tx.top + (np.arange(ny) + 0.5) * s * tx.transform.e,
               land.astype(float), levels=[0.5], colors="#8a8884",
               linewidths=0.5, zorder=zorder)


def par_outline(ax, verts, zorder=4):
    v = np.array(tuple(verts) + (verts[0],))
    ax.plot(v[:, 1], v[:, 0], color="#b3402a", lw=1.1, ls="--", zorder=zorder)


def map_frame(ax, dtm, verts, title):
    coastline(ax, dtm)
    par_outline(ax, verts)
    # The PAR bounding box with a small margin. The tracks are clipped to the
    # hexagon, so a basin-wide frame would be mostly empty.
    ax.set_xlim(113.5, 136.5)
    ax.set_ylim(3.5, 26.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=6)
    ax.set_xlabel("longitude (E)", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED, length=2)
    ax.grid(lw=0.4, color="#eeeeee", zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(LINE)
        s.set_linewidth(0.6)


# PAGASA classes, weakest to strongest. Thresholds are identical to
# classify_category() in csv2pts2segments.py, so this figure and the ArcGIS
# products classify a storm the same way.
#
# The colours are NOT taken from the ArcGIS symbology, because the symbology
# lives in the layer file rather than in any script and is not in this
# repository. These are the same warm ramp as the hotspot maps. Replace the hex
# values here to match a .lyrx exactly.
CLASSES = (("TD", 22, 34, "#f7d08a"), ("TS", 34, 48, "#f0a03c"),
           ("STS", 48, 64, "#e3712a"), ("TY", 64, 100, "#c8341f"),
           ("STY", 100, 1e9, "#7d0d0d"))


def par_segments(lat, lon, inside):
    """Split a track into the contiguous runs of points inside PAR.

    A storm that leaves the hexagon and comes back must not be drawn with a
    straight line joining the two visits: that segment is a path the storm did
    not take through PAR, and on a figure showing only the clipped tracks it
    would be indistinguishable from one it did.
    """
    out, i, n = [], 0, len(inside)
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and inside[j + 1]:
            j += 1
        if j > i:
            out.append((lat[i:j + 1], lon[i:j + 1]))
        i = j + 1
    return out


def spaghetti(ax, storms, lw=0.7, alpha=0.55):
    """Draw PAR-clipped tracks, coloured by peak class reached inside PAR.

    Weakest first, so a super typhoon is never buried under the depressions
    that outnumber it four to one.
    """
    counts = {}
    for k, (lo, hi, colour) in enumerate(
            (c[1:] for c in CLASSES)):
        name = CLASSES[k][0]
        n = 0
        for segs, peak in storms:
            if lo <= peak < hi:
                n += 1
                for la, ln in segs:
                    ax.plot(ln, la, color=colour, lw=lw, alpha=alpha,
                            zorder=2 + k, solid_capstyle="round")
        counts[name] = n
    for name, lo, hi, colour in CLASSES:
        ax.plot([], [], color=colour, lw=2.2,
                label=f"{name} ({counts[name]})")
    return counts


def median_latitude(lon, lat, month, months=range(1, 13)):
    """Median latitude per longitude band, per month."""
    out = {}
    for m in months:
        sel = month == m
        x, y = [], []
        for a, b in zip(LON_BANDS[:-1], LON_BANDS[1:]):
            k = sel & (lon >= a) & (lon < b)
            if k.sum() < MIN_PTS:
                continue
            x.append((a + b) / 2)
            y.append(np.median(lat[k]))
        if len(x) >= 2:
            out[m] = (np.array(x), np.array(y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ibtracs", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the matched synthetic track sample")
    ap.add_argument("--titles", action="store_true",
                    help="draw the figure title and subtitle into the image; off by default so the LaTeX caption is the only caption")
    a = ap.parse_args()
    FS.TITLES = a.titles
    out = a.out or a.run
    terrain.DTM_PATH = a.dtm
    from syntc_ai import CONFIG
    verts = CONFIG.par_vertices

    hist = I.load_intensity_points(a.ibtracs, season_max=2023).sort_values(
        ["SID", "time"])
    hist["month"] = hist.time.dt.month
    par_sids = set(hist.loc[in_par(hist.lat.to_numpy(), hist.lon.to_numpy()),
                            "SID"])
    hist = hist[hist.SID.isin(par_sids)]
    o_seasons = int(hist.SEASON.nunique())

    files = sorted(glob.glob(os.path.join(a.run, "synthetic_storms_ens*.csv")))
    if not files:
        raise SystemExit(f"no ensemble CSVs in {a.run}")
    # SIDs repeat across ensembles, so tag each with its ensemble index and
    # group on the pair. Grouping on SID alone would splice twenty unrelated
    # storms into one track.
    syn = pd.concat(
        [pd.read_csv(f, usecols=["SID", "STEP", "LAT", "LON", "MONTH",
                                 "WIND", "IN_PAR"]).assign(ENS=i)
         for i, f in enumerate(files)], ignore_index=True)
    syn["UID"] = syn.ENS.astype(str) + ":" + syn.SID.astype(str)
    keep = set(syn.loc[syn.IN_PAR == 1, "UID"])
    syn = syn[syn.UID.isin(keep)].sort_values(["UID", "STEP"])

    n_obs = hist.SID.nunique()
    rng = np.random.default_rng(a.seed)
    uids = syn.UID.unique()
    pick = set(rng.choice(uids, size=min(n_obs, len(uids)), replace=False))
    sample = syn[syn.UID.isin(pick)]
    print(f"observed {n_obs} storms over {o_seasons} seasons | "
          f"synthetic {len(uids):,} available, {sample.UID.nunique()} drawn "
          f"with seed {a.seed}")

    # ---- figure 1: matched spaghetti ---------------------------------
    def clipped(df, sid, la, lo, wd, inside_col=None):
        out = []
        for _, g in df.groupby(sid, sort=False):
            lat, lon = g[la].to_numpy(), g[lo].to_numpy()
            ins = (g[inside_col].to_numpy() == 1 if inside_col
                   else in_par(lat, lon))
            segs = par_segments(lat, lon, ins)
            if segs:
                out.append((segs, float(g[wd].to_numpy()[ins].max())))
        return out

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4), facecolor=SURFACE)
    co = spaghetti(axes[0], clipped(hist, "SID", "lat", "lon", "vmax_raw"))
    cs = spaghetti(axes[1], clipped(sample, "UID", "LAT", "LON", "WIND",
                                    "IN_PAR"))
    map_frame(axes[0], a.dtm, verts,
              f"(a) observed, {sum(co.values())} storms, {o_seasons} seasons")
    map_frame(axes[1], a.dtm, verts,
              f"(b) SynTC, {sum(cs.values())} storms drawn from {len(uids):,}")
    for ax in axes:
        ax.legend(fontsize=7.5, frameon=True, edgecolor=LINE,
                  facecolor="white", framealpha=0.92, loc="lower left",
                  title="peak class in PAR", title_fontsize=7.5)
    print("\n  peak class inside PAR (count of storms)")
    print(f"  {'class':<6}{'observed':>10}{'SynTC':>9}")
    for name, *_ in CLASSES:
        print(f"  {name:<6}{co[name]:>10}{cs[name]:>9}")
    axes[0].set_ylabel("latitude (N)", fontsize=9, color=MUTED)
    FS.title(fig, "Tracks of storms inside PAR",
             "clipped to the hexagon and coloured by the strongest class each "
             "storm reached inside it; equal numbers of storms in each panel")
    fig.tight_layout(rect=FS.rect(0.95))
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"tracks_observed_vs_syntc.{ext}")
        fig.savefig(p, dpi=190, bbox_inches="tight", facecolor=SURFACE)
        print(f"  {p}")
    plt.close(fig)

    # ---- figure 2: seasonal shift ------------------------------------
    hp = hist[in_par(hist.lat.to_numpy(), hist.lon.to_numpy())]
    sp = syn[syn.IN_PAR == 1]
    om = median_latitude(hp.lon.to_numpy(), hp.lat.to_numpy(),
                         hp.month.to_numpy())
    sm = median_latitude(sp.LON.to_numpy(), sp.LAT.to_numpy(),
                         sp.MONTH.to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), facecolor=SURFACE,
                             sharey=True)
    for ax, data, lab in ((axes[0], om, "(a) observed 1977-2023"),
                          (axes[1], sm, "(b) SynTC")):
        for m, (x, y) in sorted(data.items()):
            ax.plot(x, y, color=CYC((m - 1) / 12), lw=1.9, marker="o", ms=3.4,
                    zorder=3, label=MONTHS[m - 1])
        ax.set_title(lab, fontsize=10.5, color=INK, loc="left", pad=6)
        ax.set_xlabel("longitude (E)", fontsize=9, color=MUTED)
        ax.tick_params(labelsize=8, colors=MUTED, length=2)
        ax.grid(lw=0.4, color="#e8e7e2", zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(LINE)
            s.set_linewidth(0.6)
    axes[0].set_ylabel("median latitude of track points (N)", fontsize=9,
                       color=MUTED)
    axes[1].legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left",
                   handlelength=1.4, columnspacing=1.0)
    FS.title(fig, "Seasonal migration of the track corridor",
             "median latitude of PAR track points in 2.5 degree longitude "
             "bands. Month reaches the propagator only as a sine-cosine pair; "
             "the migration is fitted, not prescribed.", y=1.045)
    fig.tight_layout(rect=FS.rect())
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"seasonal_shift.{ext}")
        fig.savefig(p, dpi=190, bbox_inches="tight", facecolor=SURFACE)
        print(f"  {p}")
    plt.close(fig)

    rows = []
    for m in range(1, 13):
        if m in om and m in sm:
            xs = np.intersect1d(om[m][0], sm[m][0])
            if len(xs):
                oi = np.interp(xs, *om[m]); si = np.interp(xs, *sm[m])
                rows.append((MONTHS[m - 1], float(np.mean(oi)),
                             float(np.mean(si)), float(np.mean(si - oi))))
    d = pd.DataFrame(rows, columns=["month", "observed_lat", "syntc_lat",
                                    "bias_deg"])
    d.round(2).to_csv(os.path.join(out, "seasonal_shift.csv"), index=False)
    print(f"\n  corridor latitude, mean over shared longitude bands")
    print(f"  {'month':<6}{'observed':>10}{'SynTC':>9}{'bias':>8}")
    for _, r in d.iterrows():
        print(f"  {r.month:<6}{r.observed_lat:>10.1f}{r.syntc_lat:>9.1f}"
              f"{r.bias_deg:>+8.1f}")
    print(f"\n  seasonal range: observed {d.observed_lat.min():.1f} to "
          f"{d.observed_lat.max():.1f} N, SynTC {d.syntc_lat.min():.1f} to "
          f"{d.syntc_lat.max():.1f} N")
    print(f"  mean absolute bias {d.bias_deg.abs().mean():.2f} deg, "
          f"largest {d.bias_deg.abs().max():.2f} deg in "
          f"{d.loc[d.bias_deg.abs().idxmax(), 'month']}")

    # Reported here rather than plotted: the model keeps recurved storms alive
    # too long, which is a track-termination defect and belongs in the
    # limitations, not in a figure about the seasonal corridor.
    o_n = 100 * (hist.lat.to_numpy() > 25).mean()
    s_n = 100 * (syn.LAT.to_numpy() > 25).mean()
    print(f"\n  track points north of 25N: observed {o_n:.1f}%, "
          f"SynTC {s_n:.1f}%  (no extratropical transition is modelled)")


if __name__ == "__main__":
    main()
