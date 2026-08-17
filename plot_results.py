"""
Standard figure set for a SynTC run.

    python plot_results.py --run ./run02 \
        --ibtracs /path/to/IBTrACS.WP.list.v04r01.points.csv \
        --dtm /path/to/dtm_phil_1km.tif

Produces three figures in the run folder:

  hotspots_by_category.png   observed against synthetic, one column per PAGASA
                             intensity class
  hotspots_by_month.png      the same for the twelve calendar months
  skill_summary.png          the Murphy skill scores from validate_hotspots.py,
                             which is the result worth putting in a paper
  intensity_distribution.png observed against synthetic wind distribution

Design notes
------------
Density is a magnitude, so it is encoded with a SEQUENTIAL ramp that rises
monotonically in darkness, light to dark. No rainbow. The default is
ColorBrewer YlOrRd, matching the ArcGIS hotspot maps produced by this project;
--cmap blue gives the single-hue blue alternative.

Every panel is a SHARE, not a count: each cell holds the fraction of that
subset's track points that fall in it, so the panel sums to one. The synthetic
catalogue has twenty ensembles against one historical record, so raw counts
differ by a factor of about forty and a shared count scale would render every
synthetic panel uniformly dark while telling you nothing.

Each panel is then stretched to its own 99th percentile, so colour shows the
PATTERN and nothing else. Absolute darkness is not comparable between an
observed panel and a synthetic one: 47 seasons carry Poisson noise that inflates
the peak cell and 2,000 seasons do not, and subsampling the catalogue to the
observed count closes most of that gap. The quantitative comparison is therefore
the spatial correlation and Murphy skill printed on each synthetic panel, read
from spatial_validation.csv at the same grid.

Titles are off by default. A journal figure carries its caption in LaTeX; pass
--titles when browsing a run folder.

Skill scores are polarity, not magnitude, so they get a DIVERGING scale with a
neutral midpoint at zero. Zero is the meaningful boundary: below it the
synthetic field agrees with history worse than two halves of history agree with
each other.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import intensity as I
import terrain
import figstyle as FS
from syntc_ai import in_par

# Density ramps, selected with --cmap. Default is ColorBrewer YlOrRd, the
# yellow-to-red used by the ArcGIS hotspot maps this project has always
# produced, so the matplotlib and ArcGIS figures in the same paper read as one
# set. It is a properly ordered sequential scheme rather than a rainbow: it
# rises monotonically in darkness as well as in hue, so it survives greyscale
# printing and does not invent a boundary where the data has none.
#
# White is prepended so an empty cell is background rather than pale yellow,
# which matters on these panels because most cells are empty in the quiet
# months.
#
# There is deliberately no blue-to-red option for these panels. Blue-to-red is a
# DIVERGING scheme: it encodes a meaningful midpoint with two directions away
# from it, and a reader who knows the convention will look for what the neutral
# colour means. Track density has no midpoint. Zero storms is one end of the
# scale, not the middle of it, so a diverging ramp would assert a structure the
# quantity does not have and leave the mid-density cells looking neutral.
#
# Blue-to-red IS used in this paper, in skill_summary, where the quantity is
# Murphy skill: it has a real zero at the bootstrap noise floor, positive means
# the synthetic field beats that floor and negative means it does not. That is
# what a diverging scale is for, and it is the DIV ramp below.
#
# "deep" is the option to reach for if the heat ramp runs out of contrast at the
# top: it continues past dark red into purple, staying monotonic in lightness,
# which a rainbow does not.
RAMPS = {
    "heat": ["#ffffff", "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
             "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"],
    "deep": ["#ffffff", "#fcffa4", "#fac228", "#f57d15", "#d44842",
             "#9f2a63", "#65156e", "#280b53", "#000004"],
    "blue": ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"],
}
SEQ = LinearSegmentedColormap.from_list("seq", RAMPS["heat"])


def set_cmap(name):
    global SEQ
    SEQ = LinearSegmentedColormap.from_list("seq", RAMPS[name])
# Diverging blue-red with a neutral grey midpoint, for signed skill.
DIV = LinearSegmentedColormap.from_list(
    "div_br", ["#8c2d1c", "#d1603f", "#e8b4a3", "#f0efec",
               "#9ec5f4", "#3987e5", "#184f95"])
INK, MUTED, LINE, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"

CATEGORIES = (("TD", 22, 34), ("TS", 34, 48), ("STS", 48, 64),
              ("TY", 64, 100), ("STY", 100, 1e9))
# PAGASA class colours as used in the manuscript's original Figure 3a:
# TD blue, TS yellow, STS orange, TY red, STY purple. Kept here so the
# regenerated panel and the historical panel beside it match exactly.
CLASS_COLOURS = (("TD", 22, 34, "#5B9BD5"),
                 ("TS", 34, 48, "#FFC000"),
                 ("STS", 48, 64, "#ED7D31"),
                 ("TY", 64, 100, "#C00000"),
                 ("STY", 100, 1e9, "#7030A0"))
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
# Cell size, set by --grid. The two resolutions trade different things and
# validate_hotspots.py scores both, so the choice is recorded rather than
# argued: at 1 degree the median occupied cell in a single calendar month holds
# 3.2 observed track points and 70% of occupied cells hold fewer than five, so
# the observed row is largely Poisson noise (55% on a count of three) and the
# Murphy skill is lower in nine months of twelve. At 2 degrees the median cell
# holds 11.4 points and the noise falls to 30%. The synthetic row has forty
# times the sample and is well resolved either way; the binding constraint is
# the length of the record, not the size of the catalogue.
GRID = 1.0
LON_E = LAT_E = None


def set_grid(deg):
    global GRID, LON_E, LAT_E
    GRID = float(deg)
    LON_E = np.arange(115.0, 135.0 + GRID, GRID)
    LAT_E = np.arange(5.0, 25.0 + GRID, GRID)


set_grid(GRID)


def density(lat, lon):
    h, _, _ = np.histogram2d(lon, lat, bins=[LON_E, LAT_E])
    return h


def par_outline(ax, cfg_vertices):
    v = np.array(cfg_vertices + (cfg_vertices[0],))
    ax.plot(v[:, 1], v[:, 0], color="#2b2a28", lw=0.7, ls="--",
            zorder=4)


_COAST = {}


def coastline(ax, zorder=2):
    """Draw the Philippine coastline from the DTM.

    A density field on a bare grid gives the reader no landmark: the whole
    point of these panels is where the storms are relative to the islands, and
    without a coastline that has to be inferred from the axis ticks. Contoured
    once and cached, subsampled 10:1 because a 1 km raster renders identically
    at this scale and costs seconds at full resolution.
    """
    if "xy" not in _COAST:
        try:
            tx = terrain.get()
        except Exception:
            _COAST["xy"] = None
            return
        s = 10
        land = tx.is_land[::s, ::s]
        ny, nx = land.shape
        _COAST["xy"] = (tx.left + (np.arange(nx) + 0.5) * s * tx.transform.a,
                        tx.top + (np.arange(ny) + 0.5) * s * tx.transform.e,
                        land.astype(float))
    if _COAST["xy"] is None:
        return
    lon, lat, land = _COAST["xy"]
    ax.contour(lon, lat, land, levels=[0.5], colors="#3d3c39",
               linewidths=0.45, zorder=zorder)


def style(ax):
    ax.set_xlim(114.5, 135.5)
    ax.set_ylim(4.5, 25.5)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7, colors=MUTED, length=2, width=0.5)
    for s in ax.spines.values():
        s.set_color(LINE)
        s.set_linewidth(0.6)


def panel_grid(obs_sets, syn_sets, labels, title, subtitle, path, vertices,
               wrap=None, stats=None):
    """One column per subset: observed on top, synthetic below, shared scale
    within each column.

    With `wrap` set, the columns are broken into blocks of that many and the
    blocks are stacked, each keeping its own observed/synthetic row pair. Twelve
    columns on one line is 20 inches wide; reduced to a journal column each
    panel is half an inch across and the reader can see nothing. Six over six
    doubles the linear size of every panel on the same page.

    The observed and synthetic rows stay adjacent within a block. The comparison
    this figure exists to make is vertical, and separating the two halves would
    put six columns between a panel and the one it should be read against.
    """
    n = len(labels)
    wrap = wrap or n
    blocks = [range(i, min(i + wrap, n)) for i in range(0, n, wrap)]
    rows = 2 * len(blocks)
    fig, axes = plt.subplots(rows, wrap, figsize=(1.55 * wrap + 1.6, 2.3 * rows),
                             facecolor=SURFACE, squeeze=False)
    for b, block in enumerate(blocks):
        r0 = 2 * b
        for c, j in enumerate(block):
            # Each panel is the share of that subset's points per cell, and each
            # panel is then stretched to its own 99th percentile. Colour carries
            # the PATTERN; the agreement between the two patterns is carried by
            # the numbers printed on the panel, not by the colour.
            #
            # Two earlier versions were both wrong, in opposite directions.
            # Rescaling the synthetic field to the observed maximum forced the
            # two panels to agree at the peak by construction. Giving the pair
            # one shared scale then made the synthetic look systematically weak,
            # but that is an artefact of record length rather than a model
            # error: 47 seasons of observations have Poisson noise that inflates
            # the peak cell, and a 2,000-season field does not. Subsampling the
            # catalogue to the observed count closes most of the gap, the peak
            # ratio falling from 1.2-6.1 to 0.96-2.9 and to within 10% in the
            # peak season. So neither panel's absolute darkness means anything
            # a reader should act on, and the honest figure says so by putting
            # the quantitative claim in a number instead of in the colour bar.
            #
            # The 99th percentile rather than the maximum, so one noisy cell in
            # a 47-season record does not set the stretch for the whole panel.
            fields = []
            for f in (obs_sets[j], syn_sets[j]):
                f = f / max(f.sum(), 1e-9)
                nz = f[f > 0]
                fields.append((f, float(np.percentile(nz, 99)) if nz.size else 1.0))
            for i, (field, vmax) in enumerate(fields):
                ax = axes[r0 + i][c]
                ax.pcolormesh(LON_E, LAT_E, field.T, cmap=SEQ, vmin=0,
                              vmax=vmax, shading="flat", rasterized=True)
                coastline(ax)
                par_outline(ax, vertices)
                style(ax)
                if i == 0:
                    ax.set_title(labels[j], fontsize=9.5, color=INK, pad=4)
                    ax.set_xticklabels([])
                if c:
                    ax.set_yticklabels([])
            st = (stats or {}).get(labels[j])
            if st:
                r, sk = st
                ax = axes[r0 + 1][c]
                ax.text(0.035, 0.035, f"$r$ {r:.2f}\nS {sk:+.2f}",
                        transform=ax.transAxes, fontsize=7.5, color=INK,
                        va="bottom", ha="left", zorder=6, linespacing=1.25,
                        bbox=dict(boxstyle="square,pad=0.28", facecolor="white",
                                  edgecolor=LINE, linewidth=0.5))
        axes[r0][0].set_ylabel("observed", fontsize=9, color=INK)
        axes[r0 + 1][0].set_ylabel("SynTC", fontsize=9, color=INK)
        for c in range(len(block), wrap):        # short final block
            axes[r0][c].set_axis_off()
            axes[r0 + 1][c].set_axis_off()
    fig.tight_layout(rect=[0, 0, 1, 0.90] if FS.TITLES else None, h_pad=0.6)
    FS.title(fig, title, subtitle, x=0.02, y=1.05)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path}")


def skill_figure(val_csv, path):
    d = pd.read_csv(val_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1.7]})
    for ax, kind, order in (
        (axes[0], "category", ["TD", "TS", "STS", "TY", "STY", "All"]),
        (axes[1], "month", list(MONTHS)),
    ):
        sub = d[d.kind == kind]
        grids = sorted(sub.grid.unique())
        width = 0.8 / len(grids)
        for gi, g in enumerate(grids):
            s = sub[sub.grid == g].set_index("name").reindex(order)
            x = np.arange(len(order)) + (gi - (len(grids) - 1) / 2) * width
            colour = "#3987e5" if gi == 0 else "#184f95"
            ax.bar(x, s.skill.to_numpy(), width=width * 0.9, color=colour,
                   label=f"{g:g}° grid", zorder=2)
        ax.axhline(0, color=INK, lw=1.0, zorder=3)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, fontsize=8.5, color=MUTED)
        ax.tick_params(axis="y", labelsize=8, colors=MUTED, length=2)
        ax.grid(axis="y", lw=0.4, color="#e8e7e2", zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(LINE)
            sp.set_linewidth(0.6)
        ax.set_ylim(min(-0.1, d.skill.min() * 1.15), max(1.0, d.skill.max() * 1.15))
    axes[0].set_ylabel("Murphy skill score", fontsize=9, color=MUTED)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    FS.title(fig, "Spatial skill against a bootstrap noise floor",
             "above zero, the synthetic field matches history better than two "
             "random halves of history match each other", x=0.02, y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.93] if FS.TITLES else None)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path}")


def intensity_figure(obs, syn, path, obs_label="observed", syn_label="SynTC"):
    """Observed and synthetic wind distributions, drawn the same way.

    Two panels rather than one, and the same encoding in both: class-coloured
    bars on identical axes and identical bins. An earlier version put the
    observations in bars and the model in a step line over the top. That is a
    figure-ground relationship, not a comparison of equals: it invites the
    reader to treat the bars as truth and the line as a deviation from it, and
    it made the panel beside it redundant, since that panel showed the same
    observed distribution again.

    Sharing the y-axis is the point. Any difference in height is a real
    difference in the share of track points in that bin, not a scaling artefact.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), facecolor=SURFACE,
                             sharey=True, sharex=True)
    bins = np.arange(20, 145, 5)
    centres = bins[:-1] + 2.5
    for ax, v, tag in ((axes[0], obs, f"(a) {obs_label}"),
                       (axes[1], syn, f"(b) {syn_label}")):
        h, _ = np.histogram(v, bins=bins, density=True)
        for name, lo, hi, colour in CLASS_COLOURS:
            m = (centres >= lo) & (centres < hi)
            ax.bar(centres[m], h[m], width=5, color=colour, zorder=2,
                   edgecolor="white", linewidth=0.4,
                   label=(f"{name} ({lo}-{hi - 1} kt)" if hi < 1e9
                          else f"{name} ({lo}+ kt)"))
        ax.set_title(tag, fontsize=10.5, color=INK, loc="left", pad=6)
        ax.set_xlabel("10-minute sustained wind (kt)", fontsize=9.5,
                      color=MUTED)
        ax.set_xlim(20, 145)
        ax.tick_params(labelsize=8.5, colors=MUTED, length=2)
        ax.grid(axis="y", lw=0.4, color="#e8e7e2", zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(LINE)
            s.set_linewidth(0.6)
        ax.text(0.97, 0.94, f"n = {len(v):,}", transform=ax.transAxes,
                fontsize=8, color=MUTED, ha="right", va="top")
        ax.text(0.97, 0.86, f"mean {np.mean(v):.1f} kt",
                transform=ax.transAxes, fontsize=8, color=MUTED, ha="right",
                va="top")
    axes[0].set_ylabel("density", fontsize=9.5, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    axes[1].legend(h, l, fontsize=7.5, frameon=False, loc="upper right",
                   bbox_to_anchor=(1.0, 0.82), handlelength=1.4)
    if FS.TITLES:
        fig.suptitle("Intensity distribution inside PAR", fontsize=12.5,
                     color=INK, x=0.012, ha="left", y=1.03)

    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ibtracs", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid", type=float, default=1.0,
                    help="cell size in degrees; validate_hotspots.py scores "
                         "1 and 2, and the printed r and skill are read at "
                         "whichever is chosen here")
    ap.add_argument("--cmap", choices=sorted(RAMPS), default="heat",
                    help="density ramp: heat (ColorBrewer YlOrRd, matches the "
                         "ArcGIS maps) or blue")
    ap.add_argument("--titles", action="store_true",
                    help="draw titles into the image; off by default "
                         "so the LaTeX caption is the only caption")
    a = ap.parse_args()
    FS.TITLES = a.titles
    set_cmap(a.cmap)
    set_grid(a.grid)
    out = a.out or a.run
    os.makedirs(out, exist_ok=True)
    terrain.DTM_PATH = a.dtm

    from syntc_ai import CONFIG
    verts = CONFIG.par_vertices

    # Spatial correlation and Murphy skill, printed on each synthetic panel.
    # Read from validate_hotspots.py at the same grid the panels are drawn on,
    # rather than recomputed here, so the figure and the validation table can
    # never quote different numbers for the same quantity.
    stats, val = {}, os.path.join(a.run, "spatial_validation.csv")
    if os.path.exists(val):
        v = pd.read_csv(val)
        v = v[np.isclose(v.grid, GRID)]
        if len(v):
            stats = {r["name"]: (r["r"], r["skill"]) for _, r in v.iterrows()}
        else:
            print(f"  (no skill scores at {GRID:g} deg in spatial_validation.csv; "
                  f"rerun validate_hotspots.py)")
    else:
        print("  (no spatial_validation.csv; panels will carry no scores)")

    hist = I.load_intensity_points(a.ibtracs, season_max=2023)
    hist = hist[in_par(hist.lat.to_numpy(), hist.lon.to_numpy())].copy()
    hist["month"] = hist.time.dt.month
    hist["wind"] = hist.vmax_raw

    files = sorted(glob.glob(os.path.join(a.run, "synthetic_storms_ens*.csv")))
    if not files:
        raise SystemExit(f"no ensemble CSVs in {a.run}")
    syn = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    syn = syn[syn.IN_PAR == 1].rename(
        columns={"LAT": "lat", "LON": "lon", "WIND": "wind", "MONTH": "month"})
    print(f"historical {len(hist):,} points | synthetic {len(syn):,} points "
          f"from {len(files)} ensembles")
    print("writing:")

    labels = [c[0] for c in CATEGORIES]
    o = [density(hist[(hist.wind >= lo) & (hist.wind < hi)].lat,
                 hist[(hist.wind >= lo) & (hist.wind < hi)].lon)
         for _, lo, hi in CATEGORIES]
    s = [density(syn[(syn.wind >= lo) & (syn.wind < hi)].lat,
                 syn[(syn.wind >= lo) & (syn.wind < hi)].lon)
         for _, lo, hi in CATEGORIES]
    panel_grid(o, s, labels, "Track density by intensity class",
               f"{GRID:g} degree grid, PAR hexagon dashed. Each panel is the share of "
               f"that class's track points per cell, stretched to its own 99th "
               f"percentile: colour shows pattern, agreement is the printed "
               f"$r$ and Murphy skill S",
               os.path.join(out, "hotspots_by_category.png"), verts,
               stats=stats)

    o = [density(hist[hist.month == m].lat, hist[hist.month == m].lon)
         for m in range(1, 13)]
    s = [density(syn[syn.month == m].lat, syn[syn.month == m].lon)
         for m in range(1, 13)]
    panel_grid(o, s, list(MONTHS), "Track density by month",
               f"{GRID:g} degree grid, PAR hexagon dashed. Each panel is the share of "
               f"that month's track points per cell, stretched to its own 99th "
               f"percentile: colour shows pattern, agreement is the printed "
               f"$r$ and Murphy skill S",
               os.path.join(out, "hotspots_by_month.png"), verts, wrap=6,
               stats=stats)

    yr0, yr1 = int(hist.SEASON.min()), int(hist.SEASON.max())
    intensity_figure(hist.wind.to_numpy(), syn.wind.to_numpy(),
                     os.path.join(out, "intensity_distribution.png"),
                     obs_label=f"observed {yr0}-{yr1}",
                     syn_label=f"SynTC, {len(files)} ensembles")

    val = os.path.join(a.run, "spatial_validation.csv")
    if os.path.exists(val):
        skill_figure(val, os.path.join(out, "skill_summary.png"))
    else:
        print(f"  (skipped skill_summary.png: run validate_hotspots.py first)")


if __name__ == "__main__":
    main()
