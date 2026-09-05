"""
fig_eva_summary.pdf: the extreme value analysis in one three-panel figure.

    python make_eva_summary.py --ibtracs <IBTrACS.WP.list.v04r01.points.csv> \
        --run <run07 folder> --dtm dtm_phil_1km.tif

Replaces two separate manuscript figures, fig_annual_maxima and return_levels,
with a single row that reads left to right as one argument:

  (a) the sample        what PAR annual maximum wind looks like, observed
                        against the catalog
  (b) the family        which of five candidate distributions describes the
                        observed sample
  (c) the test          whether the catalog reproduces the observed return
                        level curve under the selected family

Panels (a) and (c) are the existing fig_annual_maxima panels, unchanged.
Panel (b) is the existing return_levels figure with its y-axis clipped.

Why panel (b) is clipped
------------------------
The original axis runs to whatever the highest fitted curve reaches, which is
the exponential fit at about 254 kt at 200 years. Every quantity a reader cares about lives
between 75 and 128 kt, so on that axis the observed points and the GEV, Weibull
and Pareto fits collapse into one trace in the bottom quarter of the frame and
nothing can be told apart. Clipping to 70-145 kt spreads them out. Two curves
then leave the top of the frame, which is a result rather than a plotting
failure. Which two, and where they reach, is stated in the caption and
printed to stdout by this script; it is not drawn into the artwork.

Nothing is refitted here. The fits are the same maximum likelihood solves on
the same 47 annual maxima that return_levels.py performs, obtained by importing
that module rather than by reimplementing it, so this figure and Table 1 cannot
disagree.

Which curves do what
--------------------
Worth stating because it is easy to get backwards, and the clipped axis is what
makes it visible:

    GEV          124.8 kt at 100 yr, flat beyond    pinned to the sample max
    Gumbel       162.1 kt at 100 yr, still rising   overestimates
    Weibull      126.4 kt at 100 yr                 tracks the record
    Exponential  230.3 kt at 100 yr, still rising   overestimates
    Pareto       125.0 kt at 100 yr, flat           pinned to the sample max

Pareto does not run away; its maximum likelihood solution collapses onto the
largest observation, 125 kt, and stays there, and GEV very nearly does the
same. Gumbel and Exponential are the two that leave the frame.

Options
-------
--gringorten  draw the panel (c) empirical points with the Gringorten plotting
              position, the one Section 2.3 of the manuscript states, instead
              of the Weibull position the current panel uses. Off by default so
              that panel (c) reproduces the existing figure exactly.
--titles      draw in-artwork titles, for browsing. Off for the manuscript.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

import terrain

# ----------------------------------------------------------------- arguments
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ibtracs", required=True)
ap.add_argument("--run", required=True, help="the stationary catalog folder, e.g. run07")
ap.add_argument("--dtm", default="dtm_phil_1km.tif")
ap.add_argument("--out", default=".")
ap.add_argument("--gringorten", action="store_true",
                help="use the Gringorten plotting position in panel (c) too")
ap.add_argument("--stem", default="fig_eva_summary")
ap.add_argument("--narrow", action="store_true",
                help="9.8 x 3.7 in, for an upright \\begin{figure}[t] at "
                     "width=\\textwidth. Default is 13.0 x 4.2 in, for "
                     "\\begin{sidewaysfigure} at width=\\textheight.")
A = ap.parse_args()
terrain.DTM_PATH = A.dtm

import intensity as I                      # noqa: E402  (needs DTM_PATH first)
from syntc_ai import in_par                # noqa: E402

# ------------------------------------------------------------------- styling
GRID = "#c8c8c8"
OBS = "#000000"
SYN = "#2ca02c"

# Same colours and dash patterns as plot_return_levels.py, so a reader holding
# the old figure next to the new one sees the same curve in the same colour.
COLOR = {"GEV": "#1f77b4", "Gumbel": "#ff7f0e", "Weibull": "#2ca02c",
         "Exponential": "#d62728", "Pareto": "#9467bd"}
DASH = {"GEV": (6, 2), "Gumbel": (2, 2), "Weibull": (9, 2, 2, 2),
        "Exponential": (4, 3), "Pareto": (1, 2)}
ORDER = ["GEV", "Gumbel", "Weibull", "Exponential", "Pareto"]
FITS = {"GEV": (stats.genextreme, {}), "Gumbel": (stats.gumbel_r, {}),
        "Weibull": (stats.weibull_min, {"floc": 0}),
        "Exponential": (stats.expon, {}), "Pareto": (stats.genpareto, {})}
RETURN_PERIODS = (1.5, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200)

# The cas-sc text block measures 466.6 pt, or 6.48 in. A sideways figure at
# width=\textheight is reduced by about 8.9/13.0 = 0.68, and the upright
# --narrow variant at width=\textwidth by 6.48/9.8 = 0.66. Both land in the
# same place, so every font here is set about 1.5x larger than it would be for
# a figure printed at its drawn size, which puts axis labels near 8 pt on the
# page and tick labels just under. Elsevier's floor is 7 pt.
#
# Three panels need the wide form. At 9.8 in each panel is 3.3 in across and
# the legends, at a size that survives the reduction, cover the curves they
# are labelling.
FIGSIZE = (9.8, 4.5) if A.narrow else (13.0, 5.0)
FS_LABEL, FS_TICK, FS_LEG, FS_NOTE, FS_PANEL = 11.5, 10.0, 9.0, 8.6, 13.0
plt.rcParams.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
                     "axes.labelsize": FS_LABEL})

# Panels (b) and (c) share these exactly. They plot the same 47 observed
# maxima; different limits would make one dataset look like two.
XLIM = (1.05, 260.0)
YLIM = (70.0, 145.0)


def grid(ax):
    ax.grid(which="major", ls="--", lw=0.6, color=GRID, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def below(ax, ncol):
    """Legend under the panel, outside the axes.

    Inside the frame there is nowhere safe to put one. Panel (b) rises from the
    bottom left to the top right, so an upper-left legend sits on the Gumbel
    and exponential curves over the return periods where they separate from the
    rest, which is the part of the panel the figure exists to show. Below the
    axis the legend cannot cover anything.
    """
    ax.legend(fontsize=FS_LEG - 1.0, ncol=ncol, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.17),
              handlelength=2.0, columnspacing=1.4, handletextpad=0.5,
              labelspacing=0.3)


def panel_letter(ax, s):
    ax.text(0.012, 1.02, s, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="bottom")


def gringorten(sample):
    x = np.sort(np.asarray(sample, float))
    n = len(x)
    i = np.arange(1, n + 1)
    return 1.0 / (1.0 - (i - 0.44) / (n + 0.12)), x


def weibull_pp(sample):
    x = np.sort(np.asarray(sample, float))
    n = len(x)
    i = np.arange(1, n + 1)
    return (n + 1) / (n + 1 - i), x


# ---------------------------------------------------------------------- data
o = I.load_intensity_points(A.ibtracs, season_max=2023)
o = o[in_par(o.lat.to_numpy(), o.lon.to_numpy())]
obs = np.sort(o.groupby("SEASON").vmax_raw.max().to_numpy().astype(float))

files = sorted(glob.glob(os.path.join(A.run, "synthetic_storms_ens*.csv")))
if not files:
    sys.exit(f"no ensemble CSVs in {A.run}")
syn = np.sort(np.concatenate([
    pd.read_csv(f, usecols=["YEAR", "WIND", "IN_PAR"])
      .query("IN_PAR == 1").groupby("YEAR").WIND.max().to_numpy()
    for f in files]).astype(float))

print(f"observed  n = {len(obs):4d}  median {np.median(obs):.1f} kt  max {obs.max():.0f} kt")
print(f"synthetic n = {len(syn):4d}  median {np.median(syn):.1f} kt  max {syn.max():.0f} kt")

# ------------------------------------------------------------------- figure
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=FIGSIZE, facecolor="white")

# ---- (a) densities -------------------------------------------------------
bins = np.arange(65, 146, 5)
a1.hist(obs, bins=bins, density=True, color=OBS, alpha=0.22,
        label=f"observed 1977-2023 (n = {len(obs)})")
a1.hist(syn, bins=bins, density=True, histtype="step", lw=1.8, color=SYN,
        label=f"SynTC, 20 ensembles (n = {len(syn):,})")
for v, c in ((np.median(obs), OBS), (np.median(syn), SYN)):
    a1.axvline(v, color=c, lw=1.1, ls="--", zorder=5)
a1.annotate(f"medians {np.median(syn):.1f} / {np.median(obs):.1f} kt",
            xy=(np.median(obs), a1.get_ylim()[1] * 0.52), xytext=(9, 0),
            textcoords="offset points", ha="left", fontsize=FS_NOTE, color="#444444")
a1.set_xlabel("PAR annual maximum wind (kt)", fontsize=FS_LABEL)
a1.set_ylabel("Density", fontsize=FS_LABEL)
below(a1, ncol=1)
panel_letter(a1, "(a)")
grid(a1)

# ---- (b) five candidate distributions on the observed record -------------
T = np.geomspace(*XLIM, 400)
P = 1.0 - 1.0 / T
curves = {}
for name in ORDER:
    dist, kw = FITS[name]
    params = dist.fit(obs, **kw)
    curves[name] = np.asarray(dist.ppf(P, *params), float)
    a2.plot(T, curves[name], lw=2.4 if name == "Weibull" else 1.5,
            color=COLOR[name], dashes=DASH[name], label=name,
            zorder=5 if name == "Weibull" else 4)

tg, xg = gringorten(obs)
a2.plot(tg, xg, "-o", lw=1.2, color=OBS, ms=4.5, mfc="white", mec=OBS, mew=1.0,
        zorder=6, label="observed, empirical")
wb, wkw = FITS["Weibull"]
a2.plot(RETURN_PERIODS,
        stats.weibull_min.ppf([1 - 1 / t for t in RETURN_PERIODS],
                              *stats.weibull_min.fit(obs, **wkw)),
        "o", color="red", ms=5, zorder=7, label="Weibull return levels")

# Name every curve that leaves the frame, at the height it leaves it, with
# where it actually goes. A curve running off the top is a result.
# Which fits leave the frame is stated in the LaTeX caption, not drawn here.
# Printing it twice sets the same words in a font the journal did not choose,
# makes them uneditable at proof stage, and puts bold colour on the two
# candidates the panel exists to reject. Reported to stdout so the caption and
# the artwork cannot drift apart.
for name in ORDER:
    if np.nanmax(curves[name]) > YLIM[1]:
        print(f"  leaves the {YLIM[1]:.0f} kt frame: {name:12s} "
              f"at T = {float(T[np.argmax(curves[name] > YLIM[1])]):5.1f} yr, "
              f"reaching {float(np.interp(200.0, T, curves[name])):.0f} kt at 200 yr")

pinned = [n for n in ("GEV", "Pareto")
          if abs(np.interp(200., T, curves[n]) - np.interp(20., T, curves[n]))
          < 0.25 * abs(np.interp(200., T, curves["Weibull"])
                       - np.interp(20., T, curves["Weibull"]))]
if pinned:
    a2.text(0.985, 0.035,
            f"{' and '.join(pinned)} pinned to the observed maximum "
            f"({obs.max():.1f} kt)",
            transform=a2.transAxes, fontsize=FS_NOTE - 0.6, color="#555555", ha="right",
            va="bottom", style="italic")

a2.axvline(100, color="#555555", lw=0.8, ls=":", zorder=3)
v100 = float(np.interp(100.0, T, curves["Weibull"]))
a2.annotate(f"{v100:.1f} kt", (100, v100), xytext=(-6, 9),
            textcoords="offset points", fontsize=FS_NOTE + 1.5, color=COLOR["Weibull"],
            ha="right", fontweight="bold", zorder=9)
a2.set_xscale("log")
a2.set_xlim(*XLIM)
a2.set_ylim(*YLIM)
a2.set_xlabel("Return period (years)", fontsize=FS_LABEL)
a2.set_ylabel("Annual maximum wind (kt)", fontsize=FS_LABEL)
below(a2, ncol=3)
panel_letter(a2, "(b)")
grid(a2)

# ---- (c) observed against SynTC, Weibull only ---------------------------
pos = gringorten if A.gringorten else weibull_pp
Tobs, yobs = pos(obs)
Tsyn, ysyn = pos(syn)
a3.semilogx(Tobs, yobs, "o", ms=4.5, mfc="white", mec=OBS, mew=1.0, zorder=6,
            label="observed, empirical")
a3.semilogx(Tsyn, ysyn, "-", lw=1.8, color=SYN, zorder=5, label="SynTC, empirical")
Tc = np.geomspace(*XLIM, 400)
Pc = 1 - 1 / Tc
for x, c, lab in ((obs, OBS, "observed, Weibull"), (syn, SYN, "SynTC, Weibull")):
    pr = stats.weibull_min.fit(x, floc=0)
    a3.semilogx(Tc, stats.weibull_min.ppf(Pc, *pr), ls=(0, (5, 2)), lw=1.4,
                color=c, alpha=0.85, zorder=4, label=lab)
a3.axvline(100, color="#555555", lw=0.8, ls=":", zorder=3)
o100 = stats.weibull_min.ppf(0.99, *stats.weibull_min.fit(obs, floc=0))
s100 = stats.weibull_min.ppf(0.99, *stats.weibull_min.fit(syn, floc=0))
a3.annotate(f"100-yr: {o100:.1f} obs, {s100:.1f} SynTC", xy=(100, 74),
            xytext=(-6, 0), textcoords="offset points", fontsize=FS_NOTE,
            color="#444444", va="center", ha="right")
a3.set_xlabel("Return period (years)", fontsize=FS_LABEL)
a3.set_ylabel("Annual maximum wind (kt)", fontsize=FS_LABEL)
a3.set_xlim(*XLIM)
a3.set_ylim(*YLIM)
below(a3, ncol=2)
panel_letter(a3, "(c)")
grid(a3)

# Identical ticks on (b) and (c), not merely identical limits.
a3.set_xticks(a2.get_xticks())
a3.set_xticks(a2.get_xticks(minor=True), minor=True)
a3.set_yticks(a2.get_yticks())
a3.set_xlim(*XLIM)
a3.set_ylim(*YLIM)

fig.tight_layout()
for ext in ("pdf", "png"):
    p = os.path.join(A.out, f"{A.stem}.{ext}")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    print("written:", p)
plt.close(fig)
