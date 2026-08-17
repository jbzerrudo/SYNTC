"""
Return level plot for the observed PAR annual maximum wind, drawn in the same
style as the manuscript's existing Figure 5.

    python plot_return_levels.py --ibtracs /path/to/IBTrACS.WP.list.v04r01.points.csv \
        --dtm /path/to/dtm_phil_1km.tif --out ./run02

That writes return_levels.png and .pdf: one panel, the observed record 1977-2023
against five candidate distributions. Nothing synthetic is involved and no run
folder is needed.

Add --compare together with --run or --from-csv to also write a SECOND file,
return_levels_comparison.png, holding the observed and SynTC Weibull fits with
their bootstrap bands:

    python plot_return_levels.py --ibtracs ... --dtm ... --run ./run02 --compare

Two files rather than two panels, on purpose. The observed figure answers which
distribution describes Philippine tropical cyclone maxima. That is a question
about the record, and it belongs wherever the record is described. The
comparison answers whether SynTC reproduces the record, which is a model result
and belongs with the other validation results. A model curve sitting inside the
data figure invites a reviewer to read the observed fit as something SynTC was
tuned against, which it was not.

--from-csv reuses a return_levels.csv written by return_levels.py instead of
refitting, so the figure and the table cannot disagree. IBTrACS is still needed
in that mode because the empirical points are not stored in the CSV.

Styling
-------
Matplotlib's default colour cycle in the order GEV, Gumbel, Weibull,
Exponential, Pareto, dashed lines for the fitted distributions, a solid black
line with open circles for the empirical maxima, filled red circles for the
tabulated return levels, a boxed legend headed "Distribution", and a log
return-period axis. All of that is carried over from the existing figure.

One deliberate change. The existing figure marks GEV return levels with the red
circles while its caption names Weibull as the best fit. The red circles here
mark the Weibull return levels instead, so the emphasised series and the chosen
distribution are the same one.

Empirical points use the Gringorten plotting position, p = (i - 0.44)/(n + 0.12)
with i the rank in ascending order, and T = 1/(1 - p). Gringorten is the usual
choice for extreme value work because it is closer to unbiased for the Gumbel
and Weibull families than the i/(n+1) position.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

import return_levels as RL
import terrain

# Matplotlib's default cycle, assigned in the same order as the existing
# figure's legend, so a reader comparing old and new sees the same colour for
# the same distribution.
COLOR = {
    "GEV":         "#1f77b4",
    "Gumbel":      "#ff7f0e",
    "Weibull":     "#2ca02c",
    "Exponential": "#d62728",
    "Pareto":      "#9467bd",
}
# Distinct dash patterns as well as distinct colours. On the observed record the
# GEV and Weibull fits very nearly coincide, 124.8 against 126.4 kt at 100 years
# and closer than that below it, so a shared dash pattern lets whichever is drawn
# second hide the other completely. Colour alone is not enough when two curves
# lie on top of each other.
DASH = {
    "GEV":         (6, 2),
    "Gumbel":      (2, 2),
    "Weibull":     (9, 2, 2, 2),
    "Exponential": (4, 3),
    "Pareto":      (1, 2),
}
ORDER = ["GEV", "Gumbel", "Weibull", "Exponential", "Pareto"]
TITLE_COLOR = "#1f3864"
GRID_COLOR = "#c8c8c8"
# The smallest Gringorten return period for n = 47 is 1.012 years, so the axis
# has to start below that or the lowest annual maxima fall off the left edge.
# Starting at 1.01 also puts the 10^0 tick back on the axis, as in the original.
TMIN, TMAX = 1.01, 200.0


def gringorten(sample):
    """Empirical return period for each annual maximum, ascending."""
    x = np.sort(np.asarray(sample, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    p = (i - 0.44) / (n + 0.12)
    return 1.0 / (1.0 - p), x


def curves_from_sample(sample, n_boot):
    """Dense fitted curves, the tabulated return levels, and a Weibull band."""
    T = np.geomspace(TMIN, TMAX, 240)
    probs = 1.0 - 1.0 / T
    out = {}
    for name in ORDER:
        dist, kw = RL.FITS[name]
        try:
            params = dist.fit(sample, **kw)
            out[name] = np.asarray(dist.ppf(probs, *params), dtype=float)
        except Exception:
            out[name] = np.full(len(T), np.nan)

    marks = (np.asarray(RL.RETURN_PERIODS, dtype=float),
             RL.return_levels_once(sample, "Weibull"))

    dist, kw = RL.FITS["Weibull"]
    rng = np.random.default_rng(0)
    boot = np.full((n_boot, len(T)), np.nan)
    for b in range(n_boot):
        try:
            p = dist.fit(rng.choice(sample, len(sample), replace=True), **kw)
            boot[b] = dist.ppf(probs, *p)
        except Exception:
            pass
    with np.errstate(all="ignore"):
        band = (np.nanpercentile(boot, 2.5, axis=0),
                np.nanpercentile(boot, 97.5, axis=0))
    return T, out, band, marks


def curves_from_csv(df):
    """Rebuild the curves from a stored return_levels.csv.

    The CSV holds each fit evaluated at twelve return periods. Return level
    functions are smooth and monotone in log T, so a monotone cubic through
    those twelve points is visually indistinguishable from the underlying fit
    and cannot introduce a spurious wiggle the way an unconstrained spline
    could.
    """
    t = df.return_period.to_numpy(dtype=float)
    T = np.geomspace(max(TMIN, t.min()), min(TMAX, t.max()), 240)
    lt = np.log(t)
    out = {}
    for name in ORDER:
        y = df[name].to_numpy(dtype=float)
        ok = np.isfinite(y)
        out[name] = (PchipInterpolator(lt[ok], y[ok])(np.log(T))
                     if ok.sum() > 2 else np.full(len(T), np.nan))
    band = tuple(PchipInterpolator(lt, df[f"Weibull_{k}"].to_numpy(float))(np.log(T))
                 for k in ("lo", "hi"))
    return T, out, band, (t, df["Weibull"].to_numpy(dtype=float))


def frame(ax, ylim, ylabel=True, xlim=None):
    """xlim is passed explicitly because in --from-csv mode the fitted curves
    can only be rebuilt over the return periods the CSV tabulates, the smallest
    of which is 1.5 years. Leaving the axis at TMIN then leaves a strip at the
    left where the empirical points appear with no curves behind them, which
    reads as a plotting failure rather than as a limit of the stored table."""
    ax.set_xscale("log")
    ax.set_xlim(*(xlim or (TMIN, TMAX)))
    ax.set_ylim(*ylim)
    ax.grid(which="major", ls="--", lw=0.6, color=GRID_COLOR, alpha=0.8, zorder=0)
    ax.grid(which="minor", ls="--", lw=0.4, color=GRID_COLOR, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, length=3, width=0.8)
    ax.set_xlabel("Return Period (years)", fontsize=10)
    if ylabel:
        ax.set_ylabel("Maximum Wind Speed (kts)", fontsize=10)
    for s in ax.spines.values():
        s.set_color("#333333")
        s.set_linewidth(0.9)


def draw_fits(ax, T, curves):
    for name in ORDER:
        ax.plot(T, curves[name], lw=1.6, color=COLOR[name], dashes=DASH[name],
                label=name, zorder=4)


def combined_legend(ax, extra_handles, extra_labels):
    """One boxed legend with two headed groups, as in the existing figure.

    Matplotlib has no native section headings, so the 'Empirical' heading is an
    invisible handle whose label text is emboldened afterwards. Keeping it in a
    single box matters here: two separate boxes need two free corners, and on a
    log return-period axis with five rising curves there is only one.
    """
    blank = Line2D([], [], ls="none", marker="none")
    handles = [Line2D([], [], lw=1.6, color=COLOR[n], dashes=DASH[n]) for n in ORDER]
    labels = list(ORDER)
    handles += [blank, blank] + list(extra_handles)
    labels += ["", "Empirical"] + list(extra_labels)
    lg = ax.legend(handles, labels, title="Distribution", fontsize=8.5,
                   title_fontsize=9, loc="upper left", framealpha=0.95,
                   edgecolor="#999999", borderpad=0.6, handlelength=2.2)
    lg.get_title().set_fontweight("bold")
    lg.get_texts()[len(ORDER) + 1].set_fontweight("bold")
    return lg


def notes(ax, T, curves, ymax, noun):
    """Stack the caveats in the free bottom-right corner.

    Two things need saying and neither should be left for the reader to guess.

    A curve that leaves the frame is a result, not a clipping error, so it is
    named.

    A fit that has collapsed onto the sample maximum is a failure of maximum
    likelihood, not a property of the data: for a bounded-tail GEV or
    generalised Pareto the estimator can drive the upper endpoint onto the
    largest observation, after which the fitted return level stops depending on
    the return period and the curve goes dead flat.

    The pinning test is relative rather than absolute: a fit is called pinned
    if it rises less than a quarter as much as the Weibull fit does over the
    last decade of return period. An absolute threshold in knots would pass the
    synthetic Pareto, which creeps up about a knot from 20 to 200 years while
    Weibull climbs five, and is plainly pinned despite not being flat.
    """
    def rise(y):
        return abs(np.interp(200.0, T, y) - np.interp(20.0, T, y))

    lines = []
    ref = rise(curves["Weibull"])
    flat = [n for n in ("GEV", "Pareto")
            if np.isfinite(curves[n]).all() and rise(curves[n]) < 0.25 * ref]
    if flat:
        y = float(np.interp(200.0, T, curves[flat[0]]))
        lines.append(f"{' and '.join(flat)} pinned to the {noun} maximum "
                     f"({y:.1f} kt)")
    gone = [n for n in ORDER if np.nanmax(curves[n]) > ymax]
    if gone:
        lines.append("above axis: " + ", ".join(gone))
    if lines:
        ax.text(0.985, 0.025, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, color="#555555", ha="right", va="bottom",
                style="italic", linespacing=1.5)


def annotate_100(ax, T, curves):
    v = float(np.interp(100.0, T, curves["Weibull"]))
    ax.axvline(100, color="#555555", lw=0.8, ls=":", zorder=3)
    ax.annotate(f"{v:.1f} kt", (100, v), xytext=(9, 9),
                textcoords="offset points", fontsize=10,
                color=COLOR["Weibull"], ha="left", fontweight="bold",
                zorder=9)
    return v


def save(fig, out, stem):
    png = os.path.join(out, f"{stem}.png")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"written: {png}\n         {png.replace('.png', '.pdf')}")


def observed_figure(obs, T_o, c_o, b_o, m_o, out):
    """The single-panel figure: the observed record against five candidates."""
    top = max(np.nanmax(c_o[n]) for n in ORDER)
    ylim = (50.0, float(top) * 1.05)

    fig, ax = plt.subplots(figsize=(7.8, 5.6), facecolor="white")
    draw_fits(ax, T_o, c_o)
    t, x = gringorten(obs)
    ax.plot(t, x, "-o", lw=1.3, color="black", ms=5, mfc="white", mec="black",
            mew=1.0, zorder=6)
    ax.plot(*m_o, "o", color="red", ms=6, zorder=7)
    frame(ax, ylim, xlim=(float(T_o.min()), float(T_o.max())))
    annotate_100(ax, T_o, c_o)
    notes(ax, T_o, c_o, ylim[1], "observed record")
    combined_legend(
        ax,
        [Line2D([], [], ls="-", lw=1.3, color="black", marker="o", ms=5,
                mfc="white", mec="black"),
         Line2D([], [], ls="none", marker="o", ms=6, color="red")],
        ["Observed Max Wind Speeds", "Weibull Return Levels"])
    ax.set_title("Return Level Plot: Maximum Wind Speed vs. Return Period",
                 fontsize=13, color=TITLE_COLOR, fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, out, "return_levels")


def comparison_figure(obs, T_o, c_o, b_o, T_s, c_s, b_s, run_label, out):
    """Observed against SynTC, Weibull only, on a zoomed axis.

    Deliberately a separate file rather than a panel of the figure above. The
    observed figure answers which distribution describes Philippine tropical
    cyclone maxima, which is a question about the record and belongs wherever
    the record is described. This one answers whether SynTC reproduces that
    record, which is a model result and belongs with the other validation
    results. Putting a model result inside a data figure invites a reviewer to
    read the observed fit as something SynTC was tuned against.

    The zoom is necessary because the two curves agree to a fraction of a knot,
    which is invisible on an axis wide enough to hold the exponential fit. The
    two curves also get different dash patterns as well as different colours,
    because where they agree the upper one would otherwise hide the lower.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.4), facecolor="white")
    v = {}
    for T, c, b, colour, dash, key, lab in (
        (T_o, c_o, b_o, "black", (None, None), "obs", "Observed 1977-2023"),
        (T_s, c_s, b_s, COLOR["Weibull"], (6, 2.5), "syn", f"SynTC {run_label}"),
    ):
        ax.fill_between(T, b[0], b[1], color=colour, alpha=0.15, lw=0, zorder=2)
        v[key] = float(np.interp(100.0, T, c["Weibull"]))
        ax.plot(T, c["Weibull"], color=colour, lw=2.2, dashes=dash, zorder=6,
                label=f"{lab}   {v[key]:.1f} kt")
        ax.plot([100], [v[key]], "o", ms=6, color=colour, zorder=7,
                markeredgecolor="white", markeredgewidth=0.9)
    t, x = gringorten(obs)
    ax.plot(t, x, "o", ms=4.5, mfc="white", mec="black", mew=0.9, zorder=5,
            alpha=0.7, label="Observed Max Wind Speeds")
    ax.axvline(100, color="#555555", lw=0.8, ls=":", zorder=3)
    frame(ax, (95.0, 140.0))
    ax.text(0.985, 0.04,
            f"difference at 100 years: {v['syn'] - v['obs']:+.1f} kt",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            fontweight="bold")
    lg = ax.legend(title="Weibull, 95% bootstrap band", fontsize=9,
                   title_fontsize=9.5, loc="upper left", framealpha=0.95,
                   edgecolor="#999999", borderpad=0.6)
    lg.get_title().set_fontweight("bold")
    ax.set_title("Observed and SynTC 100-year Return Levels",
                 fontsize=13, color=TITLE_COLOR, fontweight="bold", pad=12)
    fig.tight_layout()
    save(fig, out, "return_levels_comparison")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibtracs", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--compare", action="store_true",
                    help="also write return_levels_comparison.png, the "
                         "observed against SynTC Weibull overlay; needs "
                         "--run or --from-csv")
    ap.add_argument("--run", default=None, help="a SynTC run folder")
    ap.add_argument("--from-csv", default=None,
                    help="a return_levels.csv written by return_levels.py")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.compare and not (a.run or a.from_csv):
        raise SystemExit("--compare needs --run or --from-csv")

    terrain.DTM_PATH = a.dtm
    out = a.out or a.run or "."
    os.makedirs(out, exist_ok=True)

    # The observed figure never needs a run folder. Refit from IBTrACS unless a
    # stored CSV was given, in which case reuse it so the figure and the table
    # cannot disagree.
    obs = RL.observed_maxima(a.ibtracs)
    if a.from_csv:
        d = pd.read_csv(a.from_csv)
        T_o, c_o, b_o, m_o = curves_from_csv(d[d.source.str.contains("observed")])
    else:
        T_o, c_o, b_o, m_o = curves_from_sample(obs, a.bootstrap)
    observed_figure(obs, T_o, c_o, b_o, m_o, out)

    if a.compare:
        if a.run:
            syn = RL.synthetic_maxima(a.run)
            T_s, c_s, b_s, _ = curves_from_sample(syn, a.bootstrap)
            run_label = os.path.basename(a.run.rstrip(os.sep))
        else:
            ds = d[d.source.str.startswith("synthetic")]
            ds = ds[ds.source == ds.source.iloc[0]]   # drop any half-split rows
            T_s, c_s, b_s, _ = curves_from_csv(ds)
            run_label = ds.source.iloc[0].split("(")[-1].rstrip(")")
        comparison_figure(obs, T_o, c_o, b_o, T_s, c_s, b_s, run_label, out)


if __name__ == "__main__":
    main()
