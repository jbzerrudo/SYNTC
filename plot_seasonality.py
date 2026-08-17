"""
Monthly seasonality of PAR tropical cyclones, observed against SynTC.

    python plot_seasonality.py --run ./run03 \
        --ibtracs /path/to/IBTrACS.WP.list.v04r01.points.csv \
        --dtm /path/to/dtm_phil_1km.tif

Writes seasonality.png, .pdf and seasonality.csv next to the run.

Why this script exists rather than the earlier one
--------------------------------------------------
The figure it replaces compared two different populations and read as a model
bias. Over 1977-2023, IBTrACS holds about 20.9 systems per season with at least
one track point inside the PAR hexagon, but RSMC Tokyo assigns an intensity
grade to only about 16.2 of them; the remainder are mostly weak circulations
tracked by JTWC and never graded by JMA. SynTC generates the graded population,
because a generator that assigns an intensity cannot generate storms for which
no intensity was ever analysed.

The old figure put all 20.9 on the observed line and 16.2 on the synthetic one.
That is a 23% gap in every month of the year, present before the model does
anything, and it is an artefact of which population each line counted rather
than a property of the generator.

This script takes both lines from the same source and the same filter:
`load_intensity_points`, which applies the grading and quality control, then the
PAR hexagon test. The counts it produces are therefore comparable by
construction, and if a monthly discrepancy remains it belongs to the model.

A storm is assigned to the month in which it FIRST entered PAR, so a storm
spanning a month boundary is counted once. Counting by track point instead would
weight long-lived storms more heavily and is a different quantity from the one
the caption claims.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import intensity as I
import terrain
import figstyle as FS
from syntc_ai import in_par

INK, MUTED, LINE, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
OBS, SYN = "#52514e", "#3987e5"
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def first_par_month(df, sid, month_col):
    """Month of each storm's first PAR track point, one row per storm."""
    return df.groupby(sid)[month_col].first()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ibtracs", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--titles", action="store_true",
                    help="draw the figure title and subtitle into the image; off by default so the LaTeX caption is the only caption")
    a = ap.parse_args()
    FS.TITLES = a.titles
    out = a.out or a.run
    terrain.DTM_PATH = a.dtm

    hist = I.load_intensity_points(a.ibtracs, season_max=2023)
    hist = hist[in_par(hist.lat.to_numpy(), hist.lon.to_numpy())].copy()
    hist = hist.sort_values(["SID", "time"])
    hist["month"] = hist.time.dt.month
    o_seasons = int(hist.SEASON.nunique())
    o_month = first_par_month(hist, "SID", "month")
    obs = np.array([(o_month == m).sum() for m in range(1, 13)]) / o_seasons

    files = sorted(glob.glob(os.path.join(a.run, "synthetic_storms_ens*.csv")))
    if not files:
        raise SystemExit(f"no ensemble CSVs in {a.run}")
    # Per ensemble, so the spread across ensembles can be drawn. That spread is
    # the honest uncertainty band for the model: it is what a 100-season
    # realisation actually varies by, not a standard error assuming
    # independence between the storms inside one.
    per_ens = []
    for f in files:
        d = pd.read_csv(f, usecols=["SID", "STEP", "LAT", "LON", "MONTH",
                                    "YEAR", "IN_PAR"])
        d = d[d.IN_PAR == 1].sort_values(["SID", "STEP"])
        n_yr = int(d.YEAR.nunique())
        m = first_par_month(d, "SID", "MONTH")
        per_ens.append(np.array([(m == k).sum() for k in range(1, 13)]) / n_yr)
    per_ens = np.vstack(per_ens)
    syn = per_ens.mean(axis=0)
    lo = np.percentile(per_ens, 5, axis=0)
    hi = np.percentile(per_ens, 95, axis=0)

    print(f"observed {obs.sum():.2f} storms/yr over {o_seasons} seasons | "
          f"SynTC {syn.sum():.2f} over {len(files)} ensembles")
    pd.DataFrame({"month": MONTHS, "observed_per_year": obs.round(3),
                  "syntc_per_year": syn.round(3),
                  "syntc_p05": lo.round(3), "syntc_p95": hi.round(3)}
                 ).to_csv(os.path.join(out, "seasonality.csv"), index=False)

    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(8.2, 3.9), facecolor=SURFACE)
    ax.fill_between(x, lo, hi, color=SYN, alpha=0.18, lw=0, zorder=1,
                    label="SynTC, 5th-95th percentile of ensembles")
    ax.plot(x, syn, color=SYN, lw=2.0, marker="o", ms=4.5, zorder=3,
            label=f"SynTC, {len(files)} ensembles")
    ax.plot(x, obs, color=OBS, lw=2.0, marker="s", ms=4.5, zorder=4,
            label=f"observed 1977-2023")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS, fontsize=8.5, color=MUTED)
    ax.set_ylabel("storms entering PAR per year", fontsize=9.5, color=MUTED)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=8.5, colors=MUTED, length=2)
    ax.grid(axis="y", lw=0.4, color="#e8e7e2", zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(LINE)
        s.set_linewidth(0.6)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    if FS.TITLES:
        ax.set_title("Seasonal cycle of PAR entry", fontsize=13, color=INK,
                     loc="left", pad=22)
        ax.text(0, 1.045,
                f"both lines are RSMC Tokyo graded storms counted at first PAR "
                f"entry: observed {obs.sum():.2f} per year, "
                f"SynTC {syn.sum():.2f}",
                transform=ax.transAxes, fontsize=8.5, color=MUTED,
                va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"seasonality.{ext}")
        fig.savefig(p, dpi=190, bbox_inches="tight", facecolor=SURFACE)
        print(f"  {p}")
    plt.close(fig)

    print(f"\n  {'month':<6}{'observed':>10}{'SynTC':>9}{'diff':>8}")
    for i, mn in enumerate(MONTHS):
        print(f"  {mn:<6}{obs[i]:>10.2f}{syn[i]:>9.2f}{syn[i]-obs[i]:>+8.2f}")


if __name__ == "__main__":
    main()
