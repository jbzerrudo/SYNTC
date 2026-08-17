"""
The archipelago as a filter: what land does to a storm that crosses it.

    python plot_filtering.py --run ./run03 \
        --ibtracs /path/to/IBTrACS.WP.list.v04r01.points.csv \
        --dtm /path/to/dtm_phil_1km.tif

Writes filtering_effect.png and .pdf next to the run.

This is the figure for the claim the paper has always wanted to make and, in
earlier versions, made by assertion: that the Philippines strips intensity from
storms that pass over it. It is now a measurement rather than a constraint. No
overland intensity cap is imposed anywhere in SynTC, so the weakening shown here
is what the terrain decay relation and the intensity ceiling produce on their
own, and the synthetic and observed curves can therefore disagree. Whether they
do is the point of the figure.

Panel (a) is every crossing storm as one point: intensity entering land against
intensity leaving it. Distance below the 1:1 line is the loss. A storm that
crossed without weakening would sit on the line.

Panel (b) is the same information as a fractional loss against entry intensity,
with the observed and synthetic medians drawn through it. Reading it this way
makes the intensity dependence visible, which panel (a) compresses.

Storms are counted as crossings only if they left land to the west of where they
arrived, so a storm that clipped a coast and turned back out to sea is not
counted as having crossed the archipelago. Storms that dissipated over land
never produce an exit wind and appear in neither panel; they are reported in the
printed summary from filtering_effect.py instead.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import filtering_effect as F
import terrain
import figstyle as FS

INK, MUTED, LINE, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
OBS, SYN = "#52514e", "#3987e5"
BANDS = np.array([0, 34, 48, 64, 80, 100, 120, 160])


def median_curve(entry, value, edges=BANDS):
    """Median of `value` in each entry band, plus the interquartile range.

    Banded rather than smoothed because the observed sample is small and a
    smoother would invent structure between the few storms that exist at high
    entry intensity. Bands with fewer than three storms are dropped rather than
    plotted, so the curve stops where the record stops supporting it.
    """
    x, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (entry >= a) & (entry < b)
        if m.sum() < 3:
            continue
        v = value[m]
        x.append(entry[m].mean())
        med.append(np.median(v))
        lo.append(np.percentile(v, 25))
        hi.append(np.percentile(v, 75))
    return np.array(x), np.array(med), np.array(lo), np.array(hi)


def frame(ax):
    ax.tick_params(labelsize=8.5, colors=MUTED, length=2)
    ax.grid(lw=0.4, color="#e8e7e2", zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(LINE)
        s.set_linewidth(0.6)


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

    oc, o_seasons, _ = F.observed(a.ibtracs, a.dtm)
    sc, s_seasons, n_ens = F.synthetic(a.run)
    oc = oc[oc.fate == "crossing"].dropna(subset=["exit"])
    sc = sc[sc.fate == "crossing"].dropna(subset=["exit"])
    print(f"observed {len(oc)} crossings over {o_seasons} seasons | "
          f"synthetic {len(sc):,} over {s_seasons:,} ({n_ens} ensembles)")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), facecolor=SURFACE)

    ax = axes[0]
    lim = (0, max(oc.entry.max(), sc.entry.max()) * 1.06)
    ax.plot(lim, lim, color=INK, lw=0.9, ls=(0, (5, 3)), zorder=4,
            label="no weakening")
    ax.scatter(sc.entry, sc["exit"], s=3, color=SYN, alpha=0.10,
               linewidths=0, zorder=2, label=f"SynTC ({len(sc):,})")
    ax.scatter(oc.entry, oc["exit"], s=17, facecolor="none", edgecolor=OBS,
               linewidths=0.9, zorder=3, label=f"observed ({len(oc)})")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("intensity entering land (kt)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("intensity leaving land (kt)", fontsize=9.5, color=MUTED)
    ax.set_title("(a) every crossing storm", fontsize=10.5, color=INK,
                 loc="left", pad=6)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    frame(ax)

    ax = axes[1]
    for c, colour, lab in ((sc, SYN, "SynTC"), (oc, OBS, "observed")):
        x, med, lo, hi = median_curve(c.entry.to_numpy(),
                                      c.loss_pct.to_numpy())
        ax.fill_between(x, lo, hi, color=colour, alpha=0.13, lw=0, zorder=1)
        ax.plot(x, med, color=colour, lw=1.9, zorder=3,
                marker="o", ms=4, label=f"{lab}, median")
    ax.set_xlabel("intensity entering land (kt)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("intensity lost crossing (%)", fontsize=9.5, color=MUTED)
    ax.set_title("(b) fractional loss, with interquartile range",
                 fontsize=10.5, color=INK, loc="left", pad=6)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    frame(ax)

    FS.title(fig, "The Philippine archipelago as an intensity filter",
             f"storms whose centre crossed land and emerged to the west; "
             f"observed {o_seasons} seasons against SynTC {s_seasons:,}. "
             f"No overland cap is imposed: this is what the model produces.",
             y=1.045)
    fig.tight_layout(rect=FS.rect())
    for ext in ("png", "pdf"):
        p = os.path.join(out, f"filtering_effect.{ext}")
        fig.savefig(p, dpi=190, bbox_inches="tight", facecolor=SURFACE)
        print(f"  {p}")
    plt.close(fig)

    for tag, c in (("observed", oc), ("SynTC", sc)):
        sty = c[c.entry >= 100]
        if len(sty):
            print(f"  {tag:<9} entering at >=100 kt: {len(sty)} storms, "
                  f"median loss {sty.loss_pct.median():.0f}%, "
                  f"median exit {sty['exit'].median():.0f} kt, "
                  f"{100*(sty['exit'] < 100).mean():.0f}% leave below 100 kt")


if __name__ == "__main__":
    main()
