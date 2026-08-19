"""Table 3 as LaTeX, straight from spatial_validation.csv.

    python make_table_spatial.py --run run03 > tab_spatial.tex

Panel (a) is by PAGASA intensity class at both grids; panel (b) is by calendar
month at the 2 degree grid. The null and interval columns are printed only if
validate_hotspots.py wrote them, so an older CSV still produces the short form.
"""

import argparse
import os

import numpy as np
import pandas as pd

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
CLASSES = ("TD", "TS", "STS", "TY", "STY", "All")


def g(d, grid, kind, name, col):
    r = d[(d.grid == grid) & (d.kind == kind) & (d.name == name)]
    return float(r[col].iloc[0]) if len(r) and col in r else np.nan


def num(x, fmt="{:.3f}", plus=False):
    if not np.isfinite(x):
        return "--"
    s = fmt.format(x)
    return ("$+$" + s if x >= 0 else "$-$" + s.lstrip("-")) if plus else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    path = a.csv or os.path.join(a.run, "spatial_validation.csv")
    d = pd.read_csv(path)
    has_null = "skill_null" in d.columns
    has_ci = "skill_lo" in d.columns

    hist_all = int(g(d, 1.0, "category", "All", "hist_n"))
    syn_all = int(g(d, 1.0, "category", "All", "syn_n"))

    print(r"\begin{table}[htbp]")
    print(r"  \caption{Spatial pattern validation of SynTC against IBTrACS-JMA "
          r"(1977--2023) within the PAR hexagon. Pearson $r$ compares historical "
          r"and synthetic gridded point-density maps over every cell whose centre "
          r"lies inside the hexagon; floor is the median bootstrap self-correlation "
          r"across 100 random storm-ID half-splits, measured over the same cells; "
          r"the skill score is SS $= (r - \mathrm{floor})/(1 - \mathrm{floor})$. "
          + (r"$S_{0}$ is the same score for the aggregate synthetic field, which "
             r"carries no information about intensity class or calendar month, so "
             r"SS $> S_{0}$ rather than SS $> 0$ is the test of class- and "
             r"month-specific fidelity. " if has_null else "")
          + (r"Brackets are a 95\% pivotal interval from a 300-replicate "
             r"storm-clustered bootstrap. " if has_ci else "")
          + f"Counts are PAR-clipped track points: {hist_all:,} historical "
            f"against {syn_all:,} synthetic. "
          + r"Generated directly from \texttt{spatial\_validation.csv}.}")
    print(r"  \label{tab:spatial_corr}")
    print(r"  \centering")
    print(r"  \footnotesize")
    print(r"  \setlength{\tabcolsep}{4.5pt}")
    print()
    print(r"  \textbf{(a) By PAGASA intensity class.}\\[2pt]")
    ncol = 3 + (3 if not has_null else 4) * 2
    print(r"  \begin{tabular}{lrr" + "r" * ((ncol - 3)) + "}")
    print(r"  \toprule")
    span = 4 if has_null else 3
    print(r"  \textbf{Class} & \textbf{hist $n$} & \textbf{syn $n$} & "
          r"\multicolumn{%d}{c}{\textbf{1\textdegree{} grid}} & "
          r"\multicolumn{%d}{c}{\textbf{2\textdegree{} grid}} \\" % (span, span))
    print(r"  \cmidrule(lr){4-%d} \cmidrule(lr){%d-%d}"
          % (3 + span, 4 + span, 3 + 2 * span))
    head = " & $r$ & floor & SS" + (" & $S_0$" if has_null else "")
    print(r"  & &" + head + " &" + head + r" \\")
    print(r"  \midrule")
    for c in CLASSES:
        if c == "All":
            print(r"  \midrule")
        cells = []
        for grid in (1.0, 2.0):
            cells += [num(g(d, grid, "category", c, "r")),
                      num(g(d, grid, "category", c, "floor")),
                      num(g(d, grid, "category", c, "skill"), "{:.2f}", True)]
            if has_null:
                cells.append(num(g(d, grid, "category", c, "skill_null"),
                                 "{:.2f}", True))
        name = r"\textbf{All}" if c == "All" else c
        n_h = f"{int(g(d, 1.0, 'category', c, 'hist_n')):,}"
        n_s = f"{int(g(d, 1.0, 'category', c, 'syn_n')):,}"
        print(f"  {name} & {n_h} & {n_s} & " + " & ".join(cells) + r" \\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print()
    print(r"  \vspace{6pt}")
    print()
    print(r"  \textbf{(b) By calendar month, at 2\textdegree{} grid.}\\[2pt]")
    print(r"  \setlength{\tabcolsep}{3.4pt}")
    print(r"  \begin{tabular}{l" + "r" * 12 + "}")
    print(r"  \toprule")
    print(r"   & " + " & ".join(r"\textbf{%s}" % m for m in MONTHS) + r" \\")
    print(r"  \midrule")
    rows = [("$r$", "r", "{:.2f}", False), ("floor", "floor", "{:.2f}", False),
            ("SS", "skill", "{:.2f}", True)]
    if has_null:
        rows.append(("$S_0$", "skill_null", "{:.2f}", True))
    for lab, col, fmt, plus in rows:
        print(f"  {lab} & " + " & ".join(
            num(g(d, 2.0, "month", m, col), fmt, plus) for m in MONTHS) + r" \\")
    print(r"  \midrule")
    print(r"  $n$ & " + " & ".join(
        f"{int(g(d, 2.0, 'month', m, 'hist_n'))}" for m in MONTHS) + r" \\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
