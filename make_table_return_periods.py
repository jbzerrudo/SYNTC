"""
Emit the return-period table as LaTeX, straight from return_levels.csv.

    python make_table_return_periods.py --run ./run03 > tab_return_periods.tex

The table in the manuscript was typed by hand from an earlier run and had
drifted from the CSV in three ways worth recording, because each is a different
kind of error and only one of them was a typo.

1. The Weibull 100-year level read 126.0 against 126.4 in the CSV. A stale
   number: small, but it is the headline result of the section.

2. The Weibull margin-of-error column read 63.7, 251.1, 399.0 and 1016.4 kt at
   the 20, 50, 75 and 200-year levels, against 2.8 to 3.5 kt now. Those came
   from bootstrap replicates in which the two-parameter Weibull fit failed to
   converge and returned a shape parameter near zero, which sends the upper
   tail to infinity. A margin of error of 1016 kt on a 127 kt estimate is not a
   wide interval, it is a broken fit, and printing it invites a reviewer to ask
   what else in the table was not checked.

3. The Pareto column ran to 450.6 kt at 200 years. The generalised Pareto ML fit
   on 47 annual maxima is degenerate here: it pins to the sample maximum and its
   return levels flatten at 125.0 kt rather than growing. That is the honest
   output and it is what the CSV now holds. The old column was extrapolating
   from a fit that had already failed.

The point of generating the table rather than typing it is that none of these
can recur silently.
"""

import argparse
import os

import pandas as pd

DISTS = ("GEV", "Gumbel", "Weibull", "Exponential", "Pareto")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--source", default="observed",
                    help="row prefix in return_levels.csv; 'observed' is the "
                         "record and is what the paper's table reports")
    a = ap.parse_args()

    csv = os.path.join(a.run, "return_levels.csv")
    d = pd.read_csv(csv)
    d = d[d.source.str.startswith(a.source)]
    if not len(d):
        raise SystemExit(f"no rows starting with {a.source!r} in {csv}")
    n = int(d.n_maxima.iloc[0])
    src = d.source.iloc[0]

    print(r"\begin{table}[t]")
    print(r"\caption{Return levels of the annual maximum 10-minute sustained "
          r"wind (knots) in the Philippine Area of Responsibility, from five "
          r"probability distributions fitted to the " f"{n}" r"-season record ("
          f"{src.split(None, 1)[-1]}" r"), with the half-width of the "
          r"bootstrap 95\% confidence interval (ME). The two-parameter Weibull "
          r"is the fit adopted; the generalised Pareto fit is degenerate on a "
          f"{n}" r"-maximum sample and pins to the sample maximum, and the "
          r"exponential fit diverges. Both are reported rather than omitted.}")
    print(r"\label{tab:return_periods}")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\begin{tabular}{r" + "rr" * len(DISTS) + "}")
    print(r"\toprule")
    print(r"\textbf{Return} & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\textbf{{{x}}}}}" for x in DISTS) + r" \\")
    print(r"\textbf{period (yr)} & " + " & ".join(["", r"\textbf{ME}"] * len(DISTS)
                                                  ).replace("&  &", "& &") + r" \\")
    print(r"\midrule")
    for _, r in d.sort_values("return_period").iterrows():
        cells = []
        for x in DISTS:
            cells += [f"{r[x]:.1f}", f"{r[x + '_ME']:.1f}"]
        print(f"{r.return_period:g} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
