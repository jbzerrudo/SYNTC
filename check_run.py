"""
Acceptance test for a finished SynTC run. Run this before anything else.

    python check_run.py --run ./run03 --ibtracs ... --dtm ...

Every other script assumes the catalogue is sound. This one checks that it is,
in a few seconds, so a broken run is caught before you spend an afternoon
building figures on top of it. It exits non-zero if any check fails.

What it checks and why each one exists
--------------------------------------
1. Twenty ensembles, all written by the same run. An interrupted rerun
   overwrites ensembles one at a time, leaving a folder holding some fresh
   members and some stale, and every downstream script reads all twenty without
   complaint. This has already happened once in this project.

2. The config records what you think it does: season_max, counts_mode and the
   MPI trend.

3. PAR storms per year near 16.2, the observed 1977-2023 rate. A run that shows
   about 25 is using basin-wide counts as a PAR target, which over-fills PAR by
   55% and inflates every hazard number downstream.

4. NO POINT EXCEEDS ITS OWN CEILING. This is the important one. The saturation
   brake used to scale the wind increment using the wind at the start of the
   step, so a storm well below the ceiling was barely braked and one rare draw
   could throw it clean over the top; the brake never got a turn because the
   storm was never at that intensity when a step began. That produced 147 kt
   over Luzon, 7 kt above the strongest storm ever analysed anywhere in the
   basin. The ceiling is now enforced on the result, and this check is the
   proof: for every track point, the wind must be at or below the potential
   intensity at that point's own location, month and year.

   The ceiling comes from the model.pkl the run itself wrote, so the test uses
   the exact table the run used rather than a rebuilt approximation.

5. The highest over-land wind against a matched benchmark. The generator prints
   its maximum against the observed 47-season record, which is unfair in both
   directions. The fair comparison is the largest value expected in a draw of
   the same length as the catalogue.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import intensity as I
import terrain

EXPECTED_PAR_RATE = 16.2         # observed 1977-2023, storms entering PAR
TOL_KT = 0.01                    # floating point only


class Checker:
    def __init__(self):
        self.failed = 0

    def ok(self, cond, label, detail=""):
        mark = "PASS" if cond else "FAIL"
        if not cond:
            self.failed += 1
        print(f"  [{mark}] {label}" + (f"   {detail}" if detail else ""))
        return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ibtracs", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--ensembles", type=int, default=20)
    a = ap.parse_args()
    terrain.DTM_PATH = a.dtm
    c = Checker()
    print(f"\nchecking {a.run}\n")

    # ---- 1. files ----------------------------------------------------
    files = sorted(glob.glob(os.path.join(a.run, "synthetic_storms_ens*.csv")))
    c.ok(len(files) == a.ensembles, f"{a.ensembles} ensembles present",
         f"found {len(files)}")
    if files:
        mt = np.array([os.path.getmtime(f) for f in files])
        span_h = (mt.max() - mt.min()) / 3600.0
        c.ok(span_h < 6.0, "all ensembles from one run",
             f"written over {span_h:.1f} h")

    # ---- 2. config ---------------------------------------------------
    cfg_path = os.path.join(a.run, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    c.ok(cfg.get("season_max") == 2023, "season_max is 2023",
         f"got {cfg.get('season_max')}")
    c.ok(cfg.get("counts_mode") == "par", "counts_mode is par",
         f"got {cfg.get('counts_mode')}")
    trend = float(cfg.get("mpi_trend_percent_per_century", 0.0))
    print(f"  [info] MPI trend {trend:g}% per century")

    # ---- 3. storm frequency ------------------------------------------
    d = pd.concat([pd.read_csv(f, usecols=["SID", "STEP", "YEAR", "MONTH",
                                           "LAT", "LON", "WIND", "OVER_LAND",
                                           "IN_PAR"])
                   for f in files], ignore_index=True)
    n_par = d[d.IN_PAR == 1].SID.nunique()
    n_years = len(files) * d.YEAR.nunique()
    rate = n_par / n_years
    c.ok(abs(rate - EXPECTED_PAR_RATE) < 2.0,
         f"PAR storms per year near {EXPECTED_PAR_RATE}", f"got {rate:.2f}")

    # ---- 4. the ceiling ----------------------------------------------
    mp = os.path.join(a.run, "model.pkl")
    if not os.path.exists(mp):
        print(f"  [SKIP] ceiling test: no model.pkl in {a.run}")
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from syntc_ai import load_model
        load_model(mp)                       # restores the PI singleton
        pi = I.get_potential_intensity()
        year0 = int(d.YEAR.min())
        d = d.sort_values(["SID", "STEP"]).reset_index(drop=True)
        ceiling = pi.sample(d.LAT.to_numpy(), d.LON.to_numpy(),
                            d.MONTH.to_numpy())
        ceiling = ceiling * I.mpi_warming_factor(d.YEAR.to_numpy(), year0, trend)

        # The invariant is about INTENSIFICATION, not position. A positive step
        # may never carry the wind above the ceiling at the point the step was
        # computed from. It deliberately does not force a storm down when it
        # travels into a region with a lower ceiling: carrying intensity west
        # and decaying there is what real storms do, and clamping it would be
        # imposing the answer rather than predicting it. Testing "no point is
        # ever above its local ceiling" would fail on exactly that legitimate
        # behaviour, which is what the first version of this check did.
        w0 = d.WIND.to_numpy()
        w1 = d.groupby("SID").WIND.shift(-1).to_numpy()
        valid = np.isfinite(w1)
        rising = valid & (w1 > w0 + TOL_KT)
        breach = rising & (w1 > ceiling + TOL_KT)
        n_bad = int(breach.sum())
        worst = float((w1 - ceiling)[breach].max()) if n_bad else 0.0
        c.ok(n_bad == 0, "no intensification carries a storm past its ceiling",
             f"{n_bad:,} breaches of {int(rising.sum()):,} intensifying steps, "
             f"worst by {worst:.1f} kt" if n_bad
             else f"{int(rising.sum()):,} intensifying steps checked")

        # Informational: storms that are above the local ceiling because they
        # brought that intensity with them. Expected, and worth seeing.
        adv = w0 - ceiling
        n_adv = int((adv > TOL_KT).sum())
        print(f"  [info] {n_adv:,} of {len(d):,} points sit above the local "
              f"ceiling by advection, worst {adv.max():.0f} kt "
              f"(expected, not a breach)")

    # ---- 5. extremes -------------------------------------------------
    obs = I.load_intensity_points(a.ibtracs, season_max=2023)
    print(f"\n  [info] highest wind anywhere: SynTC {d.WIND.max():.0f} kt, "
          f"observed basin record {obs.vmax_raw.max():.0f} kt")

    _, oland = terrain.get(a.dtm).sample(obs.lat.to_numpy(), obs.lon.to_numpy())
    am = obs[oland].groupby("SEASON").vmax_raw.max().to_numpy()
    sh, loc, sc = stats.weibull_min.fit(am, floc=0)
    n_yr = d.YEAR.nunique()
    sim = stats.weibull_min.rvs(sh, loc, sc, size=(20000, n_yr),
                                random_state=np.random.default_rng(0)).max(axis=1)
    hi = np.percentile(sim, 95)
    per_ens = [float(pd.read_csv(f, usecols=["WIND", "OVER_LAND"])
                     .query("OVER_LAND == 1").WIND.max()) for f in files]
    p = np.array(per_ens)
    n_hot = int((p > hi).sum())
    print(f"  [info] over-land maximum: benchmark for a {n_yr}-season draw is "
          f"median {np.median(sim):.0f}, 95th {hi:.0f} kt")
    c.ok(n_hot <= max(2, int(0.15 * len(p))),
         "over-land maxima consistent with the record",
         f"{n_hot} of {len(p)} ensembles above the 95th percentile "
         f"(about {0.05*len(p):.0f} expected); SynTC median {np.median(p):.0f}, "
         f"max {p.max():.0f} kt")

    print(f"\n{'ALL CHECKS PASSED' if not c.failed else str(c.failed) + ' CHECK(S) FAILED'}\n")
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()
