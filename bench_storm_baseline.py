"""Held-out likelihood: the fitted MDN against the STORM-style binned baseline.

Both are probability densities over the same target, the 6-hourly (dlon, dlat)
increment, so their held-out log likelihoods are directly comparable. This is
the comparison models.py was written for and that the manuscript reports.

Run from the repo folder:

    cd /d D:\2026\SYNTC\SYNTC
    python bench_storm_baseline.py

Paths default to the run07 layout and can be overridden:

    python bench_storm_baseline.py --run D:\2026\SYNTC\SYNTC-AI\run08
"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=r"D:\2026\SYNTC\SYNTC-AI\run07",
                help="run folder holding config.json and model.pkl")
ap.add_argument("--ibtracs", default=None,
                help="override the IBTrACS path stored in config.json")
ap.add_argument("--dtm", default="dtm_phil_1km.tif")
A = ap.parse_args()

cfg_path = os.path.join(A.run, "config.json")
model_path = os.path.join(A.run, "model.pkl")
for f in (cfg_path, model_path):
    if not os.path.exists(f):
        sys.exit(f"not found: {f}\nPass --run pointing at the run folder.")

CFG = json.load(open(cfg_path))
IB = A.ibtracs or CFG["ibtracs"]
if not os.path.exists(IB):
    sys.exit(f"IBTrACS not found at {IB}\nPass --ibtracs with the correct path.")

import terrain
terrain.DTM_PATH = A.dtm
import data as D
from models import StormBaseline
from syntc_ai import load_model

pts = D.load_tracks(IB, CFG["season_min"], CFG["season_max"], synoptic_only=True)
tr = D.build_transitions(pts, step_hours=CFG["step_hours"])
fit = tr[tr.SEASON <= CFG["valid_max_season"]]
test = tr[tr.SEASON > CFG["train_max_season"]]
print(f"fit  {len(fit):,} transitions, {fit.SEASON.min()}-{fit.SEASON.max()}")
print(f"test {len(test):,} transitions, {test.SEASON.min()}-{test.SEASON.max()}\n")

ll_mdn = load_model(model_path).track.log_prob(test)
ll_base = StormBaseline().fit(fit).log_prob(test)
d = ll_mdn - ll_base

rng = np.random.default_rng(0)
boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
print(f"held-out mean log likelihood, {len(test):,} transitions "
      f"{test.SEASON.min()}-{test.SEASON.max()}")
print(f"  MDN            {ll_mdn.mean():+8.4f}")
print(f"  STORM binned   {ll_base.mean():+8.4f}")
print(f"  difference     {d.mean():+8.4f}   95% CI "
      f"{np.percentile(boot,2.5):+.4f} to {np.percentile(boot,97.5):+.4f}")
print(f"  MDN better on  {100*(d>0).mean():.1f}% of individual transitions")
