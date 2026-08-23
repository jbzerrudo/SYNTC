"""Does the track propagator benefit from longer memory?

The network is first-order Markov: it sees only the previous displacement. The
segment-resampling literature argues that this is what makes step-wise track
models diffusive. Before building a recurrent body, test the cheap version:
hand the feedforward network the displacement two and three steps back and see
whether the held-out likelihood moves.

Identical split, seed and epochs throughout. The only thing that changes is how
many past steps the network can see.
"""
import json, numpy as np, pandas as pd
import data as D, terrain
from models import MDNPropagator

CFG = json.load(open("config.json"))
terrain.DTM_PATH = "dtm_phil_1km.tif"

pts = D.load_tracks(CFG["ibtracs"], CFG["season_min"], CFG["season_max"],
                    synoptic_only=True)
tr = D.build_transitions(pts, step_hours=CFG["step_hours"])

# Displacement 2 and 3 steps back, zero where the storm is too young for it.
g = tr.groupby("SID", sort=False)
for k in (2, 3):
    tr[f"u_prev{k}"] = g["u_prev"].shift(k - 1).fillna(0.0)
    tr[f"v_prev{k}"] = g["v_prev"].shift(k - 1).fillna(0.0)
    tr[f"has_prev{k}"] = g["u_prev"].shift(k - 1).notna().astype(float)

fit = tr[tr.SEASON <= CFG["valid_max_season"]]
val = tr[(tr.SEASON > CFG["valid_max_season"]) & (tr.SEASON <= CFG["train_max_season"])]
test = tr[tr.SEASON > CFG["train_max_season"]]
print(f"fit {len(fit):,}  valid {len(val):,}  test {len(test):,}\n")

BASE = list(D.FEATURES)
runs = {
    "order 1  (as published)": BASE,
    "order 2  (+1 past step)": BASE + ["u_prev2", "v_prev2", "has_prev2"],
    "order 3  (+2 past steps)": BASE + ["u_prev2", "v_prev2", "has_prev2",
                                        "u_prev3", "v_prev3", "has_prev3"],
}
res = {}
for tag, feats in runs.items():
    D.FEATURES[:] = feats
    m = MDNPropagator(CFG["track_components"], CFG["track_hidden"],
                      seed=CFG["seed"]).fit(fit, val, epochs=CFG["epochs"],
                                            verbose=False)
    res[tag] = m.log_prob(test)
    print(f"{tag:<26} {len(feats):>2} features   held-out ll {res[tag].mean():+8.4f}")

a = res["order 1  (as published)"]
rng = np.random.default_rng(0)
print()
for tag in ("order 2  (+1 past step)", "order 3  (+2 past steps)"):
    d = res[tag] - a
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    print(f"{tag:<26} vs order 1: {d.mean():+7.4f}   95% CI "
          f"{np.percentile(boot,2.5):+.4f} to {np.percentile(boot,97.5):+.4f}")
