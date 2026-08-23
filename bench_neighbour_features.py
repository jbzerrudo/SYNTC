"""Does telling the network about concurrent storms improve the track density?

Adds three features and refits, changing nothing else. Both models get the same
seed, the same epochs, the same split, and are scored on the same held-out
2015-2023 transitions by log likelihood, so the comparison is the difference
those three numbers make and nothing else.

    nbr_x, nbr_y   unit vector toward the nearest concurrent TC, scaled by
                   exp(-d/800 km), so a distant storm fades to zero and the
                   absence of a neighbour is an honest zero rather than a
                   fabricated distance
    nbr_v          that storm's intensity, scaled the same way
"""
import json, numpy as np, pandas as pd
import data as D, terrain
from models import MDNPropagator

CFG = json.load(open("config.json"))
terrain.DTM_PATH = "dtm_phil_1km.tif"
NEW = ["nbr_x", "nbr_y", "nbr_v"]


def add_neighbour(tr, pts, scale_km=800.0):
    p = pts[["SID", "time", "lat", "lon", "vmax"]].dropna(subset=["lat", "lon"])
    cols = {}
    for t, g in p.groupby("time", sort=False):
        la, lo = g.lat.to_numpy(), g.lon.to_numpy()
        vm = pd.to_numeric(g.vmax, errors="coerce").fillna(0.0).to_numpy()
        n = len(g)
        if n < 2:
            for s in g.SID:
                cols[(s, t)] = (0.0, 0.0, 0.0)
            continue
        m = np.radians(la[:, None])
        dx = (lo[None, :] - lo[:, None]) * 111.32 * np.cos(m)
        dy = (la[None, :] - la[:, None]) * 111.32
        dist = np.hypot(dx, dy)
        np.fill_diagonal(dist, np.inf)
        j = dist.argmin(1)
        d = dist[np.arange(n), j]
        w = np.exp(-d / scale_km)
        norm = np.maximum(np.hypot(dx[np.arange(n), j], dy[np.arange(n), j]), 1e-9)
        ux = dx[np.arange(n), j] / norm
        uy = dy[np.arange(n), j] / norm
        for k, s in enumerate(g.SID):
            cols[(s, t)] = (w[k] * ux[k], w[k] * uy[k], w[k] * vm[j[k]] / 100.0)
    arr = np.array([cols.get((s, t), (0.0, 0.0, 0.0))
                    for s, t in zip(tr.SID, tr.time)], dtype=float)
    for i, c in enumerate(NEW):
        tr[c] = arr[:, i]
    return tr


pts = D.load_tracks(CFG["ibtracs"], CFG["season_min"], CFG["season_max"],
                    synoptic_only=True)
tr = D.build_transitions(pts, step_hours=CFG["step_hours"])
tr = add_neighbour(tr, pts)

fit = tr[tr.SEASON <= CFG["valid_max_season"]]
val = tr[(tr.SEASON > CFG["valid_max_season"]) & (tr.SEASON <= CFG["train_max_season"])]
test = tr[tr.SEASON > CFG["train_max_season"]]
nz = (test[NEW].abs().sum(1) > 0.01).mean()
print(f"fit {len(fit):,}  valid {len(val):,}  test {len(test):,}")
print(f"test transitions with a non-negligible neighbour: {100*nz:.1f}%\n")

res = {}
for tag, feats in (("baseline  9 features", list(D.FEATURES)),
                   ("neighbour 12 features", list(D.FEATURES) + NEW)):
    D.FEATURES[:] = feats                       # models.py holds this same list
    m = MDNPropagator(CFG["track_components"], CFG["track_hidden"],
                      seed=CFG["seed"]).fit(fit, val, epochs=CFG["epochs"],
                                            verbose=False)
    ll = m.log_prob(test)
    res[tag] = ll
    print(f"{tag:<22} held-out mean log likelihood {ll.mean():+8.4f}")

a, b = res["baseline  9 features"], res["neighbour 12 features"]
d = b - a
rng = np.random.default_rng(0)
boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
print(f"\ndifference {d.mean():+8.4f}   95% CI {np.percentile(boot,2.5):+.4f} "
      f"to {np.percentile(boot,97.5):+.4f}")
sub = test[NEW].abs().sum(1).to_numpy() > 0.01
print(f"on the {sub.sum():,} transitions that actually have a neighbour: "
      f"{d[sub].mean():+.4f}")
