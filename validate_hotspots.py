"""
Per-month and per-category spatial validation of a SynTC run against the
historical record, using the method of Zerrudo et al. Table 4.

    python validate_hotspots.py --run ./run01 \
        --ibtracs /path/to/IBTrACS.WP.list.v04r01.points.csv

For each PAGASA intensity class and each calendar month, this builds gridded
track-density maps for the historical and synthetic records inside the PAR
hexagon, computes their Pearson correlation, and compares it against a
bootstrap noise floor derived from the historical record splitting against
itself. The Murphy-style skill score is

    SS = (r_syn - r_floor) / (1 - r_floor)

so SS > 0 means the synthetic field agrees with history better than two random
halves of history agree with each other. That floor matters: at coarse
resolution any two TC datasets correlate highly simply because storms go where
storms go, and a raw r of 0.9 can be worse than chance.

This is the number that says whether the generator fixed the hotspots. Nothing
else in the validation suite answers that question.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

import intensity as I
import terrain
from syntc_ai import CONFIG, in_par

CATEGORIES = (("TD", 22, 34), ("TS", 34, 48), ("STS", 48, 64),
              ("TY", 64, 100), ("STY", 100, 1e9))
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
N_BOOTSTRAP = 100


def grid_edges(deg):
    return (np.arange(115.0, 135.0 + deg, deg), np.arange(5.0, 25.0 + deg, deg))


def density(lat, lon, deg):
    xe, ye = grid_edges(deg)
    h, _, _ = np.histogram2d(lon, lat, bins=[xe, ye])
    # Only cells whose centre lies inside the PAR hexagon count. Correlating
    # over empty ocean outside the domain manufactures agreement.
    cx = 0.5 * (xe[:-1] + xe[1:])
    cy = 0.5 * (ye[:-1] + ye[1:])
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    mask = in_par(gy.ravel(), gx.ravel()).reshape(gx.shape)
    return h, mask


def corr(a, b, mask):
    m = mask & ((a + b) > 0)
    if m.sum() < 5:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def bootstrap_floor(df, deg, rng, n=N_BOOTSTRAP):
    """Median self-correlation of the historical record split in half by storm."""
    sids = df.SID.unique()
    out = []
    for _ in range(n):
        perm = rng.permutation(sids)
        a = df[df.SID.isin(perm[: len(perm) // 2])]
        b = df[df.SID.isin(perm[len(perm) // 2:])]
        da, mask = density(a.lat, a.lon, deg)
        db, _ = density(b.lat, b.lon, deg)
        out.append(corr(da, db, mask))
    return float(np.nanmedian(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="directory of ensemble CSVs")
    ap.add_argument("--ibtracs", required=True)
    ap.add_argument("--dtm", required=True,
                    help="path to the Philippine DTM, same file the generator used")
    ap.add_argument("--grids", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # Must be set before any terrain lookup: terrain.get() resolves this at
    # call time and there is no sensible default on another machine.
    terrain.DTM_PATH = a.dtm

    hist = I.load_intensity_points(a.ibtracs, season_max=2023, impute_td=True)
    hist = hist[in_par(hist.lat.to_numpy(), hist.lon.to_numpy())].copy()
    hist["month"] = hist.time.dt.month
    hist["wind"] = hist.vmax_raw

    files = sorted(glob.glob(os.path.join(a.run, "synthetic_storms_ens*.csv")))
    if not files:
        raise SystemExit(f"no ensemble CSVs in {a.run}")
    syn = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    syn = syn[syn.IN_PAR == 1].rename(
        columns={"LAT": "lat", "LON": "lon", "WIND": "wind", "MONTH": "month"})
    print(f"historical {len(hist):,} points, {hist.SID.nunique()} storms")
    print(f"synthetic  {len(syn):,} points from {len(files)} ensembles\n")

    rng = np.random.default_rng(0)
    rows = []
    for deg in a.grids:
        for label, lo, hi in CATEGORIES + (("All", 0, 1e9),):
            h = hist[(hist.wind >= lo) & (hist.wind < hi)]
            s = syn[(syn.wind >= lo) & (syn.wind < hi)]
            if len(h) < 50 or len(s) < 50:
                continue
            dh, mask = density(h.lat, h.lon, deg)
            ds, _ = density(s.lat, s.lon, deg)
            r = corr(dh, ds, mask)
            fl = bootstrap_floor(h, deg, rng)
            rows.append(dict(grid=deg, kind="category", name=label,
                             hist_n=len(h), syn_n=len(s), r=r, floor=fl,
                             skill=(r - fl) / (1 - fl)))
        for mi, mname in enumerate(MONTHS, start=1):
            h = hist[hist.month == mi]
            s = syn[syn.month == mi]
            if len(h) < 50 or len(s) < 50:
                continue
            dh, mask = density(h.lat, h.lon, deg)
            ds, _ = density(s.lat, s.lon, deg)
            r = corr(dh, ds, mask)
            fl = bootstrap_floor(h, deg, rng)
            rows.append(dict(grid=deg, kind="month", name=mname,
                             hist_n=len(h), syn_n=len(s), r=r, floor=fl,
                             skill=(r - fl) / (1 - fl)))

    res = pd.DataFrame(rows)
    for deg in a.grids:
        for kind in ("category", "month"):
            sub = res[(res.grid == deg) & (res.kind == kind)]
            if sub.empty:
                continue
            print(f"--- {kind} at {deg:g} degree grid ---")
            print(f"{'':>5} {'hist_n':>8} {'syn_n':>9} {'r':>7} {'floor':>7} "
                  f"{'skill':>7}")
            for _, x in sub.iterrows():
                flag = "" if x.skill > 0 else "   <-- below floor"
                print(f"{x['name']:>5} {x.hist_n:8d} {x.syn_n:9d} {x.r:7.3f} "
                      f"{x.floor:7.3f} {x.skill:+7.3f}{flag}")
            print(f"      positive skill: {(sub.skill > 0).sum()} of {len(sub)}\n")

    path = a.out or os.path.join(a.run, "spatial_validation.csv")
    res.to_csv(path, index=False)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
