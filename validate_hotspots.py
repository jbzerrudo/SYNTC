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
    """Pearson correlation over a FIXED set of cells.

    An earlier version restricted the comparison to cells where a + b > 0. That
    is not symmetric between the two uses this function is put to. The synthetic
    field is non-zero in nearly every PAR cell, so the synthetic-versus-observed
    correlation was scored over almost the whole hexagon, including cells with
    no observed storms; the two halves of the record are both sparse, so the
    bootstrap floor was scored over a much smaller and harder set. At 1 degree
    in the super-typhoon class the two masks held 285 and 188 cells. That
    asymmetry inflated the skill score by 0.03 to 0.14, most in the rare
    classes. The mask is now passed in and is the same for both.
    """
    if mask.sum() < 5:
        return np.nan
    if np.ptp(a[mask]) == 0 or np.ptp(b[mask]) == 0:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def bootstrap_floor(df, deg, rng, n=N_BOOTSTRAP, mask=None):
    """Median self-correlation of the historical record split in half by storm.

    `mask` is the cell set the synthetic comparison is scored on, so the floor
    and the score it is subtracted from are measured over the same cells.

    The median is reported with the spread across replicates, because at small
    monthly sample sizes it is not stable: for February (57 track points) the
    100-replicate median moves by 0.06 between random seeds, against 0.008 or
    less for every subset with more than 600 points.
    """
    sids = df.SID.unique()
    out = []
    for _ in range(n):
        perm = rng.permutation(sids)
        a = df[df.SID.isin(perm[: len(perm) // 2])]
        b = df[df.SID.isin(perm[len(perm) // 2:])]
        da, m = density(a.lat, a.lon, deg)
        db, _ = density(b.lat, b.lon, deg)
        out.append(corr(da, db, mask if mask is not None else m))
    out = np.array(out, dtype=float)
    return float(np.nanmedian(out)), float(np.nanstd(out))


N_SKILL_BOOT = 300


def skill_interval(h, ds, deg, mask, rng, ss_point, n=N_SKILL_BOOT):
    """Basic (pivotal) bootstrap interval on the skill score.

    Resampling is by storm, not by track point, because points within a storm
    are not independent. Both the numerator and the floor are recomputed inside
    each replicate, since both carry the sampling error of the same 47-season
    record. The interval is pivotal rather than percentile: resampling with
    replacement thins the effective sample and pushes the replicate correlation
    below the point estimate, so a percentile interval is biased low. The two
    constructions agree for June to December and disagree in sign for January,
    February and March, which is the signal that the bootstrap is not usable at
    those sample sizes; the replicate spread is reported so that shows.
    """
    sids = h.SID.unique()
    boot = []
    for _ in range(n):
        pick = rng.choice(sids, len(sids), replace=True)
        hb = h[h.SID.isin(pick)]
        dhb, _ = density(hb.lat, hb.lon, deg)
        r = corr(dhb, ds, mask)
        a_ = h[h.SID.isin(pick[: len(pick) // 2])]
        b_ = h[h.SID.isin(pick[len(pick) // 2:])]
        da, _ = density(a_.lat, a_.lon, deg)
        db, _ = density(b_.lat, b_.lon, deg)
        f = corr(da, db, mask)
        if np.isfinite(r) and np.isfinite(f) and f < 0.995:
            boot.append((r - f) / (1 - f))
    if len(boot) < 30:
        return np.nan, np.nan
    q = np.percentile(boot, [2.5, 97.5])
    return float(2 * ss_point - q[1]), float(2 * ss_point - q[0])


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
        # The aggregate synthetic field, used as a null reference. It carries
        # the model's overall PAR climatology and nothing about intensity class
        # or calendar month, so a per-class or per-month score that does not
        # beat it is not evidence of class- or month-specific fidelity. The
        # bootstrap floor alone cannot make that distinction: the null clears it
        # in every class and in ten to eleven of twelve months.
        d_null, _ = density(syn.lat, syn.lon, deg)

        def score(h, s, label, kind):
            dh, mask = density(h.lat, h.lon, deg)
            ds, _ = density(s.lat, s.lon, deg)
            r = corr(dh, ds, mask)
            fl, fl_sd = bootstrap_floor(h, deg, rng, mask=mask)
            r_null = corr(dh, d_null, mask)
            lo_, hi_ = skill_interval(h, ds, deg, mask, rng, (r - fl) / (1 - fl))
            return dict(grid=deg, kind=kind, name=label,
                        hist_n=len(h), syn_n=len(s), cells=int(mask.sum()),
                        r=r, floor=fl, floor_sd=fl_sd,
                        skill=(r - fl) / (1 - fl),
                        r_null=r_null, skill_null=(r_null - fl) / (1 - fl),
                        skill_lo=lo_, skill_hi=hi_)

        for label, lo, hi in CATEGORIES + (("All", 0, 1e9),):
            h = hist[(hist.wind >= lo) & (hist.wind < hi)]
            s = syn[(syn.wind >= lo) & (syn.wind < hi)]
            if len(h) < 50 or len(s) < 50:
                continue
            rows.append(score(h, s, label, "category"))
        for mi, mname in enumerate(MONTHS, start=1):
            h = hist[hist.month == mi]
            s = syn[syn.month == mi]
            if len(h) < 50 or len(s) < 50:
                continue
            rows.append(score(h, s, mname, "month"))

    res = pd.DataFrame(rows)
    for deg in a.grids:
        for kind in ("category", "month"):
            sub = res[(res.grid == deg) & (res.kind == kind)]
            if sub.empty:
                continue
            print(f"--- {kind} at {deg:g} degree grid ---")
            print(f"{'':>5} {'hist_n':>8} {'syn_n':>9} {'r':>7} {'floor':>7} "
                  f"{'skill':>7} {'null':>7} {'95% CI':>15}")
            for _, x in sub.iterrows():
                flag = "" if x.skill > 0 else "   <-- below floor"
                if np.isfinite(x.skill_null) and x.skill <= x.skill_null + 0.02:
                    flag += "   <-- no better than the class/month-blind null"
                print(f"{x['name']:>5} {x.hist_n:8d} {x.syn_n:9d} {x.r:7.3f} "
                      f"{x.floor:7.3f} {x.skill:+7.3f} {x.skill_null:+7.3f} "
                      f"[{x.skill_lo:+.2f},{x.skill_hi:+.2f}]{flag}")
            print(f"      positive skill: {(sub.skill > 0).sum()} of {len(sub)}; "
                  f"beating the null: {(sub.skill > sub.skill_null + 0.02).sum()} "
                  f"of {len(sub)}\n")

    path = a.out or os.path.join(a.run, "spatial_validation.csv")
    res.to_csv(path, index=False)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
