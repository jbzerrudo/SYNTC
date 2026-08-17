"""
Extreme value analysis of PAR annual maximum wind, for the observed record and
for a SynTC run, using one estimator so the two are comparable.

    python return_levels.py --ibtracs /path/to/IBTrACS.WP.list.v04r01.points.csv \
        --dtm /path/to/dtm_phil_1km.tif --run ./run03

Method, matching the manuscript's Section 2.3
---------------------------------------------
Annual maxima are taken by block maxima: the highest 10-minute sustained wind
recorded inside the PAR hexagon in each year. Five candidate distributions are
fitted by maximum likelihood (GEV, Gumbel, Weibull, exponential, generalised
Pareto), return levels are evaluated at F^-1(1 - 1/T), and 95% confidence
intervals come from a nonparametric bootstrap over annual maxima with 1,000
resamples, reported as the half-width of the 2.5-97.5 percentile interval.

Why this script exists
----------------------
Comparing a fitted return level against a raw quantile of simulated maxima is
not a like-for-like comparison. The observed estimate in the manuscript is a
Weibull fit to 47 annual maxima; a synthetic estimate has to be the same fit to
the synthetic annual maxima, or the difference between them is partly just the
difference between two estimators.

For a synthetic catalogue the annual maxima are pooled across ensembles, so 20
ensembles of 100 years give 2,000 annual maxima rather than 47. The fitted
return level is therefore far better constrained than the observed one, and its
confidence interval will be correspondingly narrow. That is a real advantage of
a synthetic catalogue and worth stating rather than hiding.

Stationarity
------------
An extreme value fit assumes the sample is drawn from one distribution. That
holds for the observed record and for a stationary synthetic run. It does NOT
hold for a run generated with a non-zero MPI trend, where the later years come
from a shifted distribution. For such a run the fitted return level should be
read as the century-averaged value, and the script reports first-half and
second-half fits separately so the drift is visible rather than buried.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

import intensity as I
import terrain
from syntc_ai import in_par

RETURN_PERIODS = (1.5, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200)
N_BOOTSTRAP = 1000

FITS = {
    "GEV":         (stats.genextreme, {}),
    "Gumbel":      (stats.gumbel_r, {}),
    "Weibull":     (stats.weibull_min, {"floc": 0}),
    "Exponential": (stats.expon, {}),
    "Pareto":      (stats.genpareto, {}),
}


PROBS = np.array([1.0 - 1.0 / T for T in RETURN_PERIODS])


def return_levels_once(sample, dist_name):
    """Fit once, evaluate every return period from that fit.

    Refitting per return period would repeat the same maximum likelihood
    solve twelve times over for no gain, which turns a 1,000-resample
    bootstrap into 60,000 fits per distribution instead of 1,000.
    """
    dist, kw = FITS[dist_name]
    try:
        params = dist.fit(sample, **kw)
        return np.asarray(dist.ppf(PROBS, *params), dtype=float)
    except Exception:
        return np.full(len(PROBS), np.nan)


def fit_table(sample, label, rng, n_boot=N_BOOTSTRAP):
    rows = [{"source": label, "n_maxima": len(sample), "return_period": T}
            for T in RETURN_PERIODS]
    for name in FITS:
        point = return_levels_once(sample, name)
        boot = np.full((n_boot, len(PROBS)), np.nan)
        for b in range(n_boot):
            boot[b] = return_levels_once(
                rng.choice(sample, len(sample), replace=True), name)
        with np.errstate(all="ignore"):
            lo = np.nanpercentile(boot, 2.5, axis=0)
            hi = np.nanpercentile(boot, 97.5, axis=0)
        for i, row in enumerate(rows):
            row[name] = point[i]
            row[f"{name}_ME"] = (hi[i] - lo[i]) / 2.0
            row[f"{name}_lo"] = lo[i]
            row[f"{name}_hi"] = hi[i]
    return pd.DataFrame(rows)


def observed_maxima(ibtracs, season_max=2023):
    pts = I.load_intensity_points(ibtracs, season_max=season_max)
    pts = pts[in_par(pts.lat.to_numpy(), pts.lon.to_numpy())]
    return pts.groupby("SEASON").vmax_raw.max().to_numpy()


def synthetic_maxima(run, year_range=None):
    """One maximum per ensemble-year, pooled across ensembles."""
    files = sorted(glob.glob(os.path.join(run, "synthetic_storms_ens*.csv")))
    if not files:
        raise SystemExit(f"no ensemble CSVs in {run}")
    out = []
    for i, f in enumerate(files, start=1):
        d = pd.read_csv(f, usecols=["YEAR", "WIND", "IN_PAR"])
        d = d[d.IN_PAR == 1]
        if year_range:
            d = d[d.YEAR.between(*year_range)]
        if len(d):
            out.append(d.groupby("YEAR").WIND.max().to_numpy())
    return np.concatenate(out) if out else np.array([])


def show(df, label):
    print(f"\n--- {label}  (n = {int(df.n_maxima.iloc[0])} annual maxima) ---")
    print(f"{'T (yr)':>7} " + " ".join(f"{n:>9}" for n in FITS))
    for _, r in df.iterrows():
        cells = " ".join(f"{r[n]:6.1f}±{r[n+'_ME']:>3.0f}" if np.isfinite(r[n])
                         else "     n/a " for n in FITS)
        print(f"{r.return_period:7.1f} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibtracs", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--run", default=None,
                    help="a SynTC run folder; omit for the observed record only")
    ap.add_argument("--split-halves", action="store_true",
                    help="also fit the first and second halves of the "
                         "simulation separately, to expose any warming drift")
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    terrain.DTM_PATH = a.dtm
    rng = np.random.default_rng(0)

    frames = []
    obs = observed_maxima(a.ibtracs)
    f = fit_table(obs, "observed 1977-2023", rng, a.bootstrap)
    show(f, "observed 1977-2023")
    frames.append(f)

    if a.run:
        syn = synthetic_maxima(a.run)
        label = f"synthetic ({os.path.basename(a.run.rstrip(os.sep))})"
        f = fit_table(syn, label, rng, a.bootstrap)
        show(f, label)
        frames.append(f)

        if a.split_halves:
            d = pd.read_csv(sorted(glob.glob(os.path.join(
                a.run, "synthetic_storms_ens*.csv")))[0], usecols=["YEAR"])
            y0, y1 = int(d.YEAR.min()), int(d.YEAR.max())
            mid = (y0 + y1) // 2
            for lo, hi, tag in ((y0, mid, "first half"), (mid + 1, y1, "second half")):
                s = synthetic_maxima(a.run, (lo, hi))
                f = fit_table(s, f"{label} {tag} {lo}-{hi}", rng, a.bootstrap)
                show(f, f"{label} {tag} {lo}-{hi}")
                frames.append(f)

    res = pd.concat(frames, ignore_index=True)
    path = a.out or os.path.join(a.run or ".", "return_levels.csv")
    res.to_csv(path, index=False)
    print(f"\nwritten: {path}")
    print("\nThe 100-year row is the one the manuscript quotes. Compare the "
          "observed and synthetic Weibull columns there.")


if __name__ == "__main__":
    main()
