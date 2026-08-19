"""Averaging-period conversion between the two best-track wind conventions.

Two components of this model were calibrated in different conventions and have
to be reconciled explicitly rather than assumed interchangeable.

  * The terrain-decay coefficient a of Zerrudo and Servando (2026) was fitted on
    JTWC 1-minute sustained winds (their Section 2a, "1-min sustained wind from
    JTWC", and their Fig. 3 caption). Applying it to a 10-minute wind
    understates the decay, and because the wind-time term is quadratic in V0 the
    error grows with intensity: about 15% in the violent-typhoon band.
  * The track propagator is fitted on USA_WIND, the 1-minute field, which is the
    most complete intensity field in the WNP record. The generator carries a
    10-minute wind, so the feature is converted before the network sees it.

A single constant will not do. The WMO at-sea factor (10-min = 0.93 x 1-min,
Harper et al. 2008) holds near tropical-storm strength but not above it: the
empirical ratio between the two agencies in this archive rises from 0.93 at
tropical-storm strength to about 1.25 in the violent-typhoon band, as Zerrudo
and Servando note in their Section 4c. What is wanted here is not a physical
averaging-period conversion but the value the field the coefficient was fitted
to would have carried, so the empirical relation is the right object.

The map is two straight segments fitted to the median USA_WIND in each 5 kt
TOK_WIND bin over the 24,093 synoptic points in the WNP record 1977-2023 that
carry both fields, bins of at least 200 points, weighted by bin count:

    V_1min = 0.9333 * V_10min                     below the knee
    V_1min = 1.398  * V_10min - 19.81             above it

The segments meet at V_10min = 42.63 kt, so the map is continuous and monotone
over the whole range, and it is piecewise linear rather than tabulated so that
differencing it across a decay step does not manufacture structure. RMSE against
the bin medians is 2.30 kt. Below tropical-storm strength JMA assigns no wind at
all, so the lower segment is an extension of the weakest resolved ratio rather
than a fit, and it applies only to imputed tropical-depression winds where the
decay term is negligible.

Regenerate the coefficients with `python windconv.py --ibtracs <points file>`.
"""

import numpy as np

SLOPE = 1.398        # V_1min = SLOPE * V_10min + INTERCEPT, above the knee
INTERCEPT = -19.81
RATIO_LOW = 0.9333   # V_1min = RATIO_LOW * V_10min, below the knee
KNEE_10MIN = -INTERCEPT / (SLOPE - RATIO_LOW)     # 42.63 kt
KNEE_1MIN = RATIO_LOW * KNEE_10MIN                # 39.79 kt


def to_1min(v10):
    """RSMC Tokyo 10-minute sustained wind -> JTWC 1-minute sustained wind."""
    v = np.asarray(v10, dtype=float)
    return np.where(v < KNEE_10MIN, v * RATIO_LOW, SLOPE * v + INTERCEPT)


def to_10min(v1):
    """JTWC 1-minute sustained wind -> RSMC Tokyo 10-minute sustained wind."""
    v = np.asarray(v1, dtype=float)
    return np.where(v < KNEE_1MIN, v / RATIO_LOW, (v - INTERCEPT) / SLOPE)


def _rebuild(ibtracs, min_bin=200):
    import pandas as pd
    from data import TROPICAL_NATURES
    d = pd.read_csv(ibtracs, usecols=["SEASON", "BASIN", "NATURE", "TRACK_TYPE",
                                      "ISO_TIME", "USA_WIND", "TOK_WIND"],
                    low_memory=False)
    d = d[(d.SEASON.between(1977, 2023)) & (d.BASIN == "WP")
          & (d.TRACK_TYPE == "main") & (d.NATURE.isin(TROPICAL_NATURES))]
    d["t"] = pd.to_datetime(d.ISO_TIME, errors="coerce")
    d = d[d.t.dt.hour.isin((0, 6, 12, 18))]
    d["u"] = pd.to_numeric(d.USA_WIND, errors="coerce")
    d["k"] = pd.to_numeric(d.TOK_WIND, errors="coerce")
    m = d[(d.u > 0) & (d.k > 0)]
    x, y, n = [], [], []
    for lo in np.arange(20, 150, 5):
        s = m[(m.k >= lo) & (m.k < lo + 5)]
        if len(s) >= min_bin:
            x.append(lo + 2.5)
            y.append(float(s.u.median()))
            n.append(len(s))
    x, y, n = np.array(x), np.array(y), np.array(n, float)
    slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(n))
    ratio_low = y[0] / x[0]
    print(f"n matched = {len(m)}, bins = {len(x)}")
    print(f"SLOPE = {slope:.4f}\nINTERCEPT = {intercept:.4f}\n"
          f"RATIO_LOW = {ratio_low:.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibtracs", required=True)
    _rebuild(ap.parse_args().ibtracs)
