"""
Build 3-hourly track-increment training data from full-basin IBTrACS WNP.

Design notes
------------
The point of this dataset is to answer: "given a storm at (lat, lon) moving a
certain way, with a certain intensity, in a certain month, where does it go in
the next 3 hours?"  So each training row is one 3-hourly transition.

We deliberately use the FULL Western North Pacific basin, not a PAR-clipped
extract.  Recurvature happens between 20N and 35N; a dataset clipped at 25N
cannot teach a model where storms turn, which is why hand-tuned recurvature
constants were needed in the previous generator.
"""

import numpy as np
import pandas as pd

# Tropical phases only. 'ET' (extratropical) marks the end of the tropical
# lifecycle; a TC hazard generator terminates there rather than following the
# storm into the midlatitude westerlies.
TROPICAL_NATURES = ("TS", "DS", "SS", "NR")

QUANTUM = 0.1          # IBTrACS stores positions to one decimal degree

LON_MIN, LON_MAX = 100.0, 200.0
LAT_MIN, LAT_MAX = 0.0, 50.0

FEATURES = [
    "lat", "lon",          # position
    "u_prev", "v_prev",    # previous 3-h displacement (deg), the motion vector
    "vmax",                # intensity, kt (1-min sustained, USA_WIND)
    "month_sin", "month_cos",
    "age_h",               # hours since genesis
    "is_genesis",          # 1 on the first propagated step, else 0
]
TARGETS = ["dlon", "dlat"]


def load_tracks(path, season_min=1977, season_max=2024, synoptic_only=False,
                dequantize=True, dequantize_seed=12345):
    """Read IBTrACS points and return one tidy row per observation.

    `synoptic_only` keeps just 00/06/12/18 UTC.  IBTrACS carries 3-hourly
    positions for the WNP, and some of the off-synoptic points may be agency
    interpolations rather than independent fixes.  Re-running the whole
    benchmark on the synoptic subset at a 6-hour step is the check that any
    measured skill is real and not an artefact of interpolated smoothness.
    """
    cols = [
        "SID", "SEASON", "BASIN", "NATURE", "TRACK_TYPE",
        "ISO_TIME", "LAT", "LON", "USA_WIND", "TOK_WIND", "WMO_WIND",
    ]
    df = pd.read_csv(path, usecols=cols, low_memory=False)

    df = df[
        (df.SEASON >= season_min)
        & (df.SEASON <= season_max)
        & (df.BASIN == "WP")
        & (df.TRACK_TYPE == "main")
        & (df.NATURE.isin(TROPICAL_NATURES))
    ].copy()

    df["time"] = pd.to_datetime(df.ISO_TIME, errors="coerce")
    df["lat"] = pd.to_numeric(df.LAT, errors="coerce")
    df["lon"] = pd.to_numeric(df.LON, errors="coerce")

    # Intensity: USA_WIND (1-min) is the most complete field in the WNP record.
    # Gaps are filled inside each storm by time interpolation, then by the
    # storm's own median, so an occasional missing wind does not drop a whole
    # transition.
    df["vmax"] = pd.to_numeric(df.USA_WIND, errors="coerce")

    df = df.dropna(subset=["time", "lat", "lon"])
    df = df[
        df.lon.between(LON_MIN, LON_MAX) & df.lat.between(LAT_MIN, LAT_MAX)
    ]
    if synoptic_only:
        df = df[df["time"].dt.hour.isin((0, 6, 12, 18))]

    df = df.sort_values(["SID", "time"]).reset_index(drop=True)

    # IBTrACS positions are stored on a 0.1 degree grid.  Left as-is, that
    # rounding puts a hard spike of probability mass on discrete values and a
    # flexible density model will chase it with vanishing variance, inflating
    # log-likelihood without learning anything about storm motion.  Treating
    # each rounded value as uniform over its 0.1 degree cell is the standard
    # correction for discretised measurements.
    df["lat_raw"] = df["lat"]
    df["lon_raw"] = df["lon"]
    if dequantize:
        rng = np.random.default_rng(dequantize_seed)
        half = QUANTUM / 2.0
        df["lat"] = df["lat"] + rng.uniform(-half, half, len(df))
        df["lon"] = df["lon"] + rng.uniform(-half, half, len(df))

    df["vmax"] = (
        df.groupby("SID")["vmax"]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )
    df["vmax"] = df["vmax"].fillna(df["vmax"].median())

    return df[["SID", "SEASON", "time", "lat", "lon", "lat_raw", "lon_raw", "vmax"]]


def build_transitions(df, step_hours=3.0, tol=0.01, drop_interpolated=True):
    """Turn per-storm point sequences into (features, target) transition rows.

    Only consecutive pairs exactly `step_hours` apart are kept, so a gap in the
    record never becomes a fake giant displacement.
    """
    g = df.groupby("SID", sort=False)

    out = df.copy()
    out["lat_next"] = g["lat"].shift(-1)
    out["lon_next"] = g["lon"].shift(-1)
    out["dt_next"] = g["time"].diff(-1).dt.total_seconds().mul(-1) / 3600.0

    # Previous displacement = the motion vector the storm arrived with.
    out["u_prev"] = out["lon"] - g["lon"].shift(1)
    out["v_prev"] = out["lat"] - g["lat"].shift(1)
    out["dt_prev"] = g["time"].diff().dt.total_seconds() / 3600.0

    # A genesis step has no incoming motion; the model is told so explicitly
    # rather than being fed a fabricated zero it cannot distinguish from a
    # genuinely stalled storm.
    genesis = out["dt_prev"].isna() | (
        (out["dt_prev"] - step_hours).abs() > tol
    )
    out["is_genesis"] = genesis.astype(float)
    out.loc[genesis, ["u_prev", "v_prev"]] = 0.0

    out["age_h"] = g.cumcount() * step_hours

    out["dlon"] = out["lon_next"] - out["lon"]
    out["dlat"] = out["lat_next"] - out["lat"]

    month = out["time"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # Part of the WNP record is agency interpolation between real fixes: those
    # points reproduce the previous displacement to the last stored digit.  On
    # such a run the "next" position is a deterministic function of the last
    # two, so any density model can drive its variance to zero there and score
    # an arbitrarily high likelihood that reflects IBTrACS bookkeeping, not
    # atmospheric predictability.  The flag is computed on the raw grid values,
    # before dequantisation, which is the only place the duplication is exact.
    d_raw = np.column_stack([
        (g["lon_raw"].shift(-1) - out["lon_raw"]).to_numpy(),
        (g["lat_raw"].shift(-1) - out["lat_raw"]).to_numpy(),
    ])
    p_raw = np.column_stack([
        (out["lon_raw"] - g["lon_raw"].shift(1)).to_numpy(),
        (out["lat_raw"] - g["lat_raw"].shift(1)).to_numpy(),
    ])
    out["is_interpolated"] = (
        (np.abs(d_raw - p_raw) < 1e-9).all(axis=1) & (out["is_genesis"] == 0)
    )

    valid = (out["dt_next"] - step_hours).abs() <= tol
    out = out[valid].dropna(subset=FEATURES + TARGETS)

    if drop_interpolated:
        out = out[~out["is_interpolated"]]

    return out.reset_index(drop=True)


def genesis_points(df):
    """First observation of each storm: the seeds an autoregressive run needs."""
    first = df.sort_values(["SID", "time"]).groupby("SID", sort=False).head(1)
    return first[["SID", "SEASON", "time", "lat", "lon", "vmax"]].reset_index(drop=True)


def track_lengths(df):
    """Number of 3-hourly steps per storm, used to bound simulated tracks."""
    return df.groupby("SID").size()


def split_by_season(frame, train_max=2014):
    """Chronological split. Random splits would leak: points from one storm
    would land on both sides, and neighbouring steps are near-duplicates."""
    return frame[frame.SEASON <= train_max], frame[frame.SEASON > train_max]
