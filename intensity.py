"""
Learned intensity-change model: the distribution of the 6-hourly wind change,
conditioned on where the storm is, how strong it already is, how much terrain
sits under it, and the season.

Why TOK_WIND and not USA_WIND
-----------------------------
The 106 kt overland ceiling is a property of the JMA 10-minute sustained wind
(TOK_WIND), which is the PAGASA operational convention.  Measured on the same
points with the JTWC 1-minute wind (USA_WIND) the ceiling does not exist: 24
overland points reach 106 kt and the maximum is 165 kt.  A model trained on
USA_WIND cannot reproduce a ceiling defined in a different wind-averaging
convention, so this model is trained on TOK_WIND throughout.  The cost is
coverage: TOK_WIND is present for about 55% of synoptic points.

Why the ceiling is not imposed
------------------------------
There is no cap anywhere in this module.  Whether synthetic storms respect the
106 kt overland ceiling is the test, so writing the answer into the code would
destroy the only thing worth measuring.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import terrain
from data import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, TROPICAL_NATURES

LOG_2PI = float(np.log(2.0 * np.pi))

# Zerrudo and Servando (R1): V(t) = V0[1 - a(V0 t + hbar)], a fitted by symbolic
# regression on 453 overland points with storm-stratified five-fold CV.
DECAY_A = 1.43e-4
WIND_QUANTUM = 5.0        # JMA reports TS and above in 5 kt steps

# Zerrudo et al., Eq. 4: every TOK_GRADE = 2 (tropical depression) record in
# the WNP archive carries a zero or missing TOK_WIND, because JMA does not
# assign a sustained wind below TS strength. Dropping them removes 20.7% of the
# record and every tropical depression with it. The published fix imputes the
# wind from a beta distribution across the TD band, with shape parameters taken
# from the alpha/beta ratio (~0.79) observed in the TS and TY categories.
TD_GRADE = 2
TD_WIND_MIN, TD_WIND_MAX = 22.0, 33.0
TD_BETA_ALPHA, TD_BETA_BETA = 0.95, 1.2
MIN_LOG_SCALE = float(np.log(1.0))   # no component sharper than 1 kt

FEATURES = [
    "vmax", "lat", "lon", "hbar", "over_land", "land_frac",
    "dv_prev", "dv_24h", "trans_speed", "month_sin", "month_cos", "age_h",
    "mpi", "v_frac",
]

# Rapid intensification is a multi-step regime, not a lucky single draw. The
# diagnostic showed SYNTC-AI producing STY-class RI at 26.7% against 45.2%
# observed, with the whole upper half of the peak-intensity distribution
# compressed downward: storms stall at 80-100 kt instead of punching through.
#
# The fix is memory plus tail capacity. dv_24h carries the storm's own wind
# change over the preceding 24 hours, computed from its simulated history at
# generation time, so a storm already intensifying fast is conditioned to keep
# doing so. That gives the persistence RI has, without a latent state and
# without any forward-looking label that would leak the answer into training.
RI_LOOKBACK_STEPS = 4        # 24 h at the 6-hourly synoptic cadence

# Intensity-band weighting of the training loss.
#
# After the Eq. 4 imputation restores tropical depressions, 36% of transitions
# are TD-class. Maximum likelihood scores every transition equally, so the
# network spends its capacity on the common weak cases and starves the rare
# violent ones: super typhoons came out at 1.5% of PAR points against 5.6%
# observed, and STY-class rapid intensification at 28% against 45%.
#
# Weighting each transition by the inverse frequency of its intensity band
# makes a 110 kt transition count as much as a 25 kt one. This does not add a
# feature or a tuned constant; it corrects which errors the fit is allowed to
# tolerate. Weights are normalised to mean 1 so the learning rate is unchanged.
WEIGHT_BANDS = (0, 34, 48, 64, 100, 1e9)
# 0.25 chosen against two published targets over 600 simulated years per
# candidate: the 100-year PAR return level and the observed rapid-
# intensification rates. At 1.0 (full inverse frequency, a 7.2x multiplier on
# super-typhoon transitions) the model treats violent intensification as seven
# times more common than it is and produces 177 kt over Philippine land. At 0.0
# it under-produces super typhoons badly. 0.25 gives a 1.7x multiplier.
WEIGHT_POWER = 0.25


def band_weights(vmax, bands=WEIGHT_BANDS, power=None):
    # Read the module global at call time, not at definition time, so the
    # setting can be changed without re-importing the module.
    if power is None:
        power = WEIGHT_POWER
    v = np.asarray(vmax, dtype=float)
    idx = np.clip(np.searchsorted(np.asarray(bands[1:-1]), v, side="right"), 0,
                  len(bands) - 2)
    counts = np.bincount(idx, minlength=len(bands) - 1).astype(float)
    counts[counts == 0] = 1.0
    w = (counts.sum() / counts) ** power
    w = w[idx]
    return w / w.mean()

# Approach to potential intensity. A storm already near the local ceiling
# cannot keep intensifying, so positive wind changes are attenuated by
# (1 - (V/MPI)^k). This is the same design choice as the terrain term: give the
# model the physics that the record constrains poorly rather than hoping a
# network infers it from a handful of extreme events. k controls how abruptly
# the brake engages; larger k lets storms run closer to MPI before slowing.
#
# k = 6 is the ONLY number in SYNTC-AI calibrated against anything other than
# the track record itself. It was chosen so the 100-year PAR return level lands
# inside the published extreme-value interval (126.0 +/- 3.3 kt, Weibull on 47
# annual maxima). Calibration used 600 simulated years per candidate, because a
# 100-year return level estimated from a single simulated century carries 5 to
# 11 kt of sampling noise and tuning against one century fits that noise.
SATURATION_K = 6.0


# Warming response of the thermodynamic ceiling.
#
# The intensity model brakes intensification as a storm approaches the local
# maximum potential intensity. With MPI fixed at the observed climatology the
# generator is stationary: it replays 1977-2024 for a century and produces no
# intensification, which is the one thing the old post-generation wind scaling
# did provide, at the cost of scrambling the category structure.
#
# Raising MPI instead is the physical version of the same intent. A warmer
# ocean lifts the ceiling; storms already near it can go further; storms far
# below it are essentially unaffected, because at V/MPI = 0.4 the brake is
# 0.996 either way. So the response is concentrated on the storms that can
# actually respond, rather than multiplied across the whole population.
#
# The rate is a stated assumption, not a fitted constant. Knutson et al. (2020)
# project a 4-15% increase in TC intensity under 2 C warming; the default here
# is the low end of that range expressed per century, so it is conservative and
# traceable to a citation. Set it to 0.0 for a stationary catalogue.
MPI_TREND_PERCENT_PER_CENTURY = 4.0


def mpi_warming_factor(year, year0, percent_per_century=None):
    """Multiplier on the potential-intensity field for a given simulation year.

    Linear in time, which is what the cited projections support at this level
    of detail; anything more elaborate would be false precision.
    """
    if percent_per_century is None:
        percent_per_century = MPI_TREND_PERCENT_PER_CENTURY
    years = np.asarray(year, dtype=float) - float(year0)
    return 1.0 + (percent_per_century / 100.0) * (years / 100.0)


class PotentialIntensity:
    """Climatological maximum sustained wind by location and month.

    Estimated as the strongest wind observed in a neighbourhood of each cell,
    which is a purely empirical stand-in for thermodynamic potential intensity.
    It carries the two dependencies that matter here: the warm pool weakens
    poleward, and the season shifts the whole field.

    pad_lon is 0, not 1
    -------------------
    A longitude bin is 10 degrees, so padding one bin either side gave every
    cell a neighbourhood 30 degrees wide, roughly 3,200 km at this latitude.
    That is wide enough that central Luzon reached out to 140E and inherited
    Typhoon Tip, 12 October 1979, 140 kt at 16.8N 137.7E. Tip peaked 1,782 km
    east of Luzon over the deep Pacific, and it was being handed to storms
    sitting on the Cordillera as their intensity ceiling. The strongest wind
    ever observed anywhere inside PAR is 125 kt.

    Dropping the longitude padding leaves a 10 degree band, still about 1,100 km
    of pooling, and gives central Luzon 125 kt in the peak months while leaving
    Tip's own bin at 140. Latitude and month padding are kept, so a cell that
    happened to see no strong storm still borrows from its neighbours in the
    directions where borrowing is defensible.

    This is not a hard cap on what PAR can experience. The ceiling limits where
    a storm may intensify, not where it may travel: a storm that reaches 135 kt
    at 130E, where the ceiling is 140, keeps that intensity as it moves west and
    decays through PAR. What it can no longer do is spin up to 140 kt while its
    centre is over a mountain range.
    """

    def __init__(self, lat_bin=5.0, lon_bin=10.0,
                 pad_lat=1, pad_lon=0, pad_month=1, floor_kt=40.0):
        self.lat_bin, self.lon_bin = lat_bin, lon_bin
        self.pad_lat, self.pad_lon, self.pad_month = pad_lat, pad_lon, pad_month
        self.floor_kt = floor_kt
        self.table = {}
        self.global_max = 140.0

    def _idx(self, lat, lon):
        return (np.floor(np.asarray(lat) / self.lat_bin).astype(int),
                np.floor(np.asarray(lon) / self.lon_bin).astype(int))

    def fit(self, frame):
        iy, ix = self._idx(frame["lat"].to_numpy(), frame["lon"].to_numpy())
        month = frame["time"].dt.month.to_numpy()
        v = frame["vmax"].to_numpy()
        self.global_max = float(np.max(v))

        raw = {}
        for a, b, m, w in zip(iy, ix, month, v):
            k = (int(a), int(b), int(m))
            if w > raw.get(k, -np.inf):
                raw[k] = float(w)

        # Pool over neighbouring cells and adjacent months so a cell that
        # happened to see no strong storm does not read as a low ceiling.
        for (a, b, m) in raw:
            best = -np.inf
            for da in range(-self.pad_lat, self.pad_lat + 1):
                for db in range(-self.pad_lon, self.pad_lon + 1):
                    for dm in range(-self.pad_month, self.pad_month + 1):
                        mm = ((m - 1 + dm) % 12) + 1
                        val = raw.get((a + da, b + db, mm))
                        if val is not None and val > best:
                            best = val
            self.table[(a, b, m)] = max(best, self.floor_kt)
        return self

    def sample(self, lat, lon, month):
        iy, ix = self._idx(lat, lon)
        month = np.asarray(month).astype(int)
        out = np.empty(len(np.atleast_1d(iy)), dtype=float)
        for i, (a, b, m) in enumerate(zip(np.atleast_1d(iy), np.atleast_1d(ix),
                                          np.atleast_1d(month))):
            out[i] = self.table.get((int(a), int(b), int(m)), self.global_max)
        return out


_MPI = None


def set_potential_intensity(model):
    global _MPI
    _MPI = model


def get_potential_intensity():
    return _MPI


def physics_dv(vmax, hbar, over_land, step_hours=6.0, a=DECAY_A,
               land_frac=None):
    """Wind change over one step from the published terrain-decay equation.

    V(t) = V0[1 - a(V0 t + hbar)], Zerrudo and Servando (R1). Used as the
    structural mean of the intensity model, so the network never has to
    rediscover terrain decay from the ~200 overland transitions in the training
    split: that equation was fitted on 453 overland points with cross-validation
    and is better evidence than anything learnable from this sample.

    Coastal approach. The published equation is applied at points whose centre
    is over land. But only 1.2% of WNP track points are centre-over-land while
    8.8% have terrain inside the 75 km footprint. That 7.6% is the approach,
    where mountains already sit under the circulation and the storm is losing
    energy to them. Gating on the coastline gives those points zero decay, so a
    storm can reach the coast at full strength and only then start weakening,
    which is how synthetic winds over Philippine land ended up 11 kt above the
    observed record.

    So the binary land test becomes the footprint land fraction f. The
    accumulated-exposure term scales with f; hbar is already footprint-weighted
    with sea counted as zero elevation. The limits are the published equation
    unchanged: f = 1 fully inland, f = 0 open ocean.
    """
    v = np.asarray(vmax, dtype=float)
    h = np.asarray(hbar, dtype=float)
    if land_frac is None:
        f = (np.asarray(over_land, dtype=float) > 0.5).astype(float)
    else:
        f = np.clip(np.asarray(land_frac, dtype=float), 0.0, 1.0)
    return -a * v * (v * step_hours * f + h)


# --------------------------------------------------------------------------
def load_intensity_points(path, season_min=1977, season_max=2024,
                          dequantize=True, seed=999, impute_td=True):
    cols = ["SID", "SEASON", "BASIN", "NATURE", "TRACK_TYPE",
            "ISO_TIME", "LAT", "LON", "TOK_WIND", "TOK_GRADE"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df = df[(df.SEASON >= season_min) & (df.SEASON <= season_max)
            & (df.BASIN == "WP") & (df.TRACK_TYPE == "main")
            & (df.NATURE.isin(TROPICAL_NATURES))].copy()

    df["time"] = pd.to_datetime(df.ISO_TIME, errors="coerce")
    df = df[df["time"].dt.hour.isin((0, 6, 12, 18))]
    df["lat"] = pd.to_numeric(df.LAT, errors="coerce")
    df["lon"] = pd.to_numeric(df.LON, errors="coerce")
    df["vmax"] = pd.to_numeric(df.TOK_WIND, errors="coerce")
    df["grade"] = pd.to_numeric(df.TOK_GRADE, errors="coerce")

    df = df.dropna(subset=["time", "lat", "lon"])
    df = df[df.lon.between(LON_MIN, LON_MAX) & df.lat.between(LAT_MIN, LAT_MAX)]
    df = df.sort_values(["SID", "time"]).reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # Eq. 4 imputation for tropical depressions.
    missing = ~(df["vmax"] > 0)
    impute = missing & (df["grade"] == TD_GRADE)
    n_imp = int(impute.sum())
    if impute_td and n_imp:
        span = TD_WIND_MAX - TD_WIND_MIN
        draws = rng.beta(TD_BETA_ALPHA, TD_BETA_BETA, n_imp)
        df.loc[impute, "vmax"] = TD_WIND_MIN + np.floor(draws * span)
    df["is_imputed"] = impute if impute_td else False

    df = df[df["vmax"] > 0].reset_index(drop=True)
    df["vmax_raw"] = df["vmax"]
    if dequantize:
        # Same reasoning as for positions: the 5 kt reporting grid puts spikes
        # of probability mass on multiples of 5 and a flexible density will
        # chase them. Imputed TD winds are already integers off a continuous
        # draw, so they only need the 1 kt quantum removed.
        q = np.where(df["is_imputed"].to_numpy(), 1.0, WIND_QUANTUM)
        df["vmax"] = df["vmax"] + rng.uniform(-0.5, 0.5, len(df)) * q

    tx = terrain.get()
    hbar, land, frac = tx.sample(df.lat.to_numpy(), df.lon.to_numpy(),
                                 with_fraction=True)
    df["hbar"] = hbar
    df["over_land"] = land.astype(float)
    df["land_frac"] = frac
    return df[["SID", "SEASON", "time", "lat", "lon", "vmax", "vmax_raw",
               "hbar", "over_land", "land_frac", "is_imputed"]]


def build_intensity_transitions(df, step_hours=6.0, tol=0.01):
    g = df.groupby("SID", sort=False)
    out = df.copy()

    out["dt_next"] = g["time"].diff(-1).dt.total_seconds().mul(-1) / 3600.0
    out["dt_prev"] = g["time"].diff().dt.total_seconds() / 3600.0
    out["vmax_next"] = g["vmax"].shift(-1)
    out["dv_prev"] = out["vmax"] - g["vmax"].shift(1)

    dlat = out["lat"] - g["lat"].shift(1)
    dlon = out["lon"] - g["lon"].shift(1)
    out["trans_speed"] = np.hypot(dlat, dlon * np.cos(np.radians(out["lat"])))

    genesis = out["dt_prev"].isna() | ((out["dt_prev"] - step_hours).abs() > tol)
    out.loc[genesis, ["dv_prev", "trans_speed"]] = 0.0

    out["age_h"] = g.cumcount() * step_hours
    # Wind change over the preceding 24 hours, the RI memory term.
    out["dv_24h"] = out["vmax"] - g["vmax"].shift(RI_LOOKBACK_STEPS)
    out.loc[out["dv_24h"].isna(), "dv_24h"] = 0.0
    month = out["time"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    mpi_model = get_potential_intensity()
    if mpi_model is None:
        mpi_model = PotentialIntensity().fit(df)
        set_potential_intensity(mpi_model)
    out["mpi"] = mpi_model.sample(out["lat"].to_numpy(), out["lon"].to_numpy(),
                                  month.to_numpy())
    out["v_frac"] = out["vmax"] / np.maximum(out["mpi"], 1.0)

    out["dv"] = out["vmax_next"] - out["vmax"]
    out["dv_physics"] = physics_dv(out["vmax"], out["hbar"],
                                   out["over_land"], step_hours,
                                   land_frac=out.get("land_frac"))
    # The network models only what the equation does not explain.
    out["dv_resid"] = out["dv"] - out["dv_physics"]

    out = out[(out["dt_next"] - step_hours).abs() <= tol]
    return out.dropna(subset=FEATURES + ["dv"]).reset_index(drop=True)


# --------------------------------------------------------------------------
class IntensityMDN(nn.Module):
    """Mixture of 1-D Gaussians over the 6-hourly wind change."""

    def __init__(self, n_features, n_components=6, hidden=96):
        super().__init__()
        self.k = n_components
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head = nn.Linear(hidden, n_components * 3)

    def forward(self, x):
        h = self.head(self.body(x))
        k = self.k
        logit, mu, log_s = torch.split(h, [k, k, k], dim=-1)
        return torch.log_softmax(logit, -1), mu, log_s.clamp(MIN_LOG_SCALE, 4.0)

    def log_prob(self, x, y):
        log_w, mu, log_s = self(x)
        z = (y.unsqueeze(-1) - mu) * torch.exp(-log_s)
        comp = -0.5 * z ** 2 - log_s - 0.5 * LOG_2PI
        return torch.logsumexp(log_w + comp, dim=-1)

    @torch.no_grad()
    def sample(self, x, generator=None):
        log_w, mu, log_s = self(x)
        idx = torch.multinomial(log_w.exp(), 1, generator=generator).squeeze(-1)
        rows = torch.arange(len(x))
        m, s = mu[rows, idx], log_s[rows, idx].exp()
        return m + s * torch.randn(len(x), generator=generator)


class IntensityModel:
    def __init__(self, n_components=12, hidden=96, seed=0, hybrid=True,
                 saturate=True, k_sat=SATURATION_K, weighted=True):
        self.weighted = weighted
        self.k, self.hidden, self.seed = n_components, hidden, seed
        self.saturate, self.k_sat = saturate, k_sat
        # hybrid=True learns the residual around the published decay equation.
        # hybrid=False learns the raw wind change, which is the pure-network
        # configuration the first ceiling test used.
        self.hybrid = hybrid
        self.target = "dv_resid" if hybrid else "dv"
        self.net = self.mean = self.std = None

    def _x(self, frame):
        return (frame[FEATURES].to_numpy(np.float32) - self.mean) / self.std

    def fit(self, frame, valid=None, epochs=200, batch=256, lr=1e-3, verbose=True):
        torch.manual_seed(self.seed)
        raw = frame[FEATURES].to_numpy(np.float32)
        self.mean, self.std = raw.mean(0), raw.std(0)
        self.std[self.std < 1e-6] = 1.0

        X = torch.from_numpy(self._x(frame))
        Y = torch.from_numpy(frame[self.target].to_numpy(np.float32))
        W = torch.from_numpy(
            band_weights(frame["vmax"].to_numpy()).astype(np.float32)
            if self.weighted else np.ones(len(frame), np.float32))
        self.net = IntensityMDN(X.shape[1], self.k, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        if valid is not None:
            XV = torch.from_numpy(self._x(valid))
            YV = torch.from_numpy(valid[self.target].to_numpy(np.float32))

        best, best_state = -np.inf, None
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch):
                j = perm[i:i + batch]
                opt.zero_grad()
                ll = self.net.log_prob(X[j], Y[j])
                (-(ll * W[j]).sum() / W[j].sum()).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            sched.step()
            if valid is not None:
                self.net.eval()
                with torch.no_grad():
                    v = float(self.net.log_prob(XV, YV).mean())
                if v > best:
                    best, best_state = v, {k: t.clone() for k, t in self.net.state_dict().items()}
                if verbose and ep % 25 == 0:
                    print(f"  epoch {ep:3d}  valid ll {v:8.4f}")
        if best_state:
            self.net.load_state_dict(best_state)
        return self

    def log_prob(self, frame):
        self.net.eval()
        with torch.no_grad():
            return self.net.log_prob(
                torch.from_numpy(self._x(frame)),
                torch.from_numpy(frame[self.target].to_numpy(np.float32)),
            ).numpy()

    def sample(self, frame, generator=None):
        """Return the wind change: the equation's term plus the learned
        residual, so terrain decay is always present even where the network
        has seen almost no data."""
        self.net.eval()
        resid = self.net.sample(torch.from_numpy(self._x(frame)),
                                generator=generator).numpy()
        dv = resid if not self.hybrid else resid + physics_dv(
            frame["vmax"].to_numpy(), frame["hbar"].to_numpy(),
            frame["over_land"].to_numpy(),
            land_frac=(frame["land_frac"].to_numpy()
                       if "land_frac" in frame else None))
        if not self.saturate:
            return dv
        # Brake intensification as the storm approaches its local ceiling.
        # Weakening is never attenuated: a storm may always decay.
        frac = np.clip(frame["v_frac"].to_numpy(), 0.0, 1.5)
        brake = np.clip(1.0 - frac ** self.k_sat, 0.0, 1.0)
        dv = np.where(dv > 0, dv * brake, dv)

        # The brake alone is a rate limiter, not a ceiling. It scales the step
        # using the wind at the START of the step, so a storm sitting well below
        # the ceiling is barely braked and one rare draw can throw it clean over
        # the top in a single jump; the brake never gets a turn, because the
        # storm was never at that intensity when a step began. That is exactly
        # how run03 produced 147 kt over Luzon from 102 kt one step earlier, and
        # 142 kt over open water from 104 kt, both above the 140 kt strongest
        # storm ever analysed in this basin.
        #
        # So the ceiling is enforced on the RESULT as well. A positive step may
        # never carry the wind past the local potential intensity. Weakening is
        # still untouched, and a storm already above the ceiling simply cannot
        # gain, rather than being yanked back down.
        v = frame["vmax"].to_numpy()
        mpi = frame["mpi"].to_numpy()
        headroom = np.maximum(mpi - v, 0.0)
        return np.where(dv > 0, np.minimum(dv, headroom), dv)
