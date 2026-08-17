"""
SYNTC-AI: synthetic tropical cyclone generator for the Philippine Area of
Responsibility, learned from IBTrACS 1977-2024.

    python syntc_ai.py --years 2026 2125 --ensembles 20 --out D:/2026/SYNTC-AI/run01

This is the replacement for SYNTC. It does the same five jobs:

    1. how many storms per year   -> historical distribution, or your SARIMAX
                                     counts via --counts-csv
    2. where and when they form   -> genesis density learned per month
    3. where each storm moves     -> learned track propagator (MDN)
    4. how strong it is           -> learned intensity model + the published
                                     terrain-decay equation
    5. the wind field             -> Willoughby et al. (2006), published
                                     coefficients

What is different from SYNTC
----------------------------
Every constant lives in CONFIG below. There are no tuning constants buried in
the generation code, no PATCH flags, and no post-hoc caps. If the model puts a
storm somewhere wrong, the fix is training data or a CONFIG value, not a new
magic number three thousand lines down.

Two honest limitations, stated here rather than discovered later:

  * Intensity is trained on TOK_WIND (JMA 10-minute), because the 106 kt
    overland ceiling is a property of that wind convention. TOK_WIND is only
    reported at roughly TS strength and above, so tropical depressions are
    weakly constrained.
  * The overland ceiling is still not reproduced. Over 600 simulated years the
    model reaches about 121 kt over Philippine land against an observed record
    maximum of 110 kt (Haiyan, 2013). Set CONFIG.overland_cap_kt to enforce it,
    but know that you are then imposing the result rather than predicting it.
    Default is None, i.e. no cap, i.e. honest.
  * Super typhoons are under-produced: about 2.9% of PAR track points against
    5.6% observed. The 100-year return level is right, so the distribution has
    the correct ceiling but too little mass just below it.

Calibration status
------------------
The 100-year PAR return level is 125.6 kt over 600 simulated years, against the
published extreme-value estimate of 126.0 +/- 3.3 kt. That agreement comes from
one calibrated parameter, intensity.SATURATION_K; everything else is fitted to
the IBTrACS record or taken from published equations.

Author: Jef Zerrudo (DOST-PAGASA). Track and intensity models built with
Claude. Terrain decay from Zerrudo and Servando; wind profile from Willoughby
et al. (2006) as evaluated for the WNP in Zerrudo and Bala.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde

import data as D
import intensity as I
import terrain
from models import MDNPropagator


# ===========================================================================
# CONFIG - every tunable number in the model lives here and nowhere else
# ===========================================================================
@dataclass
class Config:
    # --- data ---------------------------------------------------------
    # No defaults. A path baked in here is a path that exists on exactly one
    # machine, and a wrong one fails deep inside rasterio rather than at the
    # command line. Both are required arguments.
    ibtracs: str = None
    dtm: str = None
    season_min: int = 1977
    # 2023, not 2024. IBTrACS v04r01 is provisional for the most recent season,
    # which is why return_levels.py, validate_hotspots.py and plot_results.py
    # all evaluate against 1977-2023. If a season is too provisional to validate
    # against then it is too provisional to fit on, and a model trained over a
    # period the paper never validates leaves two training periods in the
    # methods section with no way to justify either.
    season_max: int = 2023
    step_hours: float = 6.0          # synoptic cadence; see note in data.py

    # --- domain -------------------------------------------------------
    lon_min: float = 100.0
    lon_max: float = 180.0
    lat_min: float = 0.0
    lat_max: float = 45.0
    # Genesis is restricted to the corridor that actually feeds PAR.
    genesis_lon_max: float = 160.0

    # --- PAR hexagon (PAGASA official vertices) -----------------------
    par_vertices: tuple = (
        (25.0, 120.0), (25.0, 135.0), (5.0, 135.0),
        (5.0, 115.0), (15.0, 115.0), (21.0, 120.0),
    )

    # --- models -------------------------------------------------------
    track_components: int = 8
    track_hidden: int = 128
    intensity_components: int = 12
    intensity_hidden: int = 96
    epochs: int = 200
    train_max_season: int = 2014     # fit years; later years are held out
    valid_max_season: int = 2009     # early-stopping split inside the fit years
    seed: int = 42

    # --- storm lifecycle ----------------------------------------------
    genesis_kde_bandwidth: float = 0.25
    min_wind_kt: float = 20.0        # below this the storm has dissipated
    max_age_hours: float = 480.0     # 20 days
    max_steps_outside_domain: int = 2
    # "par": counts from --counts-csv mean storms entering PAR (what a SARIMAX
    # fitted to PAR-entry counts predicts). "basin": counts mean storms formed
    # anywhere in the generation domain.
    counts_mode: str = "par"
    max_genesis_rounds: int = 6
    # Percentage increase in maximum potential intensity per century, applied
    # linearly from the first simulated year. 0.0 gives a stationary-climate
    # catalogue. The default of 4.0 is the low end of the 4-15% intensification
    # projected under 2 C warming by Knutson et al. (2020), so the catalogue is
    # conservative and the assumption is a citable input rather than a tuned
    # constant. This is the ONLY place climate change enters the model.
    mpi_trend_percent_per_century: float = 4.0

    # --- physics ------------------------------------------------------
    decay_a: float = 1.43e-4         # Zerrudo & Servando (R1)
    terrain_footprint_km: float = 75.0
    p_env_hpa: float = 1010.0
    # Set to a number (e.g. 106.0) to force the overland ceiling. Leaving it
    # None means the ceiling is a prediction, not an assumption.
    overland_cap_kt: float | None = None

    # --- PAGASA intensity classes (kt, 10-minute sustained) -----------
    categories: tuple = (
        ("TD", 22, 33), ("TS", 34, 47), ("STS", 48, 63),
        ("TY", 64, 99), ("STY", 100, 999),
    )

    # --- output -------------------------------------------------------
    wind_radii_kt: tuple = (34, 50, 64)


CONFIG = Config()


# ===========================================================================
# Physics: published relations, used as published
# ===========================================================================
def category_of(wind_kt, cfg=CONFIG):
    w = np.asarray(wind_kt)
    out = np.full(w.shape, "TD", dtype=object)
    for name, lo, hi in cfg.categories:
        out[(w >= lo) & (w <= hi)] = name
    return out


def central_pressure(wind_kt, lat_deg, cfg=CONFIG):
    """Atkinson and Holliday (1977) with the Holland-style latitude term."""
    v = np.asarray(wind_kt, dtype=float)
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    base = cfg.p_env_hpa - (v / 5.896) ** (1.0 / 0.644)
    return base * (1.5 - 0.5 * np.cos(np.abs(phi))) / 1.5


def radius_max_wind(wind_kt, lat_deg):
    """Knaff et al. (2015) climatological RMW, in km."""
    v = np.asarray(wind_kt, dtype=float)
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    rmw = (218.3784 - 1.2014 * v + (v / 10.9884) ** 2
           - (v / 35.3052) ** 3 - 145.5090 * np.cos(phi))
    return np.clip(rmw, 10.0, 300.0)


def willoughby_profile(r_km, vmax_kt, rmax_km, lat_deg):
    """Willoughby et al. (2006) dual-exponential radial wind profile.

    The published coefficients are used unchanged. Zerrudo and Bala found that
    a WNP-specific refit does not beat them out of sample, because the skill
    comes from the functional form rather than from basin-tuned constants.
    """
    v_ms = np.asarray(vmax_kt, dtype=float) * 0.514444
    rmax = np.asarray(rmax_km, dtype=float)
    lat = np.abs(np.asarray(lat_deg, dtype=float))
    r = np.asarray(r_km, dtype=float)

    n = 2.1340 + 0.0077 * v_ms - 0.4522 * np.log(rmax) - 0.0038 * lat
    X1 = 287.6 - 1.942 * v_ms + 7.799 * np.log(rmax) + 1.819 * lat
    A = np.clip(0.5913 + 0.0029 * v_ms - 0.1361 * np.log(rmax) - 0.0042 * lat,
                0.0, 1.0)
    X2 = 25.0

    inner = np.where(rmax > 0, (r / np.maximum(rmax, 1e-6)) ** n, 0.0)
    dr = np.maximum(r - rmax, 0.0)
    outer = (1.0 - A) * np.exp(-dr / X1) + A * np.exp(-dr / X2)
    shape = np.where(r < rmax, inner, outer)
    return np.asarray(vmax_kt, dtype=float) * shape


def wind_radius(threshold_kt, vmax_kt, rmax_km, lat_deg, r_max_search=800.0):
    """Radius at which the Willoughby profile falls to `threshold_kt`."""
    grid = np.linspace(0.0, r_max_search, 400)
    out = np.full(np.shape(vmax_kt), np.nan)
    vmax_kt = np.atleast_1d(vmax_kt)
    rmax_km = np.atleast_1d(rmax_km)
    lat_deg = np.atleast_1d(lat_deg)
    out = np.full(len(vmax_kt), np.nan)
    for i in range(len(vmax_kt)):
        if vmax_kt[i] <= threshold_kt:
            continue
        prof = willoughby_profile(grid, vmax_kt[i], rmax_km[i], lat_deg[i])
        beyond = grid >= rmax_km[i]
        below = beyond & (prof < threshold_kt)
        if below.any():
            out[i] = grid[np.argmax(below)]
    return out


def in_par(lat, lon, cfg=CONFIG):
    """Point-in-polygon for the PAGASA PAR hexagon (ray casting)."""
    verts = np.array(cfg.par_vertices)
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    inside = np.zeros(len(lat), dtype=bool)
    n = len(verts)
    for i in range(n):
        y1, x1 = verts[i]
        y2, x2 = verts[(i + 1) % n]
        cond = ((y1 > lat) != (y2 > lat))
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
        inside ^= cond & (lon < xin)
    return inside


# ===========================================================================
# Genesis: where and when storms form
# ===========================================================================
class GenesisModel:
    """Per-month kernel density over genesis position, plus the monthly share
    of storms and the distribution of genesis intensity.

    A KDE rather than a fitted parametric surface because genesis in the WNP is
    multi-modal (monsoon trough, Philippine Sea warm pool, South China Sea) and
    a mixture chosen by hand is exactly the kind of decision this rewrite is
    meant to remove.
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.kde = {}
        self.month_p = None
        self.wind0 = {}

    def fit(self, points):
        first = points.sort_values(["SID", "time"]).groupby("SID").head(1).copy()
        first["month"] = first["time"].dt.month
        first = first[first.lon <= self.cfg.genesis_lon_max]

        counts = first.groupby("month").size().reindex(range(1, 13), fill_value=0)
        self.month_p = (counts / counts.sum()).to_numpy()

        allpts = np.vstack([first.lon.to_numpy(), first.lat.to_numpy()])
        for m in range(1, 13):
            g = first[first.month == m]
            xy = (np.vstack([g.lon.to_numpy(), g.lat.to_numpy()])
                  if len(g) >= 20 else allpts)
            self.kde[m] = gaussian_kde(xy, bw_method=self.cfg.genesis_kde_bandwidth)
            self.wind0[m] = (g.vmax.to_numpy() if len(g) >= 20
                             else first.vmax.to_numpy())
        return self

    def sample(self, n, rng):
        months = rng.choice(np.arange(1, 13), size=n, p=self.month_p)
        lon = np.empty(n)
        lat = np.empty(n)
        wind = np.empty(n)
        for m in np.unique(months):
            k = months == m
            xy = self.kde[m].resample(int(k.sum()), seed=int(rng.integers(1 << 31)))
            lon[k], lat[k] = xy[0], xy[1]
            wind[k] = rng.choice(self.wind0[m], int(k.sum()))
        lon = np.clip(lon, self.cfg.lon_min, self.cfg.genesis_lon_max)
        lat = np.clip(lat, self.cfg.lat_min + 1.0, self.cfg.lat_max - 5.0)
        return months, lat, lon, wind


# ===========================================================================
# The generator
# ===========================================================================
class SyntcAI:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.genesis = None
        self.track = None
        self.intensity = None
        self.annual_counts = None

    # -- training ------------------------------------------------------
    def fit(self, verbose=True):
        cfg = self.cfg
        terrain.DTM_PATH = cfg.dtm

        if verbose:
            print("loading IBTrACS ...")
        track_pts = D.load_tracks(cfg.ibtracs, cfg.season_min, cfg.season_max,
                                  synoptic_only=True)
        track_tr = D.build_transitions(track_pts, step_hours=cfg.step_hours)
        t_fit = track_tr[track_tr.SEASON <= cfg.valid_max_season]
        t_val = track_tr[(track_tr.SEASON > cfg.valid_max_season)
                         & (track_tr.SEASON <= cfg.train_max_season)]

        int_pts = I.load_intensity_points(cfg.ibtracs, cfg.season_min, cfg.season_max)
        int_tr = I.build_intensity_transitions(int_pts, step_hours=cfg.step_hours)
        i_fit = int_tr[int_tr.SEASON <= cfg.valid_max_season]
        i_val = int_tr[(int_tr.SEASON > cfg.valid_max_season)
                       & (int_tr.SEASON <= cfg.train_max_season)]

        if verbose:
            print(f"  track transitions     {len(t_fit):,} fit / {len(t_val):,} valid")
            print(f"  intensity transitions {len(i_fit):,} fit / {len(i_val):,} valid")
            print("training track propagator ...")
        self.track = MDNPropagator(cfg.track_components, cfg.track_hidden,
                                   seed=cfg.seed).fit(t_fit, t_val,
                                                      epochs=cfg.epochs,
                                                      verbose=False)
        if verbose:
            print("training intensity model ...")
        self.intensity = I.IntensityModel(cfg.intensity_components,
                                          cfg.intensity_hidden,
                                          seed=cfg.seed,
                                          hybrid=True).fit(i_fit, i_val,
                                                           epochs=cfg.epochs,
                                                           verbose=False)
        if verbose:
            print("fitting genesis density ...")
        self.genesis = GenesisModel(cfg).fit(int_pts)

        # These counts are the fallback used when no --counts-csv is given, and
        # they have to be counted on the same footing as the target they will
        # become. Under counts_mode="par" the generation loop treats each value
        # as a number of storms ENTERING PAR and keeps generating until it is
        # met. Handing it a basin-wide count there inflates PAR by the inverse
        # of the basin-to-PAR ratio: the WNP averages 25.3 storms a season while
        # only 16.2 of them reach PAR, so the fallback was over-filling PAR by
        # about 55% and every hazard statistic downstream inherited it.
        counted = int_pts
        scope = "basin"
        if cfg.counts_mode == "par":
            counted = int_pts[in_par(int_pts.lat.to_numpy(),
                                     int_pts.lon.to_numpy(), cfg)]
            scope = "PAR-entering"
        per_year = counted.groupby("SEASON").SID.nunique()
        self.annual_counts = per_year.to_numpy()
        if verbose:
            print(f"  historical {scope} storms/year: mean {per_year.mean():.1f}, "
                  f"range {per_year.min()}-{per_year.max()}")
        return self

    # -- generation ----------------------------------------------------
    def _advance(self, months, lat, lon, wind, sid, year, rng, gen, tx, cfg):
        """Step a cohort of storms from their seed positions to termination.

        Returns (row frames, per-storm PAR flag, per-storm land-crossing flag).
        Storms are advanced in lockstep so one model call moves the whole cohort
        by one 6-hourly step.

        This is separated from _simulate_year so that the seed can come from
        somewhere other than the genesis model. genesis_forecast.py asks what
        happens to a storm formed at one chosen point in one chosen month, and
        that question has to be answered by exactly the same propagator that
        built the catalogue, not by a second copy of the stepping loop that can
        drift away from it.
        """
        n = len(lat)
        lat = np.array(lat, dtype=float)
        lon = np.array(lon, dtype=float)
        wind = np.array(wind, dtype=float)
        months = np.asarray(months)
        alive = np.ones(n, dtype=bool)
        u = np.zeros(n); v = np.zeros(n); dv_prev = np.zeros(n)
        crossed = np.zeros(n, dtype=bool)
        seen_par = np.zeros(n, dtype=bool)
        sid = np.asarray(sid)
        hist = [wind.copy()]
        out = []

        for step in range(int(cfg.max_age_hours / cfg.step_hours)):
            if not alive.any():
                break
            idx = np.where(alive)[0]
            hbar, land, lfrac = tx.sample(lat[idx], lon[idx], with_fraction=True)
            crossed[idx] |= land
            par_now = in_par(lat[idx], lon[idx], cfg)
            seen_par[idx] |= par_now

            out.append(pd.DataFrame({
                "SID": sid[idx], "YEAR": year, "MONTH": months[idx],
                "STEP": step, "LAT": lat[idx], "LON": lon[idx],
                "WIND": wind[idx], "ELEVATION": hbar,
                "OVER_LAND": land.astype(int),
                "HAS_CROSSED_LAND": crossed[idx].astype(int),
            }))

            msin = np.sin(2 * np.pi * months[idx] / 12.0)
            mcos = np.cos(2 * np.pi * months[idx] / 12.0)
            age = step * cfg.step_hours

            tf = pd.DataFrame({
                "lat": lat[idx], "lon": lon[idx], "u_prev": u[idx],
                "v_prev": v[idx], "vmax": wind[idx],
                "month_sin": msin, "month_cos": mcos, "age_h": age,
                "is_genesis": 1.0 if step == 0 else 0.0,
            })[D.FEATURES]
            d = self.track.sample(tf, rng=rng, generator=gen)

            mpi = I.get_potential_intensity().sample(lat[idx], lon[idx], months[idx])
            # A warmer ocean raises the ceiling storms are braked
            # against. This is the only place climate change enters.
            mpi = mpi * I.mpi_warming_factor(
                year, self.year0, cfg.mpi_trend_percent_per_century)
            inten = pd.DataFrame({
                "vmax": wind[idx], "lat": lat[idx], "lon": lon[idx],
                "hbar": hbar, "over_land": land.astype(float),
                "land_frac": lfrac, "dv_prev": dv_prev[idx],
                "dv_24h": (wind[idx] - hist[-I.RI_LOOKBACK_STEPS - 1][idx]
                           if len(hist) > I.RI_LOOKBACK_STEPS
                           else np.zeros(len(idx))),
                "trans_speed": np.hypot(v[idx], u[idx] * np.cos(np.radians(lat[idx]))),
                "month_sin": msin, "month_cos": mcos, "age_h": age,
                "mpi": mpi, "v_frac": wind[idx] / np.maximum(mpi, 1.0),
            })[I.FEATURES]
            dv = self.intensity.sample(inten, generator=gen)

            lon[idx] = lon[idx] + d[:, 0]
            lat[idx] = lat[idx] + d[:, 1]
            u[idx], v[idx] = d[:, 0], d[:, 1]
            wind[idx] = np.maximum(0.0, wind[idx] + dv)
            dv_prev[idx] = dv
            hist.append(wind.copy())
            if len(hist) > I.RI_LOOKBACK_STEPS + 2:
                hist.pop(0)

            if cfg.overland_cap_kt is not None:
                _, land_now = tx.sample(lat[idx], lon[idx])
                wind[idx] = np.where(land_now,
                                     np.minimum(wind[idx], cfg.overland_cap_kt),
                                     wind[idx])

            outside = ((lon[idx] < cfg.lon_min) | (lon[idx] > cfg.lon_max)
                       | (lat[idx] < cfg.lat_min) | (lat[idx] > cfg.lat_max))
            alive[idx] = (wind[idx] >= cfg.min_wind_kt) & ~outside

        return out, seen_par, crossed

    def _simulate_year(self, year, n, ensemble_id, seq0, rng, gen, tx, cfg,
                       par_target=None):
        """Generate n storms for one year and step them all to termination.

        Returns (row frames, number that entered PAR, next sequence number).
        """
        months, lat, lon, wind = self.genesis.sample(int(n), rng)
        sid = np.array([f"SYN_{ensemble_id:02d}_{year}_{seq0 + i:04d}"
                        for i in range(int(n))])
        out, seen_par, _ = self._advance(months, lat, lon, wind, sid, year,
                                         rng, gen, tx, cfg)

        # In PAR mode, keep only as many PAR-entering storms as still needed, so
        # the final count matches the projection rather than overshooting.
        if par_target is not None:
            keep_par = np.where(seen_par)[0][:par_target]
            keep = set(sid[keep_par].tolist())
            out = [f[f.SID.isin(keep)] for f in out]
            out = [f for f in out if len(f)]
            return out, len(keep_par), seq0 + int(n)
        return out, int(seen_par.sum()), seq0 + int(n)

    def simulate_from_genesis(self, lat0, lon0, month, n=1000, wind0=None,
                              year=None, seed=0, dtm=None):
        """Answer the tool's question: a storm forms here, in this month, where
        does it go?

        Draws n independent realisations from the same trained propagator that
        built the catalogue and returns one long frame of their tracks. If
        wind0 is None the starting intensity is drawn from the genesis model's
        own distribution for that month, so the seed is climatologically
        consistent rather than an arbitrary choice.

        year controls only the potential-intensity warming factor; it defaults
        to the first simulated year, meaning no warming applied.
        """
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        gen = torch.Generator().manual_seed(seed)
        tx = terrain.get(dtm or cfg.dtm)
        self.year0 = int(year or getattr(self, "year0", None) or 2026)

        if wind0 is None:
            _, _, _, w = self.genesis.sample(int(n), rng)
            wind0 = w
        wind = np.full(int(n), float(wind0)) if np.isscalar(wind0) else np.asarray(wind0, float)
        months = np.full(int(n), int(month))
        lat = np.full(int(n), float(lat0))
        lon = np.full(int(n), float(lon0))
        sid = np.array([f"GEN_{i:05d}" for i in range(int(n))])

        out, seen_par, crossed = self._advance(
            months, lat, lon, wind, sid, self.year0, rng, gen, tx, cfg)
        if not out:
            return pd.DataFrame(), seen_par, crossed
        df = pd.concat(out, ignore_index=True)
        df["CATEGORY"] = category_of(df.WIND.to_numpy(), cfg)
        df["IN_PAR"] = in_par(df.LAT.to_numpy(), df.LON.to_numpy(), cfg).astype(int)
        return df, seen_par, crossed


    COUNT_COLS = ("storm_count", "count", "storms", "n", "forecast")
    YEAR_COLS = ("year", "season")

    def _counts_for(self, years, rng, counts_csv=None):
        """Storms per year: from a SARIMAX projection if given, else resampled
        from the historical record."""
        if not counts_csv:
            return rng.choice(self.annual_counts, size=len(years))

        tab = pd.read_csv(counts_csv)
        lower = {c.lower(): c for c in tab.columns}
        col = next((lower[k] for k in self.COUNT_COLS if k in lower), None)
        ycol = next((lower[k] for k in self.YEAR_COLS if k in lower), None)
        if col is None or ycol is None:
            raise SystemExit(
                f"{counts_csv}: need a year column ({'/'.join(self.YEAR_COLS)}) "
                f"and a count column ({'/'.join(self.COUNT_COLS)}). "
                f"Found: {list(tab.columns)}")
        lut = dict(zip(tab[ycol].astype(int), tab[col].astype(int)))
        missing = [y for y in years if y not in lut]
        if missing:
            print(f"  note: {len(missing)} of {len(years)} years absent from "
                  f"{os.path.basename(counts_csv)}, using the file's median")
        med = int(np.median(list(lut.values())))
        return np.array([lut.get(y, med) for y in years])

    def generate(self, year_start, year_end, ensemble_id=1, counts_csv=None,
                 verbose=True):
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed + ensemble_id)
        gen = torch.Generator().manual_seed(cfg.seed + ensemble_id)
        tx = terrain.get(cfg.dtm)

        years = np.arange(year_start, year_end + 1)
        self.year0 = int(year_start)
        counts = self._counts_for(years, rng, counts_csv)
        if verbose and cfg.mpi_trend_percent_per_century:
            f0 = I.mpi_warming_factor(year_start, year_start,
                                      cfg.mpi_trend_percent_per_century)
            f1 = I.mpi_warming_factor(year_end, year_start,
                                      cfg.mpi_trend_percent_per_century)
            print(f"  potential intensity scaled {100*(f0-1):+.1f}% to "
                  f"{100*(f1-1):+.1f}% over {year_start}-{year_end}")

        rows = []
        for year, n in zip(years, counts):
            if n <= 0:
                continue
            # A SARIMAX projection counts storms ENTERING PAR, not storms formed
            # anywhere in the basin. Only about 62% of the storms this genesis
            # model produces ever cross into the PAR hexagon, so feeding the
            # projection in directly would under-fill PAR by roughly a third.
            #
            # Rather than divide by a fixed fraction, generate in rounds until
            # the PAR-entering count is met. That is exact, self-calibrating,
            # and adds no constant to tune.
            target = int(n)
            got, attempt, seq = 0, 0, 0
            while got < target and attempt < cfg.max_genesis_rounds:
                need = target - got
                # first round asks for the target directly, later rounds scale
                # by the entry rate actually observed so far
                factor = 1.0 if attempt == 0 else max(1.2, target / max(got, 1))
                batch = max(1, int(np.ceil(need * factor)))
                part, entered, seq = self._simulate_year(
                    year, batch, ensemble_id, seq, rng, gen, tx, cfg,
                    par_target=(target - got) if cfg.counts_mode == "par" else None)
                rows.extend(part)
                got += entered if cfg.counts_mode == "par" else batch
                attempt += 1
            if verbose:
                print(f"  {year}: {got} storms"
                      f"{' entering PAR' if cfg.counts_mode == 'par' else ''}")

        df = pd.concat(rows, ignore_index=True)
        return self._finalise(df, ensemble_id)

    def _finalise(self, df, ensemble_id):
        cfg = self.cfg
        df["CATEGORY"] = category_of(df.WIND.to_numpy(), cfg)
        df["PRES"] = central_pressure(df.WIND.to_numpy(), df.LAT.to_numpy(), cfg)
        df["RMW"] = radius_max_wind(df.WIND.to_numpy(), df.LAT.to_numpy())
        df["IN_PAR"] = in_par(df.LAT.to_numpy(), df.LON.to_numpy(), cfg).astype(int)
        df["ENSEMBLE"] = ensemble_id
        base = pd.Timestamp("2026-01-01")
        df["ISO_TIME"] = (
            base + pd.to_timedelta(df.STEP * cfg.step_hours, unit="h")
            + pd.to_timedelta((df.YEAR - df.YEAR.min()) * 365.25, unit="D")
        )
        for thr in cfg.wind_radii_kt:
            df[f"R{thr}"] = wind_radius(thr, df.WIND.to_numpy(),
                                        df.RMW.to_numpy(), df.LAT.to_numpy())
        cols = ["SID", "ENSEMBLE", "YEAR", "MONTH", "STEP", "ISO_TIME",
                "LAT", "LON", "WIND", "PRES", "RMW", "CATEGORY",
                "ELEVATION", "OVER_LAND", "HAS_CROSSED_LAND", "IN_PAR"]
        cols += [f"R{t}" for t in cfg.wind_radii_kt]
        return df[cols]


# ===========================================================================
MODEL_FORMAT = 1


def save_model(model, path):
    """Persist a fitted SyntcAI so nothing has to be refitted.

    Fitting takes about ten minutes, which is fine once per catalogue and
    unacceptable for a tool that answers one question at a time. Saving here
    also means the tool and the paper provably share a single fitted model
    rather than two fits that merely used the same seed.

    Torch modules, numpy arrays and the potential-intensity lookup pickle
    directly. The per-month genesis KDEs do not: scipy stores the bandwidth as
    a closure inside gaussian_kde.set_bandwidth, and a lambda cannot be
    pickled. Each KDE is therefore reduced to its dataset and bandwidth factor
    and rebuilt on load, which reproduces it exactly because those two values
    are all a Gaussian KDE is.

    The module-level potential-intensity singleton is stored explicitly. It
    lives outside the model object, and a restored model that could not see it
    would fail at the first intensity step.

    Library versions are recorded so a mismatch is reported rather than
    producing a silently wrong unpickle.
    """
    import pickle
    import sys
    import scipy
    genesis_kde = {m: (k.dataset, float(k.factor))
                   for m, k in model.genesis.kde.items()}
    stashed, model.genesis.kde = model.genesis.kde, {}
    payload = {
        "format": MODEL_FORMAT,
        "genesis_kde": genesis_kde,
        "versions": {"python": sys.version.split()[0],
                     "torch": torch.__version__,
                     "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(model.cfg).items()},
        "model": model,
        "potential_intensity": I.get_potential_intensity(),
    }
    try:
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        model.genesis.kde = stashed
    return path


def _make_unpickler(pickle_mod):
    class _Unpickler(pickle_mod.Unpickler):
        """Resolve classes that were pickled under __main__.

        Running "python syntc_ai.py" makes this module __main__, so SyntcAI,
        Config and GenesisModel get recorded with that module name. Any other
        program then fails to load the model with a confusing AttributeError
        about a class that plainly exists. Rewriting __main__ to this module
        fixes it, and the guard means nothing is rewritten when the loader is
        itself the script that saved it.
        """

        def find_class(self, module, name):
            if module == "__main__" and __name__ != "__main__":
                module = __name__
            return super().find_class(module, name)

    return _Unpickler


def load_model(path, strict=False):
    """Restore a fitted SyntcAI, including the potential-intensity singleton."""
    import pickle
    import sys
    import scipy
    with open(path, "rb") as fh:
        payload = _make_unpickler(pickle)(fh).load()
    if payload.get("format") != MODEL_FORMAT:
        raise SystemExit(
            f"{path}: model format {payload.get('format')} but this code reads "
            f"{MODEL_FORMAT}. Regenerate it with --save-model.")
    now = {"python": sys.version.split()[0], "torch": torch.__version__,
           "numpy": np.__version__, "scipy": scipy.__version__}
    drift = {k: (v, now[k]) for k, v in payload["versions"].items()
             if now.get(k) != v}
    if drift:
        msg = ", ".join(f"{k} {a} -> {b}" for k, (a, b) in drift.items())
        if strict:
            raise SystemExit(f"{path}: library versions changed ({msg}).")
        print(f"  note: model was saved under {msg}; loading anyway")
    I.set_potential_intensity(payload["potential_intensity"])
    m = payload["model"]
    m.genesis.kde = {mo: gaussian_kde(data, bw_method=f)
                     for mo, (data, f) in payload["genesis_kde"].items()}
    return m


def main():
    ap = argparse.ArgumentParser(description="SYNTC-AI storm generator")
    ap.add_argument("--years", nargs=2, type=int, default=[2026, 2125])
    ap.add_argument("--ensembles", type=int, default=1)
    ap.add_argument("--out", default="syntc_ai_out")
    ap.add_argument("--ibtracs", required=True)
    ap.add_argument("--dtm", required=True)
    ap.add_argument("--counts-csv", default=None,
                    help="CSV with year and storm-count columns from your "
                         "SARIMAX projection")
    ap.add_argument("--mpi-trend", type=float, default=None,
                    help="percent increase in maximum potential intensity per "
                         "century (default 4.0, the low end of Knutson et al. "
                         "2020; use 0 for a stationary-climate catalogue)")
    ap.add_argument("--counts-mode", choices=("par", "basin"), default="par",
                    help="whether --counts-csv numbers are storms entering PAR "
                         "(default) or storms formed anywhere in the domain")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--save-model", default=None,
                    help="where to write the fitted model; defaults to "
                         "<out>/model.pkl, or the word none to skip")
    ap.add_argument("--cap-overland", type=float, default=None,
                    help="force an overland wind ceiling in kt "
                         "(imposes the result instead of predicting it)")
    a = ap.parse_args()

    cfg = CONFIG
    if a.ibtracs:
        cfg.ibtracs = a.ibtracs
    if a.dtm:
        cfg.dtm = a.dtm
    if a.epochs:
        cfg.epochs = a.epochs
    cfg.counts_mode = a.counts_mode
    if a.mpi_trend is not None:
        cfg.mpi_trend_percent_per_century = a.mpi_trend
    cfg.overland_cap_kt = a.cap_overland

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(cfg).items()}, fh, indent=2, default=str)

    model = SyntcAI(cfg).fit()

    if (a.save_model or "").lower() != "none":
        mp = a.save_model or os.path.join(a.out, "model.pkl")
        save_model(model, mp)
        print(f"  fitted model saved: {mp}")
    for e in range(1, a.ensembles + 1):
        print(f"\nensemble {e}/{a.ensembles}")
        df = model.generate(a.years[0], a.years[1], ensemble_id=e,
                            counts_csv=a.counts_csv, verbose=False)
        path = os.path.join(a.out, f"synthetic_storms_ens{e:02d}.csv")
        df.to_csv(path, index=False)
        par = df[df.IN_PAR == 1]
        print(f"  {df.SID.nunique():,} storms, {len(df):,} points "
              f"({len(par):,} in PAR) -> {path}")
        print("  " + ", ".join(
            f"{k} {v}" for k, v in par.CATEGORY.value_counts().items()))
        land = df[df.OVER_LAND == 1]
        if len(land):
            print(f"  max wind over Philippine land: {land.WIND.max():.0f} kt "
                  f"(observed record maximum is 110 kt)")


if __name__ == "__main__":
    main()
