"""
Terrain exposure for every track point: mean elevation within a 75 km
storm-centred footprint, plus a land flag.

The 75 km footprint and the land threshold follow Zerrudo and Servando (R1),
which found the footprint radius by calibration and showed the result is
insensitive to DTM resolution between 20 m and 1 km, so the 1 km Philippine
DTM is sufficient here.

The footprint mean is precomputed once by convolving the whole DTM with a
circular kernel, then sampled per track point.  Doing it per point instead
would repeat the same disc average millions of times.
"""

import os

import numpy as np
import rasterio
from scipy.signal import fftconvolve

DTM_PATH = "/mnt/user-data/uploads/SYNTC/SYNTC-main/dtm_phil_1km.tif"

FOOTPRINT_KM = 75.0
KM_PER_DEG_LAT = 111.32
LAND_THRESHOLD_M = 0.5      # same land test SynTC uses on this DTM


class TerrainExposure:
    def __init__(self, path=None, footprint_km=FOOTPRINT_KM):
        path = path or DTM_PATH
        src = rasterio.open(path)
        self.transform = src.transform
        self.width, self.height = src.width, src.height
        self.left, self.bottom, self.right, self.top = src.bounds

        elev = src.read(1).astype(np.float64)
        # Nodata marks sea or outside coverage. Both contribute zero terrain,
        # which is what "exposure" should mean: a storm half over water feels
        # half the mean elevation of one fully inland.
        elev[~np.isfinite(elev)] = 0.0
        elev[elev < -1000] = 0.0
        elev[elev < 0] = 0.0
        self.elev = elev
        self.is_land = elev > LAND_THRESHOLD_M

        res_lat = abs(self.transform.e)
        res_lon = abs(self.transform.a)
        mid_lat = 0.5 * (self.bottom + self.top)

        # Radius in cells, accounting for longitude convergence at this
        # latitude. Fixed at the DTM's mid-latitude: across 4.5N to 21N the
        # cos factor varies by about 4%, far below the sensitivity the
        # calibration showed.
        ry = footprint_km / KM_PER_DEG_LAT / res_lat
        rx = footprint_km / (KM_PER_DEG_LAT * np.cos(np.radians(mid_lat))) / res_lon
        ny, nx = int(np.ceil(ry)), int(np.ceil(rx))
        yy, xx = np.mgrid[-ny:ny + 1, -nx:nx + 1]
        kernel = ((yy / ry) ** 2 + (xx / rx) ** 2) <= 1.0
        kernel = kernel.astype(np.float64)

        # Mean over the disc, including the sea cells inside it.
        total = fftconvolve(elev, kernel, mode="same")
        self.hbar = np.clip(total / kernel.sum(), 0.0, None)

        # Fraction of the same disc that is land. This is what lets decay
        # engage on the coastal approach: a storm whose circulation already
        # covers Luzon is losing energy to the terrain before its centre
        # crosses the coastline, and a centre-only land test cannot see that.
        self.land_frac = np.clip(
            fftconvolve(self.is_land.astype(np.float64), kernel, mode="same")
            / kernel.sum(), 0.0, 1.0)

    def _index(self, lat, lon):
        col = ((np.asarray(lon) - self.left) / self.transform.a).astype(int)
        row = ((np.asarray(lat) - self.top) / self.transform.e).astype(int)
        inside = (
            (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        )
        return np.clip(row, 0, self.height - 1), np.clip(col, 0, self.width - 1), inside

    def sample(self, lat, lon, with_fraction=False):
        """Return (hbar_m, over_land) for each point, and optionally the land
        fraction of the footprint. Outside DTM coverage the storm is over open
        ocean as far as Philippine terrain is concerned."""
        row, col, inside = self._index(lat, lon)
        hbar = np.where(inside, self.hbar[row, col], 0.0)
        land = np.where(inside, self.is_land[row, col], False)
        if not with_fraction:
            return hbar, land.astype(bool)
        frac = np.where(inside, self.land_frac[row, col], 0.0)
        return hbar, land.astype(bool), frac


_CACHE = {}


def get(path=None):
    """Return the terrain model, building it once per DTM path.

    The default resolves DTM_PATH at call time, not at import time. Binding it
    as a default argument means a caller that sets terrain.DTM_PATH after
    import is silently ignored, which is exactly how a --dtm flag ends up
    having no effect.
    """
    path = path or DTM_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DTM not found: {path}\n"
            "Set terrain.DTM_PATH to your digital terrain model, or pass --dtm "
            "on the command line. The packaged default points at the author's "
            "machine and will not exist on yours."
        )
    if path not in _CACHE:
        _CACHE[path] = TerrainExposure(path)
    return _CACHE[path]
