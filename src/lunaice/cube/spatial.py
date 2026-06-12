from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pyproj

from lunaice.cube.config import GridConfig

logger = logging.getLogger(__name__)


LUNAR_RADIUS = 1737400.0


def get_polar_stereo_crs() -> pyproj.CRS:
    return pyproj.CRS.from_proj4(
        "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
        f"+a={LUNAR_RADIUS} +b={LUNAR_RADIUS} +units=m +no_defs"
    )


def get_geographic_crs() -> pyproj.CRS:
    return pyproj.CRS.from_proj4(
        f"+proj=longlat +a={LUNAR_RADIUS} +b={LUNAR_RADIUS} +no_defs"
    )


def xy_to_lonlat(
    x: np.ndarray,
    y: np.ndarray,
    src_crs: Optional[pyproj.CRS] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if src_crs is None:
        src_crs = get_polar_stereo_crs()
    dst_crs = get_geographic_crs()
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat


def lonlat_to_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    dst_crs: Optional[pyproj.CRS] = None,
) -> tuple[np.ndarray, np.ndarray]:
    src_crs = get_geographic_crs()
    if dst_crs is None:
        dst_crs = get_polar_stereo_crs()
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def xy_to_pixel_indices(
    x: float,
    y: float,
    grid: GridConfig,
) -> tuple[int, int]:
    col = int((x - grid.bounds_left) / grid.pixel_size_m)
    row = int((grid.bounds_top - y) / grid.pixel_size_m)
    return row, col


def lonlat_to_pixel_indices(
    lon: float,
    lat: float,
    grid: GridConfig,
) -> tuple[int, int]:
    x, y = lonlat_to_xy(np.array([lon]), np.array([lat]))
    return xy_to_pixel_indices(float(x[0]), float(y[0]), grid)
