from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import dask.array as da
import numpy as np
import xarray as xr

from lunaice.cube.core import PolarisDataCube

logger = logging.getLogger(__name__)


@dataclass
class PixelQuery:
    x: float
    y: float
    lon: Optional[float] = None
    lat: Optional[float] = None
    variables: Optional[list[str]] = None


@dataclass
class PatchQuery:
    x_center: float
    y_center: float
    size_m: float
    variables: Optional[list[str]] = None


@dataclass
class CraterQuery:
    crater_name: str
    buffer_m: float = 0.0
    variables: Optional[list[str]] = None


class QueryAPI:
    def __init__(self, cube: PolarisDataCube):
        self.cube = cube
        self.ds = cube.ds

    def pixel(self, query: PixelQuery) -> dict[str, float]:
        ds = self.ds
        x_idx = int((query.x - float(ds.x.values[0])) / float(ds.x.values[1] - ds.x.values[0]))
        y_idx = int((float(ds.y.values[0]) - query.y) / float(ds.y.values[1] - ds.y.values[0]))
        x_idx = np.clip(x_idx, 0, ds.dims["x"] - 1)
        y_idx = np.clip(y_idx, 0, ds.dims["y"] - 1)
        result = {"x": query.x, "y": query.y, "row": int(y_idx), "col": int(x_idx)}
        vars_to_query = query.variables or list(ds.data_vars)
        for var in vars_to_query:
            if var in ds.data_vars:
                val = ds[var].values[y_idx, x_idx]
                result[var] = float(val) if not np.isnan(val) else None
        return result

    def patch(self, query: PatchQuery) -> xr.Dataset:
        ds = self.ds
        dx = float(ds.x.values[1] - ds.x.values[0])
        dy = -dx
        half = int(query.size_m / (2 * abs(dx)))
        x_center_idx = int((query.x_center - float(ds.x.values[0])) / dx)
        y_center_idx = int((float(ds.y.values[0]) - query.y_center) / abs(dy))
        x_slice = slice(max(0, x_center_idx - half), min(ds.dims["x"], x_center_idx + half))
        y_slice = slice(max(0, y_center_idx - half), min(ds.dims["y"], y_center_idx + half))
        if query.variables:
            patch_ds = ds[query.variables].isel(x=x_slice, y=y_slice)
        else:
            patch_ds = ds.isel(x=x_slice, y=y_slice)
        logger.info(
            "Patch query at (%.1f, %.1f) size %.1f m: %s",
            query.x_center, query.y_center, query.size_m,
            dict(patch_ds.dims),
        )
        return patch_ds

    def crater(self, query: CraterQuery) -> xr.Dataset:
        from lunaice.cube.craters import CraterCatalog
        cat = CraterCatalog()
        record = cat.lookup(query.crater_name)
        if record is None:
            raise KeyError(f"Crater not found: {query.crater_name}")
        buf = query.buffer_m
        x_min = record.x_m - buf
        x_max = record.x_m + buf
        y_min = record.y_m - buf
        y_max = record.y_m + buf
        ds = self.ds
        mask = (
            (ds.x >= x_min) & (ds.x <= x_max) &
            (ds.y >= y_min) & (ds.y <= y_max)
        )
        patch = ds.where(mask, drop=True)
        if query.variables:
            patch = patch[query.variables]
        logger.info(
            "Crater query '%s' (%.1f, %.1f) buffer %.0f m: %s",
            query.crater_name, record.x_m, record.y_m, query.buffer_m,
            dict(patch.dims),
        )
        return patch

    def compute(self, data: xr.Dataset | xr.DataArray) -> np.ndarray:
        return data.compute()
