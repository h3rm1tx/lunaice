from __future__ import annotations

import logging

import dask.array as da
import numpy as np
import xarray as xr

from lunaice.cube.config import GridConfig
from lunaice.cube.core import PolarisDataCube

logger = logging.getLogger(__name__)


class PyramidBuilder:
    def __init__(self, cube: PolarisDataCube):
        self.cube = cube
        self.ds = cube.ds
        self.config = cube.config

    def build(self, levels: int = 5) -> list[str]:
        paths = []
        base_path = self.config.cube_path
        base_res = self.config.grid.pixel_size_m
        for level in range(1, levels + 1):
            factor = 2 ** level
            res = base_res * factor
            h = self.config.grid.height // factor
            w = self.config.grid.width // factor
            if h < 1 or w < 1:
                break
            level_path = base_path.replace(".zarr", f"_pyramid_{int(res)}m.zarr")
            downsampled = {}
            for var_name in self.ds.data_vars:
                arr = self.ds[var_name].data
                if hasattr(arr, "rechunk"):
                    arr = arr.rechunk({d: -1 for d in arr.chunks.keys()})
                down = self._downsample(arr, factor)
                downsampled[var_name] = xr.DataArray(
                    down,
                    dims=("y", "x"),
                    attrs=self.ds[var_name].attrs,
                )
            coarsened_y = self.ds.y.values[::factor][:h]
            coarsened_x = self.ds.x.values[::factor][:w]
            level_ds = xr.Dataset(
                downsampled,
                coords={"y": coarsened_y, "x": coarsened_x},
                attrs={**self.ds.attrs, "pyramid_level": level, "resolution_m": res},
            )
            encoding = {}
            for vn in level_ds.data_vars:
                encoding[vn] = {"compressor": "zstd", "chunks": (256, 256)}
            level_ds.to_zarr(level_path, mode="w", encoding=encoding, consolidated=True)
            paths.append(level_path)
            logger.info("Pyramid level %d: %d x %d (%.0f m) -> %s", level, w, h, res, level_path)
        return paths

    def _downsample(self, arr: da.Array, factor: int) -> da.Array:
        h, w = arr.shape
        h_trim = h // factor * factor
        w_trim = w // factor * factor
        arr = arr[:h_trim, :w_trim]
        return arr.reshape(h_trim // factor, factor, w_trim // factor, factor).mean(axis=(1, 3))
