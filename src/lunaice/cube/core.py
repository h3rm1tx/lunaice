from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import dask.array as da
import numpy as np
import xarray as xr
import zarr

from lunaice.cube.config import CubeConfig

logger = logging.getLogger(__name__)


class PolarisDataCube:
    def __init__(self, config: CubeConfig):
        self.config = config
        self.config.resolve()
        self._ds: Optional[xr.Dataset] = None
        self._zarr_store: Optional[zarr.Group] = None

    @property
    def ds(self) -> xr.Dataset:
        if self._ds is None:
            self._ds = self._load()
        return self._ds

    @property
    def zarr_store(self) -> zarr.Group:
        if self._zarr_store is None:
            self._zarr_store = zarr.open_group(str(self.config.cube_path), mode="r")
        return self._zarr_store

    def _load(self) -> xr.Dataset:
        path = self.config.cube_path
        if not Path(path).exists():
            raise FileNotFoundError(f"Cube not found at {path}. Run `lunaice cube build` first.")
        chunks = self.config.dask_chunks
        ds = xr.open_zarr(path, chunks=chunks)
        logger.info("Loaded cube from %s: %s", path, list(ds.data_vars))
        return ds

    def info(self) -> dict[str, Any]:
        ds = self.ds
        info = {
            "path": self.config.cube_path,
            "shape": dict(ds.dims),
            "variables": list(ds.data_vars),
            "coords": list(ds.coords),
            "crs": ds.attrs.get("crs_proj4", "unknown"),
            "pixel_size_m": ds.attrs.get("pixel_size_m", "unknown"),
            "bounds": {
                "x_min": float(ds.x.values.min()),
                "x_max": float(ds.x.values.max()),
                "y_min": float(ds.y.values.min()),
                "y_max": float(ds.y.values.max()),
            },
            "chunks": {k: v for k, v in ds.chunks.items()} if ds.chunks else {},
            "n_bands": len(ds.data_vars),
            "total_pixels": int(np.prod(list(ds.dims.values()))),
        }
        return info

    def summary(self) -> str:
        info = self.info()
        lines = [
            f"POLARIS Data Cube: {info['path']}",
            f"  Grid:       {info['shape']['x']} x {info['shape']['y']} px",
            (f"  Variables:  {info['n_bands']} ({', '.join(info['variables'][:6])}...)"
             if len(info['variables']) > 6
             else f"  Variables:  {info['n_bands']} ({', '.join(info['variables'])})"),
            f"  CRS:        {info['crs']}",
            f"  Pixel:      {info['pixel_size_m']} m",
            f"  Bounds x:   [{info['bounds']['x_min']:.1f}, {info['bounds']['x_max']:.1f}] m",
            f"  Bounds y:   [{info['bounds']['y_min']:.1f}, {info['bounds']['y_max']:.1f}] m",
            f"  Total px:   {info['total_pixels']:,}",
        ]
        return "\n".join(lines)

    def close(self) -> None:
        self._ds = None
        self._zarr_store = None

    def __repr__(self) -> str:
        return f"<PolarisDataCube '{self.config.cube_path}'>"
