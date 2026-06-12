from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import dask.array as da
import numpy as np
import xarray as xr
import zarr

from lunaice.cube.config import CubeConfig, GridConfig
from lunaice.cube.sources import (
    DFSARSource,
    IlluminationSource,
    LOLASource,
    OHRCSource,
)

logger = logging.getLogger(__name__)


class CubeBuilder:
    def __init__(self, config: CubeConfig):
        self.config = config
        self.config.resolve()
        self._data_vars: dict[str, xr.DataArray] = {}
        self._coords: dict[str, np.ndarray | xr.DataArray] = {}

    def build(self) -> str:
        t0 = time.time()
        cube_path = Path(self.config.cube_path)
        if cube_path.exists() and not self.config.overwrite:
            raise FileExistsError(
                f"Cube already exists at {cube_path}. Use --overwrite to replace."
            )
        if cube_path.exists():
            import shutil
            shutil.rmtree(cube_path)

        logger.info("Building POLARIS cube at %s", cube_path)

        grid = self.config.grid
        y = np.arange(grid.bounds_top, grid.bounds_bottom, -grid.pixel_size_m, dtype=np.float64)
        x = np.arange(grid.bounds_left, grid.bounds_right, grid.pixel_size_m, dtype=np.float64)
        y = y[:grid.height]
        x = x[:grid.width]

        chunks = {"y": self.config.chunk_size, "x": self.config.chunk_size}

        self._coords["y"] = y
        self._coords["x"] = x

        if self.config.include_dfsar and self.config.sources.dfsar_path:
            self._ingest_dfsar(grid, chunks)
        if self.config.include_ohrc and self.config.sources.ohrc_path:
            self._ingest_ohrc(grid, chunks)
        if self.config.include_lola and self.config.sources.lola_dem_path:
            self._ingest_lola(grid, chunks)
        if self.config.include_illumination and self.config.sources.illumination_dir:
            self._ingest_illumination(grid, chunks)

        if not self._data_vars:
            logger.warning("No data sources configured; building empty cube shell")

        ds = xr.Dataset(
            self._data_vars,
            coords=self._coords,
            attrs={
                "title": "POLARIS Lunar Data Cube",
                "institution": "LUNAICE Project",
                "crs_proj4": grid.crs_proj4,
                "crs_epsg": grid.crs_epsg,
                "pixel_size_m": grid.pixel_size_m,
                "bounds_left": grid.bounds_left,
                "bounds_right": grid.bounds_right,
                "bounds_bottom": grid.bounds_bottom,
                "bounds_top": grid.bounds_top,
                "reference_body_radius": grid.reference_body_radius,
                "history": f"Built by POLARIS CubeBuilder",
                "build_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        encoding = {}
        for var_name in ds.data_vars:
            encoding[var_name] = {
                "compressor": zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
                "chunks": (chunks["y"], chunks["x"]),
            }

        ds.to_zarr(str(cube_path), mode="w", encoding=encoding, consolidated=True)
        elapsed = time.time() - t0
        logger.info("Cube built in %.1f s: %s (%d vars)", elapsed, cube_path, len(ds.data_vars))
        return str(cube_path)

    def _ingest_dfsar(self, grid: GridConfig, chunks: dict[str, int]) -> None:
        logger.info("Ingesting DFSAR data from %s", self.config.sources.dfsar_path)
        source = DFSARSource(self.config.sources.dfsar_path, grid, chunks)
        for name, arr in source.load().items():
            self._data_vars[name] = arr
            logger.debug("  Added DFSAR variable: %s", name)

    def _ingest_ohrc(self, grid: GridConfig, chunks: dict[str, int]) -> None:
        logger.info("Ingesting OHRC data from %s", self.config.sources.ohrc_path)
        source = OHRCSource(self.config.sources.ohrc_path, grid, chunks)
        for name, arr in source.load().items():
            self._data_vars[name] = arr
            logger.debug("  Added OHRC variable: %s", name)

    def _ingest_lola(self, grid: GridConfig, chunks: dict[str, int]) -> None:
        logger.info("Ingesting LOLA DEM from %s", self.config.sources.lola_dem_path)
        source = LOLASource(
            dem_path=self.config.sources.lola_dem_path,
            slope_path=self.config.sources.lola_slope_path,
            grid=grid,
            chunks=chunks,
        )
        for name, arr in source.load().items():
            self._data_vars[name] = arr
            logger.debug("  Added LOLA variable: %s", name)

    def _ingest_illumination(self, grid: GridConfig, chunks: dict[str, int]) -> None:
        logger.info("Ingesting illumination data from %s", self.config.sources.illumination_dir)
        source = IlluminationSource(self.config.sources.illumination_dir, grid, chunks)
        for name, arr in source.load().items():
            self._data_vars[name] = arr
            logger.debug("  Added illumination variable: %s", name)
