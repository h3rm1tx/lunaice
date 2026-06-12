from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import dask.array as da
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from lunaice.cube.config import GridConfig

logger = logging.getLogger(__name__)


class IngestionSource(ABC):
    def __init__(self, grid: GridConfig, chunks: dict[str, int]):
        self.grid = grid
        self.chunks = chunks

    @abstractmethod
    def load(self) -> dict[str, da.Array]:
        ...

    def _warp_to_grid(self, src_path: str | Path, band: int = 1) -> da.Array:
        h = self.grid.height
        w = self.grid.width
        dst_transform = (
            self.grid.pixel_size_m, 0.0, self.grid.bounds_left,
            0.0, -self.grid.pixel_size_m, self.grid.bounds_top,
        )
        dst_crs = self.grid.crs_proj4
        with rasterio.open(src_path) as src:
            dest = np.zeros((h, w), dtype=np.float32)
            reproject(
                source=rasterio.band(src, band),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
        return da.from_array(dest, chunks=(self.chunks["y"], self.chunks["x"]))

    def _empty_chunked(self) -> da.Array:
        h = self.grid.height
        w = self.grid.width
        return da.full((h, w), np.nan, dtype=np.float32, chunks=(self.chunks["y"], self.chunks["x"]))


class DFSARSource(IngestionSource):
    def __init__(self, dfsar_path: Optional[str], grid: GridConfig, chunks: dict[str, int]):
        super().__init__(grid, chunks)
        self.dfsar_path = Path(dfsar_path) if dfsar_path else None

    def load(self) -> dict[str, da.Array]:
        if not self.dfsar_path or not self.dfsar_path.exists():
            logger.warning("DFSAR path not found: %s", self.dfsar_path)
            return {}
        vars = {}
        geotiffs = list(self.dfsar_path.glob("*.tif")) + list(self.dfsar_path.glob("*.tiff"))
        dfsar_vars = {
            "entropy", "alpha_deg", "anisotropy", "cpr", "dop",
            "sigma_hh", "sigma_hv", "sigma_vv",
            "gamma_hh", "gamma_hv", "gamma_vv",
            "span", "odd_bounce", "double_bounce", "volume_scattering",
        }
        found = set()
        for tif in geotiffs:
            stem = tif.stem.lower()
            if stem in dfsar_vars:
                arr = self._warp_to_grid(tif)
                vars[stem] = arr
                found.add(stem)
        if not found:
            logger.warning("No DFSAR GeoTIFFs found matching expected vars in %s", self.dfsar_path)
            zarr_path = self.dfsar_path / "polarimetric_cube.zarr"
            if zarr_path.exists():
                import xarray as xr
                ds = xr.open_zarr(str(zarr_path))
                for var_name in ds.data_vars:
                    if var_name in dfsar_vars:
                        arr = da.from_array(ds[var_name].values, chunks=(self.chunks["y"], self.chunks["x"]))
                        vars[var_name] = arr
                        found.add(var_name)
        logger.info("Loaded %d DFSAR variables: %s", len(vars), list(vars.keys()))
        return vars


class OHRCSource(IngestionSource):
    def __init__(self, ohrc_path: Optional[str], grid: GridConfig, chunks: dict[str, int]):
        super().__init__(grid, chunks)
        self.ohrc_path = Path(ohrc_path) if ohrc_path else None

    def load(self) -> dict[str, da.Array]:
        if not self.ohrc_path or not self.ohrc_path.exists():
            logger.warning("OHRC path not found: %s", self.ohrc_path)
            return {}
        vars = {}
        if self.ohrc_path.is_dir():
            tifs = list(self.ohrc_path.glob("*.tif")) + list(self.ohrc_path.glob("*.img"))
        else:
            tifs = [self.ohrc_path]
        for tif in tifs[:1]:
            arr = self._warp_to_grid(tif)
            vars["pan"] = arr
            break
        if not vars:
            vars["pan"] = self._empty_chunked()
        logger.info("Loaded OHRC panchromatic band")
        return vars


class LOLASource(IngestionSource):
    def __init__(
        self,
        dem_path: Optional[str],
        grid: GridConfig,
        chunks: dict[str, int],
        slope_path: Optional[str] = None,
    ):
        super().__init__(grid, chunks)
        self.dem_path = Path(dem_path) if dem_path else None
        self.slope_path = Path(slope_path) if slope_path else None

    def load(self) -> dict[str, da.Array]:
        vars = {}
        if self.dem_path and self.dem_path.exists():
            vars["elevation"] = self._warp_to_grid(self.dem_path)
        else:
            vars["elevation"] = self._empty_chunked()
        if self.slope_path and self.slope_path.exists():
            vars["slope"] = self._warp_to_grid(self.slope_path)
        else:
            vars["slope"] = self._empty_chunked()
        logger.info("Loaded LOLA DEM variables: %s", list(vars.keys()))
        return vars


class IlluminationSource(IngestionSource):
    def __init__(self, illum_dir: Optional[str], grid: GridConfig, chunks: dict[str, int]):
        super().__init__(grid, chunks)
        self.illum_dir = Path(illum_dir) if illum_dir else None

    def load(self) -> dict[str, da.Array]:
        if not self.illum_dir or not self.illum_dir.exists():
            logger.warning("Illumination dir not found: %s", self.illum_dir)
            return {}
        vars = {}
        tifs = list(self.illum_dir.glob("*.tif")) + list(self.illum_dir.glob("*.img"))
        for tif in tifs:
            stem = tif.stem.lower()
            if "incidence" in stem or "illum" in stem:
                vars["incidence_angle"] = self._warp_to_grid(tif)
                break
        if "incidence_angle" not in vars:
            vars["incidence_angle"] = self._empty_chunked()
        logger.info("Loaded illumination variables: %s", list(vars.keys()))
        return vars
