from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.profiles import DefaultGTiffProfile
import xarray as xr
import zarr

from lunaice.models.schemas import DecompositionProducts

logger = logging.getLogger(__name__)


class GeoTIFFWriter:
    DEFAULT_PROFILE = {
        "driver": "GTiff",
        "dtype": "float32",
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "interleave": "band",
        "nodata": np.nan,
    }

    def __init__(self, output_dir: str | Path, crs: str = "EPSG:4326", transform: Optional[list] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crs = crs
        self.transform = transform

    def write_band(
        self,
        array: np.ndarray,
        name: str,
        profile_override: Optional[dict] = None,
    ) -> Path:
        path = self.output_dir / f"{name}.tif"
        if array.ndim == 3:
            n_bands = array.shape[0]
        else:
            n_bands = 1
            array = array[np.newaxis, :, :]
        profile = {**self.DEFAULT_PROFILE, "count": n_bands, "height": array.shape[1], "width": array.shape[2]}
        if profile_override:
            profile.update(profile_override)
        if self.crs:
            profile["crs"] = self.crs
        with rasterio.open(path, "w", **profile) as dst:
            for i in range(n_bands):
                dst.write(array[i].astype(np.float32), i + 1)
                dst.set_band_description(i + 1, f"{name}_b{i+1}" if n_bands > 1 else name)
        logger.info("Wrote GeoTIFF: %s (shape=%s)", path, array.shape)
        return path

    def write_multiband(
        self,
        bands: dict[str, np.ndarray],
        name: str,
    ) -> Path:
        first = next(iter(bands.values()))
        h, w = first.shape
        count = len(bands)
        profile = {**self.DEFAULT_PROFILE, "count": count, "height": h, "width": w}
        if self.crs:
            profile["crs"] = self.crs
        path = self.output_dir / f"{name}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            for i, (band_name, arr) in enumerate(bands.items(), 1):
                dst.write(arr.astype(np.float32), i)
                dst.set_band_description(i, band_name)
        logger.info("Wrote multi-band GeoTIFF: %s (%d bands)", path, count)
        return path


class ZarrWriter:
    def __init__(self, output_dir: str | Path, chunks: tuple[int, int] = (256, 256), compressor: str = "blosc"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = chunks
        self.compressor = compressor

    def write_product_cube(self, products: DecompositionProducts, name: str = "polarimetric_cube") -> Path:
        data_vars = {}
        for field_name in [
            "entropy", "alpha_deg", "anisotropy",
            "cpr", "dop", "span",
            "sigma_hh", "sigma_hv", "sigma_vv",
            "gamma_hh", "gamma_hv", "gamma_vv",
            "odd_bounce", "double_bounce", "volume_scattering",
        ]:
            arr = getattr(products, field_name, None)
            if arr is not None:
                data_vars[field_name] = (("y", "x"), arr)
        if not data_vars:
            raise ValueError("No valid products to write")
        ds = xr.Dataset(data_vars, attrs={"description": "DFSAR polarimetric feature cube"})
        path = self.output_dir / f"{name}.zarr"
        ds.to_zarr(str(path), mode="w", encoding={k: {"compressor": zarr.Blosc(cname=self.compressor)} for k in data_vars})
        logger.info("Wrote Zarr cube: %s (%d variables)", path, len(data_vars))
        return path


class ReportWriter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_summary(self, summary: dict, name: str = "processing_summary.json") -> Path:
        path = self.output_dir / name
        serializable = {}
        for k, v in summary.items():
            if isinstance(v, np.generic):
                serializable[k] = v.item()
            elif isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                serializable[k] = v.item()
            else:
                serializable[k] = v
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info("Wrote summary: %s", path)
        return path

    def write_statistics(self, products: DecompositionProducts, name: str = "statistics.json") -> Path:
        stats = {}
        for field_name in [
            "entropy", "alpha_deg", "anisotropy",
            "cpr", "dop", "span",
            "sigma_hh", "sigma_hv", "sigma_vv",
            "odd_bounce", "double_bounce", "volume_scattering",
        ]:
            arr = getattr(products, field_name, None)
            if arr is not None:
                valid = arr[~np.isnan(arr)]
                if valid.size > 0:
                    stats[field_name] = {
                        "min": float(valid.min()),
                        "max": float(valid.max()),
                        "mean": float(valid.mean()),
                        "std": float(valid.std()),
                        "p5": float(np.percentile(valid, 5)),
                        "p25": float(np.percentile(valid, 25)),
                        "p50": float(np.percentile(valid, 50)),
                        "p75": float(np.percentile(valid, 75)),
                        "p95": float(np.percentile(valid, 95)),
                    }
        return self.write_summary(stats, name)
