from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.profiles import DefaultGTiffProfile

logger = logging.getLogger(__name__)


class IceGeoTIFFWriter:
    PROFILE = {
        "driver": "GTiff",
        "dtype": "float32",
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "interleave": "band",
        "nodata": np.nan,
    }

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, array: np.ndarray, name: str, profile_override: dict | None = None) -> Path:
        path = self.output_dir / f"{name}.tif"
        profile = {**self.PROFILE, "height": array.shape[0], "width": array.shape[1]}
        if profile_override:
            profile.update(profile_override)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array.astype(np.float32), 1)
            dst.set_band_description(1, name)
        logger.info("Wrote: %s (shape=%s)", path, array.shape)
        return path

    def write_int(self, array: np.ndarray, name: str, profile_override: dict | None = None) -> Path:
        path = self.output_dir / f"{name}.tif"
        profile = {**self.PROFILE, "dtype": "int16", "height": array.shape[0], "width": array.shape[1],
                   "nodata": -1}
        if profile_override:
            profile.update(profile_override)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array.astype(np.int16), 1)
            dst.set_band_description(1, name)
        logger.info("Wrote: %s (int)", path)
        return path
