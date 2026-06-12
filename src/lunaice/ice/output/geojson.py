from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)


class CandidateGeoJSONWriter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, gdf: gpd.GeoDataFrame, name: str = "candidate_regions") -> Path:
        if gdf.empty:
            logger.warning("No candidate regions to write")
            return self.output_dir / f"{name}.geojson"
        path = self.output_dir / f"{name}.geojson"
        gdf.to_file(path, driver="GeoJSON")
        logger.info("Wrote %d candidate polygons to %s", len(gdf), path)
        return path
