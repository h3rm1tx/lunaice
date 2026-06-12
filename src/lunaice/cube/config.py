from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from lunaice.config import ProcessingConfig


@dataclass
class GridConfig:
    projection: str = "polar_stereographic"
    crs_proj4: str = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs"
    crs_epsg: str = "MOON_ME"
    pixel_size_m: float = 20.0
    bounds_left: float = -50000.0
    bounds_right: float = 50000.0
    bounds_bottom: float = -50000.0
    bounds_top: float = 50000.0
    reference_body_radius: float = 1737400.0

    @property
    def width(self) -> int:
        return int((self.bounds_right - self.bounds_left) / self.pixel_size_m)

    @property
    def height(self) -> int:
        return int((self.bounds_top - self.bounds_bottom) / self.pixel_size_m)


@dataclass
class SourceConfig:
    dfsar_path: Optional[str] = None
    ohrc_path: Optional[str] = None
    lola_dem_path: Optional[str] = None
    lola_slope_path: Optional[str] = None
    illumination_dir: Optional[str] = None
    crater_catalog: Optional[str] = None

    def validate(self) -> list[str]:
        missing = []
        for name, path in [
            ("dfsar_path", self.dfsar_path),
            ("ohrc_path", self.ohrc_path),
            ("lola_dem_path", self.lola_dem_path),
        ]:
            if path and not Path(path).exists():
                missing.append(f"{name}: {path}")
        return missing


@dataclass
class CubeConfig:
    cube_path: str = "polaris_cube.zarr"
    grid: GridConfig = field(default_factory=GridConfig)
    sources: SourceConfig = field(default_factory=SourceConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    chunk_size: int = 256
    dask_chunks: dict[str, int] = field(default_factory=lambda: {"y": 256, "x": 256})
    pyramid_levels: int = 5
    overwrite: bool = False
    logging_level: str = "INFO"
    include_dfsar: bool = True
    include_ohrc: bool = True
    include_lola: bool = True
    include_illumination: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> CubeConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cube config not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw.get("cube", raw))

    @classmethod
    def _from_dict(cls, raw: dict) -> CubeConfig:
        grid_raw = raw.get("grid", {})
        grid = GridConfig(**grid_raw)
        sources_raw = raw.get("sources", {})
        sources = SourceConfig(**sources_raw)
        proc_raw = raw.get("processing", {})
        proc = ProcessingConfig(**proc_raw)
        return cls(
            cube_path=raw.get("cube_path", "polaris_cube.zarr"),
            grid=grid,
            sources=sources,
            processing=proc,
            chunk_size=raw.get("chunk_size", 256),
            dask_chunks=raw.get("dask_chunks", {"y": 256, "x": 256}),
            pyramid_levels=raw.get("pyramid_levels", 5),
            overwrite=raw.get("overwrite", False),
            logging_level=raw.get("logging_level", "INFO"),
            include_dfsar=raw.get("include_dfsar", True),
            include_ohrc=raw.get("include_ohrc", True),
            include_lola=raw.get("include_lola", True),
            include_illumination=raw.get("include_illumination", True),
        )

    def resolve(self) -> None:
        self.cube_path = str(Path(self.cube_path).expanduser().resolve())
        if self.sources.dfsar_path:
            self.sources.dfsar_path = str(Path(self.sources.dfsar_path).expanduser().resolve())
        if self.sources.ohrc_path:
            self.sources.ohrc_path = str(Path(self.sources.ohrc_path).expanduser().resolve())
        if self.sources.lola_dem_path:
            self.sources.lola_dem_path = str(Path(self.sources.lola_dem_path).expanduser().resolve())
        if self.sources.lola_slope_path:
            self.sources.lola_slope_path = str(Path(self.sources.lola_slope_path).expanduser().resolve())
        if self.sources.illumination_dir:
            self.sources.illumination_dir = str(Path(self.sources.illumination_dir).expanduser().resolve())
