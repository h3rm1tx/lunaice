from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SpeckleFilterConfig:
    method: str = "refined_lee"
    window_size: int = 7
    damping_factor: float = 1.0
    n_looks: int = 4
    sigma_sar: float = 0.0

    def __post_init__(self):
        valid = {"refined_lee", "boxcar", "lee_sigma", "idani", "bilateral"}
        if self.method not in valid:
            raise ValueError(f"Speckle filter must be one of {valid}, got {self.method}")
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError(f"window_size must be odd and >= 3, got {self.window_size}")


@dataclass
class ProcessingConfig:
    radiometric_calibration: bool = True
    polarimetric_calibration: bool = True
    speckle_filter: Optional[SpeckleFilterConfig] = None
    multilook_range: int = 1
    multilook_azimuth: int = 1
    generate_coherency_matrix: bool = True
    generate_cloude_pottier: bool = True
    generate_cpr: bool = True
    generate_dop: bool = True
    generate_backscatter: bool = True
    output_dtype: str = "float32"
    clip_percentile: float = 99.5
    zarr_compressor: str = "blosc"
    zarr_chunks: tuple[int, int] = (256, 256)
    geotiff_compress: str = "LZW"


@dataclass
class DFSARConfig:
    input_file: str = ""
    output_dir: str = "output"
    band: str = "L"
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging_level: str = "INFO"
    verbose: bool = False
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> DFSARConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> DFSARConfig:
        proc = raw.get("processing", {})
        speckle = proc.get("speckle_filter")
        speckle_cfg = None
        if speckle:
            speckle_cfg = SpeckleFilterConfig(**speckle)
        proc_cfg = ProcessingConfig(
            **{k: v for k, v in proc.items() if k != "speckle_filter"},
            speckle_filter=speckle_cfg,
        )
        return cls(
            input_file=raw.get("input_file", ""),
            output_dir=raw.get("output_dir", "output"),
            band=raw.get("band", "L"),
            processing=proc_cfg,
            logging_level=raw.get("logging_level", "INFO"),
            verbose=raw.get("verbose", False),
            overwrite=raw.get("overwrite", False),
        )

    def resolve(self) -> None:
        self.input_file = str(Path(self.input_file).expanduser().resolve())
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())
        os.makedirs(self.output_dir, exist_ok=True)
