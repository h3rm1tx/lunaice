from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class IndicatorWeights:
    cpr: float = 0.25
    dop: float = 0.15
    psr: float = 0.20
    roughness: float = 0.10
    multi_frequency: float = 0.15
    scattering: float = 0.15

    def validate(self) -> None:
        total = sum([self.cpr, self.dop, self.psr, self.roughness, self.multi_frequency, self.scattering])
        if abs(total - 1.0) > 1e-6:
            import warnings
            warnings.warn(f"Indicator weights sum to {total:.4f}, expected 1.0")


@dataclass
class ScatteringThresholds:
    surface_alpha_max: float = 42.5
    volume_alpha_min: float = 37.5
    volume_alpha_max: float = 52.5
    dihedral_alpha_min: float = 47.5
    entropy_low: float = 0.5
    entropy_high: float = 0.9


@dataclass
class IcePriorConfig:
    cube_path: str = "polaris_cube.zarr"
    output_dir: str = "ice_products"
    weights: IndicatorWeights = field(default_factory=IndicatorWeights)
    thresholds: ScatteringThresholds = field(default_factory=ScatteringThresholds)
    cpr_threshold: float = 1.0
    dop_threshold: float = 0.3
    roughness_percentile: float = 30.0
    overwrite: bool = False
    logging_level: str = "INFO"
    generate_plots: bool = True
    generate_report: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> IcePriorConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Ice config not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw.get("ice_prior", raw))

    @classmethod
    def _from_dict(cls, raw: dict) -> IcePriorConfig:
        w = raw.get("weights", {})
        weights = IndicatorWeights(**w)
        t = raw.get("thresholds", {})
        thresholds = ScatteringThresholds(**t)
        return cls(
            cube_path=raw.get("cube_path", "polaris_cube.zarr"),
            output_dir=raw.get("output_dir", "ice_products"),
            weights=weights,
            thresholds=thresholds,
            cpr_threshold=raw.get("cpr_threshold", 1.0),
            dop_threshold=raw.get("dop_threshold", 0.3),
            roughness_percentile=raw.get("roughness_percentile", 30.0),
            overwrite=raw.get("overwrite", False),
            logging_level=raw.get("logging_level", "INFO"),
            generate_plots=raw.get("generate_plots", True),
            generate_report=raw.get("generate_report", True),
        )

    def resolve(self) -> None:
        self.cube_path = str(Path(self.cube_path).expanduser().resolve())
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())

    def __post_init__(self):
        self.weights.validate()
