from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LabelWeights:
    cpr: float = 0.20
    dop: float = 0.15
    psr: float = 0.20
    multi_frequency: float = 0.15
    scattering: float = 0.10
    roughness: float = 0.05
    illumination: float = 0.05
    published_regions: float = 0.10

    def validate(self) -> None:
        total = sum([self.cpr, self.dop, self.psr, self.multi_frequency,
                     self.scattering, self.roughness, self.illumination,
                     self.published_regions])
        if abs(total - 1.0) > 1e-6:
            import warnings
            warnings.warn(f"Label weights sum to {total:.4f}, expected 1.0")


@dataclass
class LabelThresholds:
    cpr_low: float = 0.8
    cpr_high: float = 1.2
    dop_low: float = 0.2
    dop_high: float = 0.4
    roughness_low: float = 2.0
    roughness_high: float = 5.0
    confidence_low: float = 0.3
    confidence_high: float = 0.7
    ice_prior_possible: float = 0.3
    ice_prior_likely: float = 0.5
    ice_prior_high: float = 0.7
    agreement_threshold: float = 0.6
    min_labeling_fraction: float = 0.3


@dataclass
class LabelingConfig:
    cube_path: str = "polaris_cube.zarr"
    ice_prior_path: str = ""
    confidence_path: str = ""
    output_dir: str = "labeling_products"
    weights: LabelWeights = field(default_factory=LabelWeights)
    thresholds: LabelThresholds = field(default_factory=LabelThresholds)
    overwrite: bool = False
    logging_level: str = "INFO"
    generate_plots: bool = True
    generate_report: bool = True
    use_snorkel_model: bool = True
    label_fusion_method: str = "weighted"
    conflict_threshold: int = 2

    @classmethod
    def from_yaml(cls, path: str | Path) -> LabelingConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Labeling config not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw.get("labeling", raw))

    @classmethod
    def _from_dict(cls, raw: dict) -> LabelingConfig:
        w = raw.get("weights", {})
        weights = LabelWeights(**w)
        t = raw.get("thresholds", {})
        thresholds = LabelThresholds(**t)
        return cls(
            cube_path=raw.get("cube_path", "polaris_cube.zarr"),
            ice_prior_path=raw.get("ice_prior_path", ""),
            confidence_path=raw.get("confidence_path", ""),
            output_dir=raw.get("output_dir", "labeling_products"),
            weights=weights,
            thresholds=thresholds,
            overwrite=raw.get("overwrite", False),
            logging_level=raw.get("logging_level", "INFO"),
            generate_plots=raw.get("generate_plots", True),
            generate_report=raw.get("generate_report", True),
            use_snorkel_model=raw.get("use_snorkel_model", True),
            label_fusion_method=raw.get("label_fusion_method", "weighted"),
            conflict_threshold=raw.get("conflict_threshold", 2),
        )

    def resolve(self) -> None:
        self.cube_path = str(Path(self.cube_path).expanduser().resolve())
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())

    def __post_init__(self):
        self.weights.validate()
