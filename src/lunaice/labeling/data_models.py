from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class IceLabel(IntEnum):
    NO_ICE = 0
    POSSIBLE = 1
    LIKELY = 2
    HIGH_CONFIDENCE = 3


LABEL_NAMES = {
    IceLabel.NO_ICE: "No Ice",
    IceLabel.POSSIBLE: "Possible Ice",
    IceLabel.LIKELY: "Likely Ice",
    IceLabel.HIGH_CONFIDENCE: "High Confidence Ice",
}


@dataclass
class LabelSource:
    name: str
    label: np.ndarray
    confidence: np.ndarray
    rules: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.label.shape != self.confidence.shape:
            raise ValueError(
                f"label shape {self.label.shape} != confidence shape {self.confidence.shape}"
            )


@dataclass
class FusedLabelResult:
    fused_label: np.ndarray
    fused_confidence: np.ndarray
    per_source_labels: dict[str, LabelSource]
    disagreement_map: Optional[np.ndarray] = None
    n_sources: int = 0
    metadata: Optional[dict] = None
