from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from lunaice.labeling.data_models import IceLabel


@dataclass
class LabelFunctionResult:
    name: str
    label: np.ndarray
    confidence: np.ndarray
    rules: list[str] = field(default_factory=list)
    metadata: Optional[dict] = None

    def __post_init__(self):
        assert self.label.shape == self.confidence.shape, (
            f"label shape {self.label.shape} != confidence shape {self.confidence.shape}"
        )


class BaseLabelingFunction(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, **kwargs) -> LabelFunctionResult:
        ...

    def _apply_confidence_threshold(
        self, label: np.ndarray, confidence: np.ndarray,
        min_conf: float = 0.0,
    ) -> np.ndarray:
        result = label.copy()
        result[confidence < min_conf] = IceLabel.NO_ICE
        return result
