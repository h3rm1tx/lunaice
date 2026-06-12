from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IndicatorResult:
    name: str
    score: np.ndarray
    raw_value: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        assert self.score.ndim == 2, f"Indicator score must be 2D, got shape {self.score.shape}"


class BaseIndicator(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, **kwargs) -> IndicatorResult:
        ...

    def normalize(self, arr: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
        clipped = np.clip(arr, vmin, vmax)
        if vmax - vmin == 0:
            return np.zeros_like(arr)
        return (clipped - vmin) / (vmax - vmin)
