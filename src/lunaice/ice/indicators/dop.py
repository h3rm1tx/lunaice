from __future__ import annotations

import numpy as np

from lunaice.ice.indicators.base import BaseIndicator, IndicatorResult


class DOPIndicator(BaseIndicator):
    def __init__(self, threshold: float = 0.3, name: str = "dop"):
        super().__init__(name)
        self.threshold = threshold

    def compute(self, dop_l: np.ndarray, dop_s: np.ndarray, **kwargs) -> IndicatorResult:
        dop_mean = np.nanmean(np.stack([dop_l, dop_s], axis=-1), axis=-1)
        score = np.where(
            np.isnan(dop_mean), np.nan,
            np.clip(1.0 - dop_mean / self.threshold, 0.0, 1.0)
        )
        return IndicatorResult(
            name=self.name,
            score=score.astype(np.float32),
            raw_value=dop_mean.astype(np.float32),
            metadata={"threshold": self.threshold},
        )
