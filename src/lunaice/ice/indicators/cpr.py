from __future__ import annotations

import numpy as np

from lunaice.ice.indicators.base import BaseIndicator, IndicatorResult


class CPRIndicator(BaseIndicator):
    def __init__(self, threshold: float = 1.0, name: str = "cpr"):
        super().__init__(name)
        self.threshold = threshold

    def compute(self, cpr_l: np.ndarray, cpr_s: np.ndarray, **kwargs) -> IndicatorResult:
        cpr_combined = np.nanmean(np.stack([cpr_l, cpr_s], axis=-1), axis=-1)
        score = np.where(
            np.isnan(cpr_combined), np.nan,
            np.clip((cpr_combined - self.threshold) / 2.0, 0.0, 1.0)
        )
        return IndicatorResult(
            name=self.name,
            score=score.astype(np.float32),
            raw_value=cpr_combined.astype(np.float32),
            metadata={"threshold": self.threshold},
        )
