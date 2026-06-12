from __future__ import annotations

import numpy as np

from lunaice.ice.indicators.base import BaseIndicator, IndicatorResult


class RoughnessIndicator(BaseIndicator):
    def __init__(self, percentile: float = 30.0, name: str = "roughness"):
        super().__init__(name)
        self.percentile = percentile

    def compute(self, slope: np.ndarray, **kwargs) -> IndicatorResult:
        valid = slope[~np.isnan(slope)]
        if valid.size == 0:
            return IndicatorResult(name=self.name, score=np.full_like(slope, np.nan, dtype=np.float32))
        threshold = np.percentile(valid, self.percentile)
        score = np.where(
            np.isnan(slope), np.nan,
            np.clip(1.0 - slope / (threshold * 3.0), 0.0, 1.0)
        )
        return IndicatorResult(
            name=self.name,
            score=score.astype(np.float32),
            raw_value=slope.astype(np.float32),
            metadata={"slope_percentile_threshold": float(threshold)},
        )
