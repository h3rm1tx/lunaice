from __future__ import annotations

import numpy as np

from lunaice.ice.indicators.base import BaseIndicator, IndicatorResult


class PSRIndicator(BaseIndicator):
    def __init__(self, name: str = "psr"):
        super().__init__(name)

    def compute(self, psr_mask: np.ndarray, **kwargs) -> IndicatorResult:
        score = np.where(np.isnan(psr_mask), np.nan, psr_mask.astype(np.float32))
        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=score.copy(),
            metadata={"description": "Binary PSR mask (1=shadow, 0=illuminated)"},
        )
