from __future__ import annotations

import numpy as np

from lunaice.ice.indicators.base import BaseIndicator, IndicatorResult


class MultiFrequencyIndicator(BaseIndicator):
    def __init__(self, name: str = "multi_frequency"):
        super().__init__(name)

    def compute(
        self,
        cpr_l: np.ndarray,
        cpr_s: np.ndarray,
        sigma_hh_l: np.ndarray,
        sigma_hh_s: np.ndarray,
        **kwargs,
    ) -> IndicatorResult:
        cpr_ratio = np.where(
            (cpr_s > 0) & (~np.isnan(cpr_s)) & (~np.isnan(cpr_l)),
            np.minimum(cpr_l / (cpr_s + 1e-10), 5.0),
            np.nan,
        )
        consistent_high = (cpr_l > 1.0) & (cpr_s > 1.0)
        freq_advantage = (
            (cpr_l > 1.0) & (cpr_s <= 1.0) & (cpr_l / (cpr_s + 1e-10) > 1.5)
        )
        score = np.full(cpr_l.shape, 0.0, dtype=np.float32)
        score[consistent_high] = 1.0
        score[freq_advantage] = 0.7
        inconsistent = ((cpr_l > 1.0) & (cpr_s <= 1.0) & ~freq_advantage) | \
                       ((cpr_l <= 1.0) & (cpr_s > 1.0))
        score[inconsistent] = 0.3
        score[np.isnan(cpr_l) & np.isnan(cpr_s)] = np.nan
        return IndicatorResult(
            name=self.name,
            score=score,
            raw_value=cpr_ratio.astype(np.float32),
            metadata={"description": "Multi-frequency CPR consistency"},
        )
