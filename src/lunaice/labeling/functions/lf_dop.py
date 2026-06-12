from __future__ import annotations

import numpy as np

from lunaice.labeling.data_models import IceLabel
from lunaice.labeling.functions.base import BaseLabelingFunction, LabelFunctionResult


class LF_DOP(BaseLabelingFunction):
    def __init__(self, threshold_low: float = 0.2, threshold_high: float = 0.4):
        super().__init__("dop")
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def compute(self, dop_l: np.ndarray, dop_s: np.ndarray | None = None, **kwargs) -> LabelFunctionResult:
        dop_mean = np.nanmean(np.stack([dop_l, dop_s if dop_s is not None else dop_l], axis=-1), axis=-1)
        label = np.full_like(dop_mean, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        confidence = np.zeros_like(dop_mean, dtype=np.float32)

        low = dop_mean < self.threshold_low
        mid = (dop_mean >= self.threshold_low) & (dop_mean < self.threshold_high)
        high = dop_mean >= self.threshold_high

        label[low] = IceLabel.HIGH_CONFIDENCE
        confidence[low] = np.clip(1.0 - dop_mean[low] / self.threshold_low * 0.7, 0.3, 0.95)

        label[mid] = IceLabel.LIKELY
        confidence[mid] = 0.5 * (1.0 - (dop_mean[mid] - self.threshold_low) / (self.threshold_high - self.threshold_low))

        label[high] = IceLabel.NO_ICE
        confidence[high] = 0.1

        label[np.isnan(dop_mean)] = IceLabel.NO_ICE
        confidence[np.isnan(dop_mean)] = 0.0

        rules = [
            f"DOP < {self.threshold_low} -> HIGH_CONFIDENCE (low depolarization favors ice)",
            f"DOP in [{self.threshold_low}, {self.threshold_high}) -> LIKELY",
            f"DOP >= {self.threshold_high} -> NO_ICE (high depolarization)",
        ]
        return LabelFunctionResult(name=self.name, label=label, confidence=confidence, rules=rules)
