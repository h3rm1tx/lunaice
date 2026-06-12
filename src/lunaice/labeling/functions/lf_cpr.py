from __future__ import annotations

import numpy as np

from lunaice.labeling.data_models import IceLabel
from lunaice.labeling.functions.base import BaseLabelingFunction, LabelFunctionResult


class LF_CPR(BaseLabelingFunction):
    def __init__(self, threshold_low: float = 0.8, threshold_high: float = 1.2):
        super().__init__("cpr")
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def compute(self, cpr_l: np.ndarray, cpr_s: np.ndarray | None = None, **kwargs) -> LabelFunctionResult:
        cpr_mean = np.nanmean(np.stack([cpr_l, cpr_s if cpr_s is not None else cpr_l], axis=-1), axis=-1)
        label = np.full_like(cpr_mean, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        confidence = np.zeros_like(cpr_mean, dtype=np.float32)

        high = cpr_mean > self.threshold_high
        mid = (cpr_mean > self.threshold_low) & (cpr_mean <= self.threshold_high)
        low = (cpr_mean > 0.5) & (cpr_mean <= self.threshold_low)

        label[high] = IceLabel.HIGH_CONFIDENCE
        conf_high = np.clip((cpr_mean[high] - self.threshold_high) / 1.0, 0.3, 0.9)
        confidence[high] = conf_high

        label[mid] = IceLabel.LIKELY
        confidence[mid] = 0.5 + 0.3 * (cpr_mean[mid] - self.threshold_low) / (self.threshold_high - self.threshold_low)

        label[low] = IceLabel.POSSIBLE
        confidence[low] = 0.3 * (cpr_mean[low] - 0.5) / (self.threshold_low - 0.5)

        label[np.isnan(cpr_mean)] = IceLabel.NO_ICE
        confidence[np.isnan(cpr_mean)] = 0.0

        rules = [
            f"CPR > {self.threshold_high} -> HIGH_CONFIDENCE",
            f"CPR in ({self.threshold_low}, {self.threshold_high}] -> LIKELY",
            f"CPR in (0.5, {self.threshold_low}] -> POSSIBLE",
        ]
        return LabelFunctionResult(name=self.name, label=label, confidence=confidence, rules=rules)
