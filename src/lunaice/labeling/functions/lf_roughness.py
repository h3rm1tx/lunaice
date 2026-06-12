from __future__ import annotations

import numpy as np

from lunaice.labeling.data_models import IceLabel
from lunaice.labeling.functions.base import BaseLabelingFunction, LabelFunctionResult


class LF_Roughness(BaseLabelingFunction):
    def __init__(self, threshold_low: float = 2.0, threshold_high: float = 5.0):
        super().__init__("roughness")
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def compute(self, slope: np.ndarray, **kwargs) -> LabelFunctionResult:
        label = np.full_like(slope, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        confidence = np.zeros_like(slope, dtype=np.float32)

        smooth = slope < self.threshold_low
        moderate = (slope >= self.threshold_low) & (slope < self.threshold_high)
        rough = slope >= self.threshold_high

        label[smooth] = IceLabel.LIKELY
        confidence[smooth] = 0.6 * (1.0 - slope[smooth] / self.threshold_low * 0.5)

        label[moderate] = IceLabel.POSSIBLE
        confidence[moderate] = 0.3 * (1.0 - (slope[moderate] - self.threshold_low) / (self.threshold_high - self.threshold_low))

        label[rough] = IceLabel.NO_ICE
        confidence[rough] = 0.1

        label[np.isnan(slope)] = IceLabel.NO_ICE
        confidence[np.isnan(slope)] = 0.0

        rules = [
            f"Slope < {self.threshold_low} deg -> LIKELY (smooth terrain favors ice preservation)",
            f"Slope in [{self.threshold_low}, {self.threshold_high}) deg -> POSSIBLE",
            f"Slope >= {self.threshold_high} deg -> NO_ICE (rough terrain, unstable for ice)",
        ]
        return LabelFunctionResult(name=self.name, label=label, confidence=confidence, rules=rules)
