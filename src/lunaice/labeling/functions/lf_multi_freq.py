from __future__ import annotations

import numpy as np

from lunaice.labeling.data_models import IceLabel
from lunaice.labeling.functions.base import BaseLabelingFunction, LabelFunctionResult


class LF_MultiFrequency(BaseLabelingFunction):
    def __init__(self):
        super().__init__("multi_frequency")

    def compute(self, cpr_l: np.ndarray, cpr_s: np.ndarray, **kwargs) -> LabelFunctionResult:
        label = np.full_like(cpr_l, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        confidence = np.zeros_like(cpr_l, dtype=np.float32)

        both_high = (cpr_l > 1.0) & (cpr_s > 1.0)
        l_only = (cpr_l > 1.0) & (cpr_s <= 1.0)
        s_only = (cpr_s > 1.0) & (cpr_l <= 1.0)
        ratio = np.where((cpr_s > 0) & ~np.isnan(cpr_s), cpr_l / np.maximum(cpr_s, 1e-8), np.nan)
        l_high_freq_ratio = l_only & (ratio > 1.5)

        label[both_high] = IceLabel.HIGH_CONFIDENCE
        confidence[both_high] = 0.85 + 0.1 * np.clip((cpr_l[both_high] + cpr_s[both_high]) / 4.0, 0, 1)

        label[l_high_freq_ratio] = IceLabel.LIKELY
        confidence[l_high_freq_ratio] = 0.6

        other_l = l_only & ~l_high_freq_ratio
        label[other_l] = IceLabel.POSSIBLE
        confidence[other_l] = 0.4

        label[s_only] = IceLabel.POSSIBLE
        confidence[s_only] = 0.35

        nan_mask = np.isnan(cpr_l) | np.isnan(cpr_s)
        label[nan_mask] = IceLabel.NO_ICE
        confidence[nan_mask] = 0.0

        rules = [
            "Both L and S CPR > 1 -> HIGH_CONFIDENCE (consistent multi-frequency ice signature)",
            "L CPR > 1 with L/S ratio > 1.5 -> LIKELY (frequency-dependent scattering)",
            "L-only or S-only CPR > 1 -> POSSIBLE",
        ]
        return LabelFunctionResult(name=self.name, label=label, confidence=confidence, rules=rules)
