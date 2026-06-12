from __future__ import annotations

import numpy as np

from lunaice.labeling.data_models import IceLabel
from lunaice.labeling.functions.base import BaseLabelingFunction, LabelFunctionResult


class LF_PSR(BaseLabelingFunction):
    def __init__(self, psr_confidence: float = 0.7):
        super().__init__("psr")
        self.psr_confidence = psr_confidence

    def compute(self, psr_mask: np.ndarray, **kwargs) -> LabelFunctionResult:
        label = np.full_like(psr_mask, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        confidence = np.zeros_like(psr_mask, dtype=np.float32)

        inside_psr = psr_mask > 0
        label[inside_psr] = IceLabel.LIKELY
        confidence[inside_psr] = self.psr_confidence

        label[~inside_psr] = IceLabel.NO_ICE
        confidence[~inside_psr] = 0.1

        label[np.isnan(psr_mask)] = IceLabel.NO_ICE
        confidence[np.isnan(psr_mask)] = 0.0

        rules = [
            "Inside PSR -> LIKELY (PSRs are cold traps for volatiles)",
            "Outside PSR -> NO_ICE",
        ]
        return LabelFunctionResult(name=self.name, label=label, confidence=confidence, rules=rules)
