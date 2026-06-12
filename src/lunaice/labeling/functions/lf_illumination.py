from __future__ import annotations

import numpy as np

from lunaice.labeling.data_models import IceLabel
from lunaice.labeling.functions.base import BaseLabelingFunction, LabelFunctionResult


class LF_Illumination(BaseLabelingFunction):
    def __init__(self):
        super().__init__("illumination")

    def compute(self, illumination: np.ndarray | None = None, psr_mask: np.ndarray | None = None,
                **kwargs) -> LabelFunctionResult:
        if illumination is not None:
            return self._from_illumination(illumination)
        elif psr_mask is not None:
            return self._from_psr(psr_mask)
        else:
            h, w = 1, 1
            if psr_mask is not None:
                h, w = psr_mask.shape
            elif illumination is not None:
                h, w = illumination.shape
            label = np.full((h, w), fill_value=IceLabel.NO_ICE, dtype=np.int8)
            confidence = np.zeros((h, w), dtype=np.float32)
            return LabelFunctionResult(name=self.name, label=label, confidence=confidence,
                                       rules=["No illumination data available"])

    def _from_illumination(self, illum: np.ndarray) -> LabelFunctionResult:
        label = np.full_like(illum, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        confidence = np.zeros_like(illum, dtype=np.float32)

        dark = illum < 0.01
        shadow = (illum >= 0.01) & (illum < 0.1)
        lit = illum >= 0.1

        label[dark] = IceLabel.LIKELY
        confidence[dark] = 0.65

        label[shadow] = IceLabel.POSSIBLE
        confidence[shadow] = 0.35

        label[lit] = IceLabel.NO_ICE
        confidence[lit] = 0.05

        label[np.isnan(illum)] = IceLabel.NO_ICE
        confidence[np.isnan(illum)] = 0.0

        rules = [
            "Illumination < 1% -> LIKELY (persistent shadow favors ice stability)",
            "Illumination 1-10% -> POSSIBLE",
            "Illumination >= 10% -> NO_ICE",
        ]
        return LabelFunctionResult(name=self.name, label=label, confidence=confidence, rules=rules)

    def _from_psr(self, psr_mask: np.ndarray) -> LabelFunctionResult:
        label = np.full_like(psr_mask, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        confidence = np.zeros_like(psr_mask, dtype=np.float32)

        inside = psr_mask > 0
        label[inside] = IceLabel.POSSIBLE
        confidence[inside] = 0.4

        label[~inside.astype(bool)] = IceLabel.NO_ICE
        confidence[~inside.astype(bool)] = 0.05

        label[np.isnan(psr_mask)] = IceLabel.NO_ICE
        confidence[np.isnan(psr_mask)] = 0.0

        rules = ["PSR proxy for illumination -> POSSIBLE"]
        return LabelFunctionResult(name=self.name, label=label, confidence=confidence, rules=rules)
