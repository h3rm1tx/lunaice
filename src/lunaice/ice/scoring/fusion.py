from __future__ import annotations

import logging

import numpy as np

from lunaice.ice.config import IndicatorWeights
from lunaice.ice.indicators.base import IndicatorResult

logger = logging.getLogger(__name__)


class IcePriorScore:
    def __init__(self, weights: IndicatorWeights):
        self.weights = weights

    def compute(self, indicators: dict[str, IndicatorResult]) -> np.ndarray:
        shape = None
        for name, result in indicators.items():
            if result is not None and result.score is not None:
                shape = result.score.shape
                break
        if shape is None:
            raise ValueError("No valid indicator results to fuse")
        weighted_sum = np.zeros(shape, dtype=np.float32)
        weight_total = np.zeros(shape, dtype=np.float32)
        weight_map = {
            "cpr": self.weights.cpr,
            "dop": self.weights.dop,
            "psr": self.weights.psr,
            "roughness": self.weights.roughness,
            "multi_frequency": self.weights.multi_frequency,
            "scattering": self.weights.scattering,
        }
        contributions = {}
        for key, w in weight_map.items():
            result = indicators.get(key)
            if result is not None and result.score is not None:
                valid = ~np.isnan(result.score)
                weighted_sum[valid] += w * result.score[valid]
                weight_total[valid] += w
                contributions[key] = result.score
        ice_prior = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)
        logger.info(
            "Ice prior score computed: range [%.4f, %.4f], valid pixels %d/%d",
            np.nanmin(ice_prior), np.nanmax(ice_prior),
            np.sum(~np.isnan(ice_prior)), ice_prior.size,
        )
        return ice_prior.astype(np.float32)
