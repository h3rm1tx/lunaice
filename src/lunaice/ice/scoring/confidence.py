from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceEngine:
    def compute(
        self,
        ice_prior: np.ndarray,
        indicator_scores: dict[str, np.ndarray],
        indicators_available: int = 6,
    ) -> np.ndarray:
        h, w = ice_prior.shape
        confidence = np.full((h, w), np.nan, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                if np.isnan(ice_prior[i, j]):
                    continue
                n_valid = 0
                n_high = 0
                n_low = 0
                values = []
                for name, arr in indicator_scores.items():
                    val = arr[i, j] if arr is not None else np.nan
                    if np.isnan(val):
                        continue
                    n_valid += 1
                    values.append(val)
                    if val >= 0.7:
                        n_high += 1
                    elif val <= 0.3:
                        n_low += 1
                if n_valid == 0:
                    continue
                values_arr = np.array(values)
                agreement = 1.0 - float(np.std(values_arr)) * 2.0
                agreement = max(0.0, min(1.0, agreement))
                coverage = n_valid / indicators_available
                dominance = 1.0 - abs(n_high - n_low) / max(n_valid, 1)
                confidence[i, j] = 0.5 * agreement + 0.3 * coverage + 0.2 * dominance
        logger.info(
            "Confidence computed: range [%.4f, %.4f], mean %.4f",
            np.nanmin(confidence), np.nanmax(confidence), np.nanmean(confidence),
        )
        return confidence.astype(np.float32)
