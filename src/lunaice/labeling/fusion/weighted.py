from __future__ import annotations

import numpy as np

from lunaice.labeling.config import LabelWeights
from lunaice.labeling.data_models import FusedLabelResult, IceLabel, LabelSource


class WeightedVoteFusion:
    def __init__(self, weights: LabelWeights):
        self.weights = weights

    def _weight_for_key(self, key: str) -> float:
        mapping = {
            "cpr": self.weights.cpr,
            "dop": self.weights.dop,
            "psr": self.weights.psr,
            "multi_frequency": self.weights.multi_frequency,
            "scattering": self.weights.scattering,
            "roughness": self.weights.roughness,
            "illumination": self.weights.illumination,
            "published_regions": self.weights.published_regions,
        }
        return mapping.get(key, 0.0)

    def fuse(self, sources: dict[str, LabelSource]) -> FusedLabelResult:
        keys = list(sources.keys())
        if not keys:
            raise ValueError("No label sources provided")

        first = sources[keys[0]]
        shape = first.label.shape
        n_sources = len(keys)

        weighted_scores = np.zeros(shape, dtype=np.float32)
        total_weight = 0.0

        for key in keys:
            w = self._weight_for_key(key)
            src = sources[key]
            label_val = src.label.astype(np.int8)
            conf_val = src.confidence

            vote_val = np.zeros_like(label_val, dtype=np.float32)
            vote_val[label_val == int(IceLabel.POSSIBLE)] = 0.3
            vote_val[label_val == int(IceLabel.LIKELY)] = 0.6
            vote_val[label_val == int(IceLabel.HIGH_CONFIDENCE)] = 0.9

            weighted_scores += w * vote_val * conf_val
            total_weight += w

        weighted_scores = weighted_scores / max(total_weight, 1e-8)

        fused_label = np.full(shape, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        high = weighted_scores >= 0.7
        likely = (weighted_scores >= 0.4) & (weighted_scores < 0.7)
        possible = (weighted_scores >= 0.15) & (weighted_scores < 0.4)

        fused_label[high] = int(IceLabel.HIGH_CONFIDENCE)
        fused_label[likely] = int(IceLabel.LIKELY)
        fused_label[possible] = int(IceLabel.POSSIBLE)

        fused_confidence = weighted_scores.copy()

        disagreement = np.zeros(shape, dtype=np.int32)
        for key in keys:
            src = sources[key]
            disagreement += (src.label != fused_label).astype(np.int32)

        return FusedLabelResult(
            fused_label=fused_label,
            fused_confidence=fused_confidence,
            per_source_labels=sources,
            disagreement_map=disagreement,
            n_sources=n_sources,
            metadata={"method": "weighted_vote", "weights": vars(self.weights)},
        )
