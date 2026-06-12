from __future__ import annotations

import numpy as np

from lunaice.labeling.data_models import FusedLabelResult, IceLabel, LabelSource


class MajorityVoteFusion:
    def fuse(self, sources: dict[str, LabelSource]) -> FusedLabelResult:
        keys = list(sources.keys())
        if not keys:
            raise ValueError("No label sources provided")

        first = sources[keys[0]]
        shape = first.label.shape
        n_sources = len(keys)
        label_stack = np.stack([sources[k].label for k in keys], axis=-1)
        conf_stack = np.stack([sources[k].confidence for k in keys], axis=-1)

        fused_label = np.full(shape, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        fused_confidence = np.zeros(shape, dtype=np.float32)

        for label_val in [IceLabel.HIGH_CONFIDENCE, IceLabel.LIKELY, IceLabel.POSSIBLE, IceLabel.NO_ICE]:
            counts = np.sum(label_stack == int(label_val), axis=-1)
            mask = counts > n_sources / 2
            fused_label[mask] = int(label_val)

        tie_labels = np.full(shape, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        for label_val in [IceLabel.HIGH_CONFIDENCE, IceLabel.LIKELY, IceLabel.POSSIBLE, IceLabel.NO_ICE]:
            counts = np.sum(label_stack == int(label_val), axis=-1)
            mask = counts >= n_sources / 2
            tie_labels[mask] = int(label_val)

        no_majority = fused_label == IceLabel.NO_ICE
        for label_val in [IceLabel.HIGH_CONFIDENCE, IceLabel.LIKELY, IceLabel.POSSIBLE]:
            mask = no_majority & (tie_labels == int(label_val))
            fused_label[mask] = int(label_val)

        conf_valid = ~np.isnan(conf_stack).all(axis=-1)
        fused_confidence[conf_valid] = np.nanmean(
            np.where(label_stack == fused_label[..., np.newaxis], conf_stack, 0.0),
            axis=-1,
        )[conf_valid]

        disagreement = np.sum(label_stack != fused_label[..., np.newaxis], axis=-1)

        return FusedLabelResult(
            fused_label=fused_label,
            fused_confidence=fused_confidence,
            per_source_labels=sources,
            disagreement_map=disagreement,
            n_sources=n_sources,
            metadata={"method": "majority_vote"},
        )
