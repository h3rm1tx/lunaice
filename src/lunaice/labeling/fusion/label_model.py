from __future__ import annotations

import numpy as np

from lunaice.labeling.data_models import FusedLabelResult, IceLabel, LabelSource


class SnorkelLabelModel:
    def __init__(self, lr: float = 0.01, epochs: int = 100):
        self.lr = lr
        self.epochs = epochs

    def fuse(self, sources: dict[str, LabelSource]) -> FusedLabelResult:
        keys = list(sources.keys())
        if not keys:
            raise ValueError("No label sources provided")

        m = len(keys)
        first = sources[keys[0]]
        shape = first.label.shape
        n_pixels = shape[0] * shape[1]

        L = np.zeros((n_pixels, m), dtype=np.int8)
        conf = np.zeros((n_pixels, m), dtype=np.float32)

        for i, key in enumerate(keys):
            src = sources[key]
            flat_label = src.label.ravel()
            flat_conf = src.confidence.ravel()
            L[:, i] = flat_label
            conf[:, i] = flat_conf

        L_abstain = L.copy()
        L_abstain[conf < 0.1] = -1

        acc = np.ones(m, dtype=np.float32) * 0.7
        for _ in range(self.epochs):
            for i in range(m):
                other_mask = np.ones(m, dtype=bool)
                other_mask[i] = False

                agreement = np.mean(
                    (L_abstain[:, other_mask] == L_abstain[:, i][:, np.newaxis]) &
                    (L_abstain[:, other_mask] >= 0),
                    axis=1,
                )
                agreement[np.isnan(agreement)] = 0.0
                valid = (L_abstain[:, i] >= 0) & (~np.isnan(agreement))
                if np.any(valid):
                    acc[i] = 0.9 * np.mean(agreement[valid]) + 0.1

        L_weighted = np.zeros(n_pixels, dtype=np.float32)
        total_w = np.zeros(n_pixels, dtype=np.float32)

        for i in range(m):
            mask = L_abstain[:, i] >= 0
            L_weighted[mask] += acc[i] * conf[mask, i] * L_abstain[mask, i].astype(np.float32)
            total_w[mask] += acc[i] * conf[mask, i]

        total_w = np.maximum(total_w, 1e-8)
        L_weighted = L_weighted / total_w

        label_flat = np.full(n_pixels, fill_value=int(IceLabel.NO_ICE), dtype=np.int8)
        high = L_weighted >= 2.5
        likely = (L_weighted >= 1.5) & (L_weighted < 2.5)
        possible = (L_weighted >= 0.5) & (L_weighted < 1.5)

        label_flat[high] = int(IceLabel.HIGH_CONFIDENCE)
        label_flat[likely] = int(IceLabel.LIKELY)
        label_flat[possible] = int(IceLabel.POSSIBLE)

        conf_flat = np.zeros(n_pixels, dtype=np.float32)
        conf_flat[label_flat == int(IceLabel.HIGH_CONFIDENCE)] = 0.8
        conf_flat[label_flat == int(IceLabel.LIKELY)] = 0.55
        conf_flat[label_flat == int(IceLabel.POSSIBLE)] = 0.3
        conf_flat[label_flat == int(IceLabel.NO_ICE)] = 0.05

        dis_flat = np.sum(L_abstain != label_flat[:, np.newaxis], axis=1)

        return FusedLabelResult(
            fused_label=label_flat.reshape(shape),
            fused_confidence=conf_flat.reshape(shape),
            per_source_labels=sources,
            disagreement_map=dis_flat.reshape(shape),
            n_sources=m,
            metadata={
                "method": "snorkel_label_model",
                "learned_accuracies": {keys[i]: float(acc[i]) for i in range(m)},
            },
        )
