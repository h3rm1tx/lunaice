from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from lunaice.labeling.data_models import FusedLabelResult, IceLabel


@dataclass
class ConflictResult:
    disagreement_count: np.ndarray
    conflict_types: dict[str, np.ndarray]
    high_confidence_agreement: np.ndarray
    high_confidence_score: np.ndarray

    def n_conflict_pixels(self) -> int:
        return int(np.sum(self.disagreement_count > 0))

    def disagreement_rate(self) -> float:
        total = self.disagreement_count.size
        return self.n_conflict_pixels() / max(total, 1)


class ConflictResolver:
    def __init__(self, conflict_threshold: int = 2):
        self.conflict_threshold = conflict_threshold

    def analyze(self, fused: FusedLabelResult) -> ConflictResult:
        shape = fused.fused_label.shape
        label_stack = np.stack(
            [src.label for src in fused.per_source_labels.values()],
            axis=-1,
        ).astype(np.int8)

        disagreement = np.sum(label_stack != fused.fused_label[..., np.newaxis], axis=-1)

        conflict_types: dict[str, np.ndarray] = {}

        cpr_src = fused.per_source_labels.get("cpr")
        roughness_src = fused.per_source_labels.get("roughness")
        psr_src = fused.per_source_labels.get("psr")
        multi_src = fused.per_source_labels.get("multi_frequency")
        dop_src = fused.per_source_labels.get("dop")
        illum_src = fused.per_source_labels.get("illumination")

        if cpr_src is not None and roughness_src is not None:
            high_cpr = cpr_src.label >= int(IceLabel.LIKELY)
            high_rough = roughness_src.label == int(IceLabel.NO_ICE)
            conflict_types["cpr_vs_roughness"] = high_cpr & high_rough

        if cpr_src is not None and psr_src is not None:
            high_cpr_outside_psr = (cpr_src.label >= int(IceLabel.LIKELY)) & (psr_src.label <= int(IceLabel.POSSIBLE))
            conflict_types["cpr_outside_psr"] = high_cpr_outside_psr

        if dop_src is not None and multi_src is not None:
            low_dop = dop_src.label >= int(IceLabel.LIKELY)
            no_multi = multi_src.label <= int(IceLabel.POSSIBLE)
            conflict_types["dop_without_multifreq"] = low_dop & no_multi

        if cpr_src is not None and dop_src is not None:
            high_cpr_low_dop = (cpr_src.label >= int(IceLabel.LIKELY)) & (dop_src.label >= int(IceLabel.LIKELY))
            conflict_types["cpr_and_dop_agree"] = high_cpr_low_dop

        if illum_src is not None and psr_src is not None:
            dark_outside_psr = (illum_src.label >= int(IceLabel.LIKELY)) & (psr_src.label <= int(IceLabel.POSSIBLE))
            conflict_types["dark_outside_psr"] = dark_outside_psr

        high_conf_agreement = np.zeros(shape, dtype=np.float32)
        n_high = np.zeros(shape, dtype=np.int32)
        for src in fused.per_source_labels.values():
            is_high = src.label == int(IceLabel.HIGH_CONFIDENCE)
            high_conf_agreement[is_high] += src.confidence[is_high]
            n_high[is_high] += 1
        n_high = np.maximum(n_high, 1)
        high_conf_agreement = high_conf_agreement / n_high.astype(np.float32)

        high_conf_score = np.zeros(shape, dtype=np.float32)
        total_w = np.zeros(shape, dtype=np.float32)
        for src in fused.per_source_labels.values():
            is_high = src.label >= int(IceLabel.LIKELY)
            high_conf_score[is_high] += src.confidence[is_high]
            total_w[is_high] += 1.0
        total_w = np.maximum(total_w, 1.0)
        high_conf_score = high_conf_score / total_w

        return ConflictResult(
            disagreement_count=disagreement,
            conflict_types=conflict_types,
            high_confidence_agreement=high_conf_agreement,
            high_confidence_score=high_conf_score,
        )

    def resolve(self, fused: FusedLabelResult, conflict_result: ConflictResult) -> np.ndarray:
        result = fused.fused_label.copy()

        for conflict_name, mask in conflict_result.conflict_types.items():
            if conflict_name == "cpr_vs_roughness":
                high_conf = fused.fused_confidence >= 0.6
                result[mask & ~high_conf] = int(IceLabel.NO_ICE)

            elif conflict_name == "cpr_outside_psr":
                result[mask] = int(IceLabel.POSSIBLE)

            elif conflict_name == "dop_without_multifreq":
                result[mask] = int(IceLabel.POSSIBLE)

            elif conflict_name == "dark_outside_psr":
                result[mask] = int(IceLabel.POSSIBLE)

        return result
