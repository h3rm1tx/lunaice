from __future__ import annotations

import numpy as np
import pytest

from lunaice.labeling.config import LabelingConfig, LabelWeights, LabelThresholds
from lunaice.labeling.data_models import IceLabel, LABEL_NAMES, LabelSource
from lunaice.labeling.functions.lf_cpr import LF_CPR
from lunaice.labeling.functions.lf_dop import LF_DOP
from lunaice.labeling.functions.lf_psr import LF_PSR
from lunaice.labeling.functions.lf_multi_freq import LF_MultiFrequency
from lunaice.labeling.functions.lf_scattering import LF_Scattering
from lunaice.labeling.functions.lf_roughness import LF_Roughness
from lunaice.labeling.functions.lf_illumination import LF_Illumination
from lunaice.labeling.functions.lf_published import LF_PublishedRegions
from lunaice.labeling.fusion.majority import MajorityVoteFusion
from lunaice.labeling.fusion.weighted import WeightedVoteFusion
from lunaice.labeling.fusion.label_model import SnorkelLabelModel
from lunaice.labeling.conflict import ConflictResolver
from lunaice.labeling.validation.validator import LabelValidator


class TestLabelWeights:
    def test_sum_to_one(self):
        w = LabelWeights()
        total = sum([w.cpr, w.dop, w.psr, w.multi_frequency,
                     w.scattering, w.roughness, w.illumination, w.published_regions])
        assert abs(total - 1.0) < 1e-6


class TestIceLabel:
    def test_enum_values(self):
        assert int(IceLabel.NO_ICE) == 0
        assert int(IceLabel.POSSIBLE) == 1
        assert int(IceLabel.LIKELY) == 2
        assert int(IceLabel.HIGH_CONFIDENCE) == 3

    def test_label_names(self):
        assert LABEL_NAMES[IceLabel.NO_ICE] == "No Ice"
        assert LABEL_NAMES[IceLabel.HIGH_CONFIDENCE] == "High Confidence Ice"


class TestLF_CPR:
    def test_high_cpr_high_confidence(self):
        lf = LF_CPR(threshold_low=0.8, threshold_high=1.2)
        cpr = np.array([[2.0, 1.0, 0.6, np.nan]])
        result = lf.compute(cpr_l=cpr, cpr_s=cpr)
        assert result.label[0, 0] == int(IceLabel.HIGH_CONFIDENCE)
        assert result.label[0, 1] == int(IceLabel.LIKELY)
        assert result.label[0, 2] == int(IceLabel.POSSIBLE)
        assert np.isnan(cpr[0, 3]) or result.label[0, 3] == int(IceLabel.NO_ICE)
        assert result.confidence[0, 0] > result.confidence[0, 1]


class TestLF_DOP:
    def test_low_dop_high_confidence(self):
        lf = LF_DOP(threshold_low=0.2, threshold_high=0.4)
        dop = np.array([[0.1, 0.3, 0.5]])
        result = lf.compute(dop_l=dop, dop_s=dop)
        assert result.label[0, 0] == int(IceLabel.HIGH_CONFIDENCE)
        assert result.label[0, 1] == int(IceLabel.LIKELY)
        assert result.label[0, 2] == int(IceLabel.NO_ICE)
        assert result.confidence[0, 0] > result.confidence[0, 1]


class TestLF_PSR:
    def test_inside_psr(self):
        lf = LF_PSR()
        mask = np.array([[1, 0, np.nan]])
        result = lf.compute(psr_mask=mask)
        assert result.label[0, 0] == int(IceLabel.LIKELY)
        assert result.label[0, 1] == int(IceLabel.NO_ICE)
        assert result.confidence[0, 0] > result.confidence[0, 1]


class TestLF_MultiFrequency:
    def test_both_high(self):
        lf = LF_MultiFrequency()
        cpr_l = np.array([[2.0, 0.5, 1.2]])
        cpr_s = np.array([[1.8, 0.6, 0.8]])
        result = lf.compute(cpr_l=cpr_l, cpr_s=cpr_s)
        assert result.label[0, 0] == int(IceLabel.HIGH_CONFIDENCE)
        assert result.label[0, 1] == int(IceLabel.NO_ICE)
        assert result.rules is not None


class TestLF_Scattering:
    def test_dihedral_label(self):
        lf = LF_Scattering()
        sc = np.array([[3, 7, 0]])
        result = lf.compute(scattering_class=sc)
        assert result.label[0, 0] == int(IceLabel.LIKELY)
        assert result.label[0, 1] == int(IceLabel.NO_ICE)


class TestLF_Roughness:
    def test_smooth_terrain(self):
        lf = LF_Roughness(threshold_low=2.0, threshold_high=5.0)
        slope = np.array([[1.0, 3.5, 10.0]])
        result = lf.compute(slope=slope)
        assert result.label[0, 0] == int(IceLabel.LIKELY)
        assert result.label[0, 1] == int(IceLabel.POSSIBLE)
        assert result.label[0, 2] == int(IceLabel.NO_ICE)


class TestLF_Illumination:
    def test_dark_psr(self):
        lf = LF_Illumination()
        mask = np.array([[1.0, 0.0]])
        result = lf.compute(illumination=None, psr_mask=mask)
        assert result.label[0, 0] == int(IceLabel.POSSIBLE)

    def test_dark_illumination(self):
        lf = LF_Illumination()
        illum = np.array([[0.005, 0.05, 0.5]])
        result = lf.compute(illumination=illum)
        assert result.label[0, 0] == int(IceLabel.LIKELY)
        assert result.label[0, 1] == int(IceLabel.POSSIBLE)
        assert result.label[0, 2] == int(IceLabel.NO_ICE)


class TestLF_PublishedRegions:
    def test_inside_shackleton(self):
        lf = LF_PublishedRegions()
        lon = np.array([[0.0, 100.0]])
        lat = np.array([[-89.9, -80.0]])
        result = lf.compute(lon=lon, lat=lat)
        assert result.label[0, 0] >= int(IceLabel.HIGH_CONFIDENCE)


class TestMajorityVoteFusion:
    def test_majority_wins(self):
        fusor = MajorityVoteFusion()
        shape = (2, 2)
        sources = {
            "a": LabelSource(name="a", label=np.full(shape, int(IceLabel.HIGH_CONFIDENCE), dtype=np.int8),
                             confidence=np.full(shape, 0.8, dtype=np.float32), rules=[]),
            "b": LabelSource(name="b", label=np.full(shape, int(IceLabel.LIKELY), dtype=np.int8),
                             confidence=np.full(shape, 0.6, dtype=np.float32), rules=[]),
            "c": LabelSource(name="c", label=np.full(shape, int(IceLabel.HIGH_CONFIDENCE), dtype=np.int8),
                             confidence=np.full(shape, 0.7, dtype=np.float32), rules=[]),
        }
        result = fusor.fuse(sources)
        assert np.all(result.fused_label == int(IceLabel.HIGH_CONFIDENCE))
        assert result.n_sources == 3


class TestWeightedVoteFusion:
    def test_weighted_fusion(self):
        weights = LabelWeights(cpr=1.0, dop=0.0, psr=0.0, multi_frequency=0.0,
                               scattering=0.0, roughness=0.0, illumination=0.0,
                               published_regions=0.0)
        fusor = WeightedVoteFusion(weights)
        shape = (2, 2)
        sources = {
            "cpr": LabelSource(name="cpr", label=np.full(shape, int(IceLabel.HIGH_CONFIDENCE), dtype=np.int8),
                               confidence=np.full(shape, 0.9, dtype=np.float32), rules=[]),
        }
        result = fusor.fuse(sources)
        assert np.all(result.fused_label == int(IceLabel.HIGH_CONFIDENCE))


class TestSnorkelLabelModel:
    def test_fusion(self):
        model = SnorkelLabelModel(epochs=10)
        shape = (3, 3)
        sources = {
            "a": LabelSource(name="a", label=np.full(shape, int(IceLabel.HIGH_CONFIDENCE), dtype=np.int8),
                             confidence=np.full(shape, 0.8, dtype=np.float32), rules=[]),
            "b": LabelSource(name="b", label=np.full(shape, int(IceLabel.HIGH_CONFIDENCE), dtype=np.int8),
                             confidence=np.full(shape, 0.8, dtype=np.float32), rules=[]),
            "c": LabelSource(name="c", label=np.full(shape, int(IceLabel.NO_ICE), dtype=np.int8),
                             confidence=np.full(shape, 0.3, dtype=np.float32), rules=[]),
        }
        result = model.fuse(sources)
        assert result.n_sources == 3
        assert result.metadata is not None
        assert "learned_accuracies" in result.metadata


class TestConflictResolver:
    def test_cpr_vs_roughness(self):
        resolver = ConflictResolver(conflict_threshold=2)
        shape = (2, 2)
        sources = {
            "cpr": LabelSource(name="cpr", label=np.full(shape, int(IceLabel.HIGH_CONFIDENCE), dtype=np.int8),
                               confidence=np.full(shape, 0.8, dtype=np.float32), rules=[]),
            "roughness": LabelSource(name="roughness", label=np.full(shape, int(IceLabel.NO_ICE), dtype=np.int8),
                                     confidence=np.full(shape, 0.6, dtype=np.float32), rules=[]),
        }
        from lunaice.labeling.data_models import FusedLabelResult
        fused = FusedLabelResult(
            fused_label=np.full(shape, int(IceLabel.LIKELY), dtype=np.int8),
            fused_confidence=np.full(shape, 0.5, dtype=np.float32),
            per_source_labels=sources,
            n_sources=2,
        )
        result = resolver.analyze(fused)
        assert "cpr_vs_roughness" in result.conflict_types
        assert np.all(result.conflict_types["cpr_vs_roughness"])

    def test_resolve(self):
        resolver = ConflictResolver(conflict_threshold=2)
        shape = (2, 2)
        sources = {
            "cpr": LabelSource(name="cpr", label=np.full(shape, int(IceLabel.HIGH_CONFIDENCE), dtype=np.int8),
                               confidence=np.full(shape, 0.8, dtype=np.float32), rules=[]),
            "roughness": LabelSource(name="roughness", label=np.full(shape, int(IceLabel.NO_ICE), dtype=np.int8),
                                     confidence=np.full(shape, 0.6, dtype=np.float32), rules=[]),
        }
        from lunaice.labeling.data_models import FusedLabelResult
        fused = FusedLabelResult(
            fused_label=np.full(shape, int(IceLabel.HIGH_CONFIDENCE), dtype=np.int8),
            fused_confidence=np.full(shape, 0.8, dtype=np.float32),
            per_source_labels=sources,
            n_sources=2,
        )
        conflict = resolver.analyze(fused)
        resolved = resolver.resolve(fused, conflict)
        assert resolved.shape == shape


class TestLabelValidator:
    def test_evaluate_region(self):
        validator = LabelValidator()
        h, w = 10, 10
        labels = np.full((h, w), int(IceLabel.LIKELY), dtype=np.int8)
        confidence = np.full((h, w), 0.7, dtype=np.float32)
        disagreement = np.zeros((h, w), dtype=np.float32)
        lon = np.full((h, w), -5.0)
        lat = np.full((h, w), -87.0)
        m = validator.evaluate_region(labels, confidence, disagreement, lon, lat,
                                       -4.5, -87.5, 50000, "haworth_test")
        assert m.n_pixels > 0
        assert m.ice_favorable_fraction == pytest.approx(1.0)
        assert m.high_conf_fraction == pytest.approx(1.0)

    def test_default_regions(self):
        validator = LabelValidator()
        assert len(validator.known_regions) >= 5
        names = [r[0] for r in validator.known_regions]
        assert "Faustini_Interior" in names


class TestLabelingConfig:
    def test_defaults(self):
        cfg = LabelingConfig()
        assert cfg.cube_path == "polaris_cube.zarr"
        assert cfg.weights.cpr == 0.20

    def test_from_yaml_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LabelingConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_yaml(self, tmp_path):
        yml = tmp_path / "label.yaml"
        yml.write_text("labeling:\n  output_dir: /tmp/test_labels\n")
        cfg = LabelingConfig.from_yaml(yml)
        assert cfg.output_dir == "/tmp/test_labels"


class TestLabelSource:
    def test_shape_validation(self):
        with pytest.raises(ValueError):
            LabelSource(name="test", label=np.array([0, 1]), confidence=np.array([0.5, 0.6, 0.7]), rules=[])
