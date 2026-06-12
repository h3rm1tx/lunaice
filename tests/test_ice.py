from __future__ import annotations

import numpy as np
import pytest

from lunaice.ice.config import IcePriorConfig, IndicatorWeights
from lunaice.ice.indicators.cpr import CPRIndicator
from lunaice.ice.indicators.dop import DOPIndicator
from lunaice.ice.indicators.psr import PSRIndicator
from lunaice.ice.indicators.roughness import RoughnessIndicator
from lunaice.ice.indicators.scattering import ScatteringIndicator
from lunaice.ice.indicators.multi_freq import MultiFrequencyIndicator
from lunaice.ice.scoring.fusion import IcePriorScore
from lunaice.ice.scoring.confidence import ConfidenceEngine
from lunaice.ice.classification.cloude_pottier import CloudePottierClassifier, ScatteringClass
from lunaice.ice.validation.report import ValidationReport
from lunaice.ice.validation.validator import IceValidator


class TestIndicatorWeights:
    def test_sum_to_one(self):
        w = IndicatorWeights()
        total = sum([w.cpr, w.dop, w.psr, w.roughness, w.multi_frequency, w.scattering])
        assert abs(total - 1.0) < 1e-6

    def test_custom_weights(self):
        w = IndicatorWeights(cpr=0.5, dop=0.1, psr=0.1, roughness=0.1, multi_frequency=0.1, scattering=0.1)
        assert abs(w.cpr - 0.5) < 1e-6


class TestCPRIndicator:
    def test_compute_high_cpr(self):
        ind = CPRIndicator(threshold=1.0)
        cpr_l = np.array([[2.0, 0.5], [1.5, np.nan]])
        cpr_s = np.array([[1.8, 0.6], [1.2, np.nan]])
        result = ind.compute(cpr_l=cpr_l, cpr_s=cpr_s)
        assert result.score[0, 0] > result.score[0, 1]
        assert np.isnan(result.score[1, 1])

    def test_threshold_zero(self):
        ind = CPRIndicator(threshold=0.0)
        cpr_l = np.ones((4, 4))
        result = ind.compute(cpr_l=cpr_l, cpr_s=cpr_l)
        assert np.all(result.score >= 0.5)


class TestDOPIndicator:
    def test_low_dop_scores_high(self):
        ind = DOPIndicator(threshold=0.3)
        dop = np.array([[0.1, 0.5], [0.2, np.nan]])
        result = ind.compute(dop_l=dop, dop_s=dop)
        assert result.score[0, 0] > result.score[0, 1]
        assert np.isnan(result.score[1, 1])


class TestPSRIndicator:
    def test_binary_mask(self):
        ind = PSRIndicator()
        mask = np.array([[1, 0], [0, 1]], dtype=np.float32)
        result = ind.compute(psr_mask=mask)
        assert result.score[0, 0] == 1.0
        assert result.score[0, 1] == 0.0


class TestRoughnessIndicator:
    def test_low_slope_scores_high(self):
        ind = RoughnessIndicator(percentile=50.0)
        slope = np.array([[0.5, 5.0], [10.0, np.nan]], dtype=np.float32)
        result = ind.compute(slope=slope)
        assert result.score[0, 0] > result.score[0, 1]


class TestScatteringIndicator:
    def test_dihedral_ice_favored(self):
        ind = ScatteringIndicator()
        entropy = np.array([[0.3, 0.7]])
        alpha = np.array([[50.0, 30.0]])
        aniso = np.array([[0.4, 0.3]])
        result = ind.compute(entropy=entropy, alpha_deg=alpha, anisotropy=aniso)
        assert result.score[0, 0] > result.score[0, 1]


class TestMultiFrequencyIndicator:
    def test_consistent_high(self):
        ind = MultiFrequencyIndicator()
        cpr_l = np.array([[2.0, 0.5]])
        cpr_s = np.array([[1.8, 0.6]])
        result = ind.compute(cpr_l=cpr_l, cpr_s=cpr_s, sigma_hh_l=np.ones((1, 2)), sigma_hh_s=np.ones((1, 2)))
        assert result.score[0, 0] == 1.0
        assert result.score[0, 1] < 0.5


class TestIcePriorScore:
    def test_weighted_fusion(self):
        weights = IndicatorWeights(cpr=1.0, dop=0.0, psr=0.0, roughness=0.0, multi_frequency=0.0, scattering=0.0)
        fusion = IcePriorScore(weights)
        from lunaice.ice.indicators.base import IndicatorResult
        indicators = {
            "cpr": IndicatorResult(name="cpr", score=np.array([[0.8, 0.2]])),
            "dop": IndicatorResult(name="dop", score=np.array([[0.5, np.nan]])),
        }
        score = fusion.compute(indicators)
        assert score[0, 0] == 0.8
        assert score[0, 1] == 0.2


class TestConfidenceEngine:
    def test_high_agreement(self):
        engine = ConfidenceEngine()
        ice_prior = np.array([[0.8, 0.8]])
        indicator_scores = {
            "cpr": np.array([[0.8, 0.3]]),
            "dop": np.array([[0.7, 0.4]]),
        }
        conf = engine.compute(ice_prior, indicator_scores, indicators_available=2)
        assert not np.isnan(conf[0, 0])
        assert not np.isnan(conf[0, 1])


class TestCloudePottierClassifier:
    def test_surface_scattering(self):
        clf = CloudePottierClassifier()
        entropy = np.array([[0.3, 0.6, 0.95]])
        alpha = np.array([[30.0, 45.0, 50.0]])
        classes = clf.classify(entropy, alpha)
        assert classes[0, 0] == ScatteringClass.SURFACE
        assert classes[0, 2] == ScatteringClass.RANDOM

    def test_class_names(self):
        assert "Surface" in CloudePottierClassifier.class_name(1)
        assert "Undefined" in CloudePottierClassifier.class_name(0)


class TestIceValidator:
    def test_evaluate_region(self):
        validator = IceValidator()
        h, w = 10, 10
        ice = np.full((h, w), 0.5)
        conf = np.full((h, w), 0.8)
        cpr = np.full((h, w), 1.5)
        dop = np.full((h, w), 0.2)
        sc = np.full((h, w), 1, dtype=np.int8)
        psr = np.zeros((h, w))
        mask = np.zeros((h, w))
        mask[2:8, 2:8] = 1.0
        m = validator.evaluate_region(ice, conf, cpr, dop, sc, psr, mask, "test")
        assert m.n_pixels == 36
        assert m.mean_ice_prior == pytest.approx(0.5)
        assert m.mean_confidence == pytest.approx(0.8)

    def test_known_regions(self):
        validator = IceValidator()
        regions = validator.known_ice_candidate_regions()
        assert len(regions) >= 5
        names = [r[0] for r in regions]
        assert "Faustini_Interior" in names


class TestValidationReport:
    def test_generate(self, tmp_path):
        report = ValidationReport(tmp_path)
        from lunaice.ice.validation.validator import RegionMetrics
        metrics = [RegionMetrics("test", 100, 0.6, 0.1, 0.8, 1.2, 0.2, 0.5, "Surface", 0.7)]
        path = report.generate(metrics, {"mean_ice_prior": 0.6})
        assert path.exists()
        html = path.read_text()
        assert "test" in html
        assert "0.6000" in html or "0.6" in html


class TestIcePriorConfig:
    def test_defaults(self):
        cfg = IcePriorConfig()
        assert cfg.cpr_threshold == 1.0
        assert cfg.weights.cpr == 0.25

    def test_from_yaml_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            IcePriorConfig.from_yaml(tmp_path / "nonexistent.yaml")
