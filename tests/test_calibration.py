from __future__ import annotations

import numpy as np
import pytest

from lunaice.models.schemas import PolarimetricData
from lunaice.processing.calibration import PolarimetricCalibrator, RadiometricCalibrator


class TestRadiometricCalibrator:
    def test_calibrate_quad_pol(self, sample_slc_data: PolarimetricData):
        original_hh = sample_slc_data.hh.copy()
        cal = RadiometricCalibrator(cal_constant=2.0)
        result = cal.calibrate(sample_slc_data)
        assert result is sample_slc_data
        assert result.hh is not None
        expected_power = np.abs(original_hh) ** 2 / 2.0
        assert np.allclose(np.abs(result.hh) ** 2, expected_power)

    def test_calibrate_partial(self):
        data = PolarimetricData(hh=np.ones((4, 4), dtype=complex))
        cal = RadiometricCalibrator(cal_constant=4.0)
        result = cal.calibrate(data)
        assert np.allclose(np.abs(result.hh) ** 2, 0.25)


class TestPolarimetricCalibrator:
    def test_calibrate_quad_pol(self, sample_slc_data: PolarimetricData):
        cal = PolarimetricCalibrator(
            co_pol_phase_correction=-50.0,
            cross_pol_phase_correction=-5.0,
            cross_talk_hv=0.01,
            cross_talk_vh=0.01,
        )
        result = cal.calibrate(sample_slc_data)
        assert result is sample_slc_data
        assert result.hh.shape == sample_slc_data.hh.shape

    def test_skip_non_quadpol(self):
        data = PolarimetricData(hh=np.ones((4, 4), dtype=complex))
        cal = PolarimetricCalibrator()
        result = cal.calibrate(data)
        np.testing.assert_array_equal(result.hh, data.hh)
