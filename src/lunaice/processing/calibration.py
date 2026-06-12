from __future__ import annotations

import logging

import numpy as np

from lunaice.models.schemas import CalibrationConstants, PolarimetricData

logger = logging.getLogger(__name__)


class RadiometricCalibrator:
    def __init__(self, cal_constant: float = 1.0, apply_antenna_pattern: bool = False):
        self.cal_constant = cal_constant
        self.apply_antenna_pattern = apply_antenna_pattern

    def calibrate(self, data: PolarimetricData) -> PolarimetricData:
        if not data.is_quad_pol:
            logger.warning("Data is not quad-pol; calibrating available channels")
        k = data.calibration.k_hh if data.calibration else self.cal_constant
        for pol in ["hh", "hv", "vh", "vv"]:
            arr = getattr(data, pol, None)
            if arr is not None:
                cal_val = getattr(data.calibration, f"k_{pol}", k) if data.calibration else k
                scale = np.float32(1.0 / np.sqrt(max(cal_val, 1e-30)))
                setattr(data, pol, (arr * scale).astype(np.complex64))
                logger.debug("Radiometric calibration applied to %s (K=%f)", pol.upper(), cal_val)
        logger.info("Radiometric calibration complete")
        return data


class PolarimetricCalibrator:
    def __init__(
        self,
        co_pol_phase_correction: float = -50.0,
        cross_pol_phase_correction: float = -5.0,
        cross_talk_hv: float = 0.0,
        cross_talk_vh: float = 0.0,
        channel_imbalance_amp: float = 1.0,
        channel_imbalance_phase: float = 0.0,
    ):
        self.co_pol_phase_correction = np.radians(co_pol_phase_correction)
        self.cross_pol_phase_correction = np.radians(cross_pol_phase_correction)
        self.cross_talk_hv = cross_talk_hv
        self.cross_talk_vh = cross_talk_vh
        self.channel_imbalance_amp = channel_imbalance_amp
        self.channel_imbalance_phase = np.radians(channel_imbalance_phase)

    def calibrate(self, data: PolarimetricData) -> PolarimetricData:
        if not data.is_quad_pol:
            logger.warning("Polarimetric calibration requires quad-pol data; skipping")
            return data
        cal = data.calibration
        if cal:
            self.co_pol_phase_correction = np.radians(cal.phase_offset_hh_vv)
            self.cross_pol_phase_correction = np.radians(cal.phase_offset_hv_vh)
            self.cross_talk_hv = cal.cross_talk_hv
            self.cross_talk_vh = cal.cross_talk_vh
            self.channel_imbalance_amp = cal.channel_imbalance_amp
            self.channel_imbalance_phase = np.radians(cal.channel_imbalance_phase)

        imbalance = self.channel_imbalance_amp * np.exp(1j * self.channel_imbalance_phase)
        data.hh *= np.exp(1j * self.co_pol_phase_correction)
        data.vv *= np.exp(1j * self.co_pol_phase_correction)
        data.hv *= np.exp(1j * self.cross_pol_phase_correction)
        data.vh *= np.exp(1j * self.cross_pol_phase_correction)
        data.hh -= self.cross_talk_hv * data.vh
        data.vv -= self.cross_talk_vh * data.hv
        data.hv -= self.cross_talk_hv * data.vv
        data.vh -= self.cross_talk_vh * data.hh
        data.hv /= imbalance
        data.vh /= imbalance
        logger.info("Polarimetric calibration complete")
        return data
