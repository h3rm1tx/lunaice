from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


class FrequencyBand(str, Enum):
    L_BAND = "L"
    S_BAND = "S"


class PolarizationMode(str, Enum):
    QUAD_POL = "quad_pol"
    DUAL_POL = "dual_pol"
    HYBRID_POL = "hybrid_pol"


class ProcessingLevel(str, Enum):
    LEVEL_0A = "L0A"
    LEVEL_0B = "L0B"
    LEVEL_1A = "L1A"
    LEVEL_1B = "L1B"
    LEVEL_2 = "L2"


@dataclass
class Metadata:
    product_id: str
    processing_level: ProcessingLevel
    frequency_band: FrequencyBand
    polarization_mode: PolarizationMode
    acquisition_time: datetime
    orbit_number: int
    incidence_angle_deg: float
    slant_range_resolution_m: float
    azimuth_resolution_m: float
    looks_range: int
    looks_azimuth: int
    calibration_constant: float
    wavelength_cm: float
    center_latitude: float
    center_longitude: float
    pixel_spacing_m: float
    cal_params: Optional[dict] = None

    @classmethod
    def from_dict(cls, d: dict) -> Metadata:
        return cls(
            product_id=d.get("product_id", ""),
            processing_level=ProcessingLevel(d.get("processing_level", "L1A")),
            frequency_band=FrequencyBand(d.get("frequency_band", "L")),
            polarization_mode=PolarizationMode(d.get("polarization_mode", "quad_pol")),
            acquisition_time=datetime.fromisoformat(d.get("acquisition_time", "2020-01-01T00:00:00")),
            orbit_number=int(d.get("orbit_number", 0)),
            incidence_angle_deg=float(d.get("incidence_angle_deg", 30.0)),
            slant_range_resolution_m=float(d.get("slant_range_resolution_m", 15.0)),
            azimuth_resolution_m=float(d.get("azimuth_resolution_m", 15.0)),
            looks_range=int(d.get("looks_range", 1)),
            looks_azimuth=int(d.get("looks_azimuth", 1)),
            calibration_constant=float(d.get("calibration_constant", 1.0)),
            wavelength_cm=float(d.get("wavelength_cm", 24.0)),
            center_latitude=float(d.get("center_latitude", 0.0)),
            center_longitude=float(d.get("center_longitude", 0.0)),
            pixel_spacing_m=float(d.get("pixel_spacing_m", 15.0)),
            cal_params=d.get("cal_params"),
        )


@dataclass
class CalibrationConstants:
    k_hh: float = 1.0
    k_hv: float = 1.0
    k_vh: float = 1.0
    k_vv: float = 1.0
    phase_offset_hh_vv: float = 0.0
    phase_offset_hv_vh: float = 0.0
    cross_talk_hv: float = 0.0
    cross_talk_vh: float = 0.0
    channel_imbalance_amp: float = 1.0
    channel_imbalance_phase: float = 0.0


@dataclass
class PolarimetricData:
    data_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    hh: Optional[np.ndarray] = None
    hv: Optional[np.ndarray] = None
    vh: Optional[np.ndarray] = None
    vv: Optional[np.ndarray] = None
    metadata: Optional[Metadata] = None
    calibration: Optional[CalibrationConstants] = None

    @property
    def shape(self) -> tuple:
        if self.hh is not None:
            return self.hh.shape
        return (0, 0)

    @property
    def is_quad_pol(self) -> bool:
        return all(x is not None for x in [self.hh, self.hv, self.vh, self.vv])

    def validate(self) -> bool:
        shapes = []
        for arr in [self.hh, self.hv, self.vh, self.vv]:
            if arr is not None:
                shapes.append(arr.shape)
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent channel shapes: {shapes}")
        return True


@dataclass
class CoherencyMatrix:
    t11: np.ndarray
    t12: np.ndarray
    t13: np.ndarray
    t21: np.ndarray
    t22: np.ndarray
    t23: np.ndarray
    t31: np.ndarray
    t32: np.ndarray
    t33: np.ndarray

    @property
    def shape(self) -> tuple:
        return self.t11.shape


@dataclass
class DecompositionProducts:
    entropy: Optional[np.ndarray] = None
    alpha_deg: Optional[np.ndarray] = None
    anisotropy: Optional[np.ndarray] = None
    lambda_1: Optional[np.ndarray] = None
    lambda_2: Optional[np.ndarray] = None
    lambda_3: Optional[np.ndarray] = None
    alpha_1: Optional[np.ndarray] = None
    alpha_2: Optional[np.ndarray] = None
    alpha_3: Optional[np.ndarray] = None
    cpr: Optional[np.ndarray] = None
    dop: Optional[np.ndarray] = None
    sigma_hh: Optional[np.ndarray] = None
    sigma_hv: Optional[np.ndarray] = None
    sigma_vv: Optional[np.ndarray] = None
    gamma_hh: Optional[np.ndarray] = None
    gamma_hv: Optional[np.ndarray] = None
    gamma_vv: Optional[np.ndarray] = None
    span: Optional[np.ndarray] = None
    odd_bounce: Optional[np.ndarray] = None
    double_bounce: Optional[np.ndarray] = None
    volume_scattering: Optional[np.ndarray] = None


@dataclass
class ProcessingSummary:
    input_file: str
    output_dir: str
    products_generated: list[str]
    processing_time_s: float
    config_snapshot: dict
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
