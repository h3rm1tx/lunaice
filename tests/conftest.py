from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunaice.config import DFSARConfig, ProcessingConfig, SpeckleFilterConfig
from lunaice.models.schemas import Metadata, PolarimetricData


@pytest.fixture
def sample_slc_data() -> PolarimetricData:
    np.random.seed(42)
    h = 128
    w = 128
    hh = np.random.randn(h, w) + 1j * np.random.randn(h, w)
    hv = np.random.randn(h, w) + 1j * np.random.randn(h, w)
    vh = np.random.randn(h, w) + 1j * np.random.randn(h, w)
    vv = np.random.randn(h, w) + 1j * np.random.randn(h, w)
    return PolarimetricData(hh=hh, hv=hv, vh=vh, vv=vv, metadata=_test_metadata())


def _test_metadata() -> Metadata:
    return Metadata(
        product_id="TEST_001",
        processing_level="L1A",
        frequency_band="L",
        polarization_mode="quad_pol",
        acquisition_time="2020-01-15T12:00:00",
        orbit_number=1234,
        incidence_angle_deg=30.0,
        slant_range_resolution_m=15.0,
        azimuth_resolution_m=15.0,
        looks_range=1,
        looks_azimuth=1,
        calibration_constant=1.0,
        wavelength_cm=24.0,
        center_latitude=-85.0,
        center_longitude=30.0,
        pixel_spacing_m=15.0,
    )


@pytest.fixture
def default_config() -> DFSARConfig:
    return DFSARConfig(
        input_file="/tmp/test_input",
        output_dir="/tmp/test_output",
        band="L",
        processing=ProcessingConfig(
            speckle_filter=SpeckleFilterConfig(method="refined_lee", window_size=5),
        ),
    )


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d
