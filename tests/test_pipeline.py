from __future__ import annotations

import numpy as np
import pytest

from lunaice.config import DFSARConfig, ProcessingConfig, SpeckleFilterConfig


def test_config_from_yaml(tmp_path):
    yml = tmp_path / "test_config.yaml"
    yml.write_text("""
input_file: "/data/input.h5"
output_dir: "/data/output"
band: "S"
processing:
  radiometric_calibration: true
  polarimetric_calibration: true
  speckle_filter:
    method: "refined_lee"
    window_size: 5
  generate_cloude_pottier: true
  generate_cpr: true
  generate_dop: true
logging_level: "DEBUG"
""")
    config = DFSARConfig.from_yaml(yml)
    assert config.band == "S"
    assert config.processing.radiometric_calibration is True
    assert config.processing.speckle_filter is not None
    assert config.processing.speckle_filter.method == "refined_lee"
    assert config.processing.speckle_filter.window_size == 5


def test_speckle_filter_validation():
    with pytest.raises(ValueError, match="Speckle filter must be one of"):
        SpeckleFilterConfig(method="invalid_filter")

    with pytest.raises(ValueError, match="window_size must be odd"):
        SpeckleFilterConfig(method="boxcar", window_size=4)


def test_speckle_filter_methods():
    for method in ["refined_lee", "boxcar", "idani"]:
        cfg = SpeckleFilterConfig(method=method, window_size=5)
        assert cfg.method == method


def test_processing_config_defaults():
    cfg = ProcessingConfig()
    assert cfg.radiometric_calibration is True
    assert cfg.generate_cloude_pottier is True
    assert cfg.multilook_range == 1
