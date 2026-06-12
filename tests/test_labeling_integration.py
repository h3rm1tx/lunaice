"""End-to-end integration test for the labeling engine with synthetic cube data."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from lunaice.labeling.config import LabelingConfig
from lunaice.labeling.engine import LabelingEngine


@pytest.fixture
def synthetic_cube(tmp_path):
    cube_path = tmp_path / "test_cube.zarr"
    h, w = 50, 60

    ds = xr.Dataset(
        data_vars={
            "cpr": (["y", "x"], np.random.uniform(0.3, 2.5, (h, w)).astype(np.float32)),
            "dop": (["y", "x"], np.random.uniform(0.05, 0.6, (h, w)).astype(np.float32)),
            "entropy": (["y", "x"], np.random.uniform(0.1, 0.95, (h, w)).astype(np.float32)),
            "alpha_deg": (["y", "x"], np.random.uniform(10, 60, (h, w)).astype(np.float32)),
            "anisotropy": (["y", "x"], np.random.uniform(0, 1, (h, w)).astype(np.float32)),
            "slope": (["y", "x"], np.random.uniform(0, 15, (h, w)).astype(np.float32)),
            "psr_mask": (["y", "x"], np.zeros((h, w), dtype=np.float32)),
            "lon": (["y", "x"], np.random.uniform(-90, 90, (h, w)).astype(np.float32)),
            "lat": (["y", "x"], np.random.uniform(-90, -80, (h, w)).astype(np.float32)),
            "sigma_hh": (["y", "x"], np.random.uniform(0.001, 0.1, (h, w)).astype(np.float32)),
        },
        coords={
            "x": np.linspace(-50000, 50000, w),
            "y": np.linspace(50000, -50000, h),
        },
    )

    ds["psr_mask"][10:20, 15:25] = 1.0
    ds["cpr"][5:15, 5:15] = 1.8
    ds["dop"][5:15, 5:15] = 0.12
    ds["slope"][5:15, 5:15] = 1.0

    ds.chunk({"y": 25, "x": 30}).to_zarr(str(cube_path), mode="w")
    return str(cube_path)


def test_labeling_engine_end_to_end(synthetic_cube):
    out_dir = tempfile.mkdtemp()

    config = LabelingConfig(
        cube_path=synthetic_cube,
        output_dir=out_dir,
        generate_report=True,
        logging_level="WARNING",
    )

    engine = LabelingEngine(config)
    result = engine.run()

    assert "sources" in result
    assert len(result["sources"]) == 8
    assert set(result["sources"].keys()) == {
        "cpr", "dop", "psr", "multi_frequency", "scattering",
        "roughness", "illumination", "published_regions",
    }

    assert result["fused_label"] is not None
    assert result["fused_confidence"] is not None
    assert result["resolved_labels"] is not None
    assert result["disagreement_map"] is not None

    assert result["metrics"] is not None
    assert len(result["metrics"]) > 0

    assert "output_paths" in result
    paths = result["output_paths"]
    assert "weak_labels" in paths
    assert Path(paths["weak_labels"]).exists()
    assert "label_confidence" in paths
    assert Path(paths["label_confidence"]).exists()
    assert "disagreement_map" in paths
    assert Path(paths["disagreement_map"]).exists()
    assert "ice_regions" in paths
    assert Path(paths["ice_regions"]).exists()
    assert "label_statistics" in paths
    assert Path(paths["label_statistics"]).exists()
    assert "label_quality_report" in paths
    assert Path(paths["label_quality_report"]).exists()

    unique_labels = np.unique(result["resolved_labels"])
    for lbl in unique_labels:
        assert lbl in [0, 1, 2, 3], f"Unexpected label value: {lbl}"

    possible_or_better = np.sum(result["resolved_labels"] >= 1)
    assert possible_or_better > 0, "Expected at least some non-zero labels"

    label_stats_html = Path(paths["label_statistics"]).read_text()
    assert "POLARIS Label Statistics" in label_stats_html
    assert "No Ice" in label_stats_html

    quality_html = Path(paths["label_quality_report"]).read_text()
    assert "POLARIS Label Quality Report" in quality_html
    assert "Faustini_Interior" in quality_html or "Faustini" in quality_html

    shutil.rmtree(out_dir, ignore_errors=True)
