from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunaice.io.reader import DFSARReader, read_pds4_label
from lunaice.models.schemas import PolarimetricData


def test_read_pds4_label_not_found():
    with pytest.raises(FileNotFoundError):
        read_pds4_label("/nonexistent/label.xml")


def test_dfsar_reader_metadata(tmp_path):
    data_file = tmp_path / "test_slc.npy"
    np.save(str(data_file), np.zeros((10, 10), dtype=complex))
    reader = DFSARReader(data_file)
    met = reader.read_metadata()
    assert met.product_id == ""


def test_validate_consistent_shapes():
    data = PolarimetricData(
        hh=np.ones((10, 10), dtype=complex),
        hv=np.ones((10, 10), dtype=complex),
        vv=np.ones((10, 10), dtype=complex),
    )
    assert data.validate() is True


def test_validate_inconsistent_shapes():
    data = PolarimetricData(
        hh=np.ones((10, 10), dtype=complex),
        hv=np.ones((8, 8), dtype=complex),
    )
    with pytest.raises(ValueError, match="Inconsistent channel shapes"):
        data.validate()
