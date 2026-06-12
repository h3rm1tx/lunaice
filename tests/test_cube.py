from __future__ import annotations

import numpy as np
import pytest

from lunaice.cube.config import CubeConfig, GridConfig, SourceConfig
from lunaice.cube.craters import CraterCatalog, CraterRecord
from lunaice.cube.spatial import (
    lonlat_to_pixel_indices,
    lonlat_to_xy,
    xy_to_lonlat,
    xy_to_pixel_indices,
)


class TestGridConfig:
    def test_dimensions(self):
        g = GridConfig(bounds_left=-100, bounds_right=100, bounds_bottom=-100, bounds_top=100, pixel_size_m=10)
        assert g.width == 20
        assert g.height == 20

    def test_proj4_default(self):
        g = GridConfig()
        assert "stere" in g.crs_proj4
        assert "lat_0=-90" in g.crs_proj4


class TestSourceConfig:
    def test_validate_none(self):
        s = SourceConfig()
        assert s.validate() == []

    def test_validate_missing(self):
        s = SourceConfig(dfsar_path="/nonexistent/path")
        missing = s.validate()
        assert len(missing) > 0


class TestCubeConfig:
    def test_defaults(self):
        cfg = CubeConfig()
        assert cfg.chunk_size == 256
        assert cfg.include_dfsar is True
        assert cfg.pyramid_levels == 5

    def test_from_yaml_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CubeConfig.from_yaml(tmp_path / "nonexistent.yaml")


class TestCraterCatalog:
    def test_builtin_craters(self):
        cat = CraterCatalog()
        assert len(cat.list_craters()) > 5
        faustini = cat.lookup("Faustini")
        assert faustini is not None
        assert faustini.diameter_km == 42.0
        assert faustini.x_m != 0.0

    def test_lookup_case_insensitive(self):
        cat = CraterCatalog()
        assert cat.lookup("shackleton") is not None
        assert cat.lookup("SHACKLETON") is not None

    def test_lookup_missing(self):
        cat = CraterCatalog()
        assert cat.lookup("nonexistent_crater") is None

    def test_search_near(self):
        cat = CraterCatalog()
        cab = cat.lookup("Cabeus")
        assert cab is not None
        near = cat.search_near(cab.x_m, cab.y_m, radius_m=50000)
        assert len(near) >= 1
        assert any(r.name.lower() == "cabeus" for r in near)


class TestSpatial:
    def test_xy_lonlat_roundtrip(self):
        x_in, y_in = np.array([10000.0]), np.array([-20000.0])
        lon, lat = xy_to_lonlat(x_in, y_in)
        x_out, y_out = lonlat_to_xy(lon, lat)
        assert np.allclose(x_in, x_out, atol=1.0)
        assert np.allclose(y_in, y_out, atol=1.0)

    def test_pixel_indices(self):
        g = GridConfig(
            bounds_left=0, bounds_right=100,
            bounds_bottom=0, bounds_top=100,
            pixel_size_m=10,
        )
        row, col = xy_to_pixel_indices(25, 75, g)
        assert col == 2
        assert row == 2
