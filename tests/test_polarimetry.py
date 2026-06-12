from __future__ import annotations

import numpy as np
import pytest

from lunaice.models.schemas import PolarimetricData, DecompositionProducts
from lunaice.processing.polarimetry import (
    BackscatterCoefficient,
    CircularPolarizationRatio,
    CloudePottierDecomposition,
    CoherencyMatrixBuilder,
    DegreeOfPolarization,
)


class TestCoherencyMatrixBuilder:
    def test_build_quad_pol(self, sample_slc_data: PolarimetricData):
        builder = CoherencyMatrixBuilder()
        t3 = builder.build(sample_slc_data)
        assert t3.t11.shape == sample_slc_data.shape
        assert np.all(np.isfinite(t3.t11))
        assert t3.t11.dtype == np.float64

    def test_raises_on_partial(self):
        data = PolarimetricData(hh=np.ones((4, 4), dtype=complex))
        with pytest.raises(ValueError, match="Quad-pol"):
            CoherencyMatrixBuilder().build(data)


class TestCloudePottierDecomposition:
    def test_decompose(self, sample_slc_data: PolarimetricData):
        t3 = CoherencyMatrixBuilder().build(sample_slc_data)
        cp = CloudePottierDecomposition()
        products = cp.decompose(t3)
        assert products.entropy is not None
        assert products.alpha_deg is not None
        assert products.anisotropy is not None
        assert np.all((products.entropy >= 0) & (products.entropy <= 1))
        assert np.all((products.alpha_deg >= 0) & (products.alpha_deg <= 90))
        assert np.all((products.anisotropy >= 0) & (products.anisotropy <= 1))


class TestCircularPolarizationRatio:
    def test_cpr_quad_pol(self, sample_slc_data: PolarimetricData):
        products = DecompositionProducts()
        cpr_comp = CircularPolarizationRatio()
        products = cpr_comp.compute(sample_slc_data, products)
        assert products.cpr is not None
        assert np.all(products.cpr >= 0)
        assert products.cpr.shape == sample_slc_data.shape


class TestDegreeOfPolarization:
    def test_dop_quad_pol(self, sample_slc_data: PolarimetricData):
        products = DecompositionProducts()
        dop_comp = DegreeOfPolarization()
        products = dop_comp.compute(sample_slc_data, products)
        assert products.dop is not None
        assert np.all((products.dop >= 0) & (products.dop <= 1))


class TestBackscatterCoefficient:
    def test_compute(self, sample_slc_data: PolarimetricData):
        products = DecompositionProducts()
        bsc = BackscatterCoefficient(incidence_angle_deg=30.0)
        products = bsc.compute(sample_slc_data, products)
        assert products.sigma_hh is not None
        assert products.gamma_hh is not None
        assert products.span is not None
        assert np.all(products.sigma_hh >= 0)
