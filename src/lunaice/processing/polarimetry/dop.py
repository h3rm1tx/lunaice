from __future__ import annotations

import logging

import numpy as np

from lunaice.models.schemas import PolarimetricData, DecompositionProducts

logger = logging.getLogger(__name__)


class DegreeOfPolarization:
    def compute(self, data: PolarimetricData, products: DecompositionProducts) -> DecompositionProducts:
        if data.is_quad_pol:
            g11 = (np.abs(data.hh) ** 2 + np.abs(data.hv) ** 2 +
                   np.abs(data.vh) ** 2 + np.abs(data.vv) ** 2).real
            g22 = (np.abs(data.hh) ** 2 - np.abs(data.hv) ** 2 +
                   np.abs(data.vh) ** 2 - np.abs(data.vv) ** 2).real
            g33 = (2 * (data.hh * np.conj(data.hv) + data.vh * np.conj(data.vv))).real
            g44 = (2j * (data.hh * np.conj(data.hv) - data.vh * np.conj(data.vv))).real
            dop = np.sqrt(g22 ** 2 + g33 ** 2 + g44 ** 2) / (g11 + 1e-30)
        else:
            logger.warning("DOP from non-quad-pol; using approximate method")
            g11 = np.abs(data.hh) ** 2 + np.abs(data.vv) ** 2 if data.hh is not None else 1
            g22 = np.abs(data.hh) ** 2 - np.abs(data.vv) ** 2 if data.hh is not None else 0
            dop = np.sqrt(g22 ** 2) / (g11 + 1e-30)
        products.dop = np.clip(dop, 0.0, 1.0).astype(np.float32)
        logger.info("DOP computed: range [%.3f, %.3f]", np.nanmin(products.dop), np.nanmax(products.dop))
        return products
