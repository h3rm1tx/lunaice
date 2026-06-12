from __future__ import annotations

import logging

import numpy as np

from lunaice.models.schemas import PolarimetricData, DecompositionProducts

logger = logging.getLogger(__name__)


class CircularPolarizationRatio:
    def compute(self, data: PolarimetricData, products: DecompositionProducts) -> DecompositionProducts:
        if data.is_quad_pol:
            sc = np.array([
                [data.hh, data.hv],
                [data.vh, data.vv],
            ])
            rcp_tx = (sc[0, 0] - 1j * sc[1, 0]) / np.sqrt(2)
            lcp_tx = (sc[0, 0] + 1j * sc[1, 0]) / np.sqrt(2)
            sc_lcp = np.array([
                [lcp_tx, (sc[0, 1] + 1j * sc[1, 1]) / np.sqrt(2)],
                [rcp_tx, (sc[0, 1] - 1j * sc[1, 1]) / np.sqrt(2)],
            ])
            same_sense = np.abs(sc_lcp[0, 0]) ** 2 + np.abs(sc_lcp[1, 1]) ** 2
            opp_sense = np.abs(sc_lcp[0, 1]) ** 2 + np.abs(sc_lcp[1, 0]) ** 2
            cpr = same_sense / (opp_sense + 1e-30)
        elif data.metadata and data.metadata.polarization_mode:
            logger.warning("CPR from non-quad-pol data not fully supported; using dummy")
            hh_pow = np.abs(data.hh) ** 2 if data.hh is not None else 0
            vv_pow = np.abs(data.vv) ** 2 if data.vv is not None else 0
            cpr = hh_pow / (vv_pow + 1e-30) if vv_pow is not None else np.ones_like(hh_pow)
        else:
            raise ValueError("Insufficient polarization data for CPR computation")
        products.cpr = cpr.astype(np.float32)
        logger.info("CPR computed: range [%.3f, %.3f]", np.nanmin(cpr), np.nanmax(cpr))
        return products
