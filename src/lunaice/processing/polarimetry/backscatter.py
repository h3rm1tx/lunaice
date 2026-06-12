from __future__ import annotations

import logging

import numpy as np

from lunaice.models.schemas import PolarimetricData, DecompositionProducts

logger = logging.getLogger(__name__)


class BackscatterCoefficient:
    def __init__(self, incidence_angle_deg: float = 30.0):
        self.incidence_angle_rad = np.radians(incidence_angle_deg)

    def compute(self, data: PolarimetricData, products: DecompositionProducts) -> DecompositionProducts:
        pols = {"sigma_hh": data.hh, "sigma_hv": data.hv, "sigma_vv": data.vv}
        for attr, arr in pols.items():
            if arr is not None:
                sigma = np.abs(arr) ** 2
                setattr(products, attr, sigma.astype(np.float32))
                gamma = sigma * np.cos(self.incidence_angle_rad)
                gamma_attr = attr.replace("sigma", "gamma")
                setattr(products, gamma_attr, gamma.astype(np.float32))
        if data.is_quad_pol:
            products.span = (
                np.abs(data.hh) ** 2 + np.abs(data.hv) ** 2 +
                np.abs(data.vh) ** 2 + np.abs(data.vv) ** 2
            ).astype(np.float32)
        logger.info("Backscatter coefficients computed")
        return products
