from __future__ import annotations

import logging

import numpy as np

from lunaice.models.schemas import CoherencyMatrix, PolarimetricData

logger = logging.getLogger(__name__)


class CoherencyMatrixBuilder:
    def __init__(self, multilook_range: int = 1, multilook_azimuth: int = 1):
        self.multilook = (multilook_range, multilook_azimuth)

    def _multilook(self, arr: np.ndarray) -> np.ndarray:
        r, a = self.multilook
        if r == 1 and a == 1:
            return arr
        hr, ha = arr.shape[0] // r * r, arr.shape[1] // a * a
        cropped = arr[:hr, :ha]
        return cropped.reshape(hr // r, r, ha // a, a).mean(axis=(1, 3))

    def build(self, data: PolarimetricData) -> CoherencyMatrix:
        if not data.is_quad_pol:
            raise ValueError("Quad-pol (HH, HV, VH, VV) required for coherency matrix")
        hh = self._multilook(data.hh)
        hv = self._multilook(data.hv)
        vh = self._multilook(data.vh)
        vv = self._multilook(data.vv)

        k = np.stack([hh + vv, hh - vv, hv + vh], axis=-1) / np.sqrt(2.0)
        k_conj = np.conj(k)
        t11 = (k[:, :, 0] * k_conj[:, :, 0]).real
        t12 = k[:, :, 0] * k_conj[:, :, 1]
        t13 = k[:, :, 0] * k_conj[:, :, 2]
        t22 = (k[:, :, 1] * k_conj[:, :, 1]).real
        t23 = k[:, :, 1] * k_conj[:, :, 2]
        t33 = (k[:, :, 2] * k_conj[:, :, 2]).real
        logger.info("Coherency matrix (T3) built: shape=%s", t11.shape)
        return CoherencyMatrix(
            t11=t11, t12=t12, t13=t13,
            t21=np.conj(t12), t22=t22, t23=t23,
            t31=np.conj(t13), t32=np.conj(t23), t33=t33,
        )
