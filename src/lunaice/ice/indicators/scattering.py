from __future__ import annotations

import numpy as np

from lunaice.ice.indicators.base import BaseIndicator, IndicatorResult


class ScatteringIndicator(BaseIndicator):
    def __init__(self, name: str = "scattering"):
        super().__init__(name)

    def compute(
        self,
        entropy: np.ndarray,
        alpha_deg: np.ndarray,
        anisotropy: np.ndarray,
        **kwargs,
    ) -> IndicatorResult:
        ice_favored = np.full(entropy.shape, np.nan, dtype=np.float32)
        low_e_mask = entropy <= 0.5
        mid_e_mask = (entropy > 0.5) & (entropy <= 0.9)
        high_e_mask = entropy > 0.9

        surface_mask = low_e_mask & (alpha_deg <= 42.5)
        ice_favored[surface_mask] = 0.1

        dihedral_mask = low_e_mask & (alpha_deg > 47.5)
        ice_favored[dihedral_mask] = 0.8

        dipole_mask = low_e_mask & (alpha_deg > 42.5) & (alpha_deg <= 47.5)
        ice_favored[dipole_mask] = 0.4

        vol_surface = mid_e_mask & (alpha_deg <= 42.5)
        ice_favored[vol_surface] = 0.3
        vol_dipole = mid_e_mask & (alpha_deg > 42.5) & (alpha_deg <= 47.5)
        ice_favored[vol_dipole] = 0.5
        vol_dihedral = mid_e_mask & (alpha_deg > 47.5)
        ice_favored[vol_dihedral] = 0.7

        ice_favored[high_e_mask] = 0.2

        return IndicatorResult(
            name=self.name,
            score=ice_favored,
            raw_value=np.stack([entropy, alpha_deg, anisotropy], axis=-1).astype(np.float32),
            metadata={"description": "H/alpha scattering classification"},
        )
