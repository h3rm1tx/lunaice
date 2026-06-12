from __future__ import annotations

import numpy as np

from lunaice.labeling.data_models import IceLabel
from lunaice.labeling.functions.base import BaseLabelingFunction, LabelFunctionResult
from lunaice.ice.classification.cloude_pottier import ScatteringClass


class LF_Scattering(BaseLabelingFunction):
    def __init__(self):
        super().__init__("scattering")
        self._init_zone_map()

    def _init_zone_map(self) -> None:
        self.zone_labels: dict[ScatteringClass, tuple[IceLabel, float, str]] = {
            ScatteringClass.SURFACE: (IceLabel.NO_ICE, 0.1, "Surface scattering: no ice"),
            ScatteringClass.DIPOLE: (IceLabel.POSSIBLE, 0.4, "Dipole: possible ice"),
            ScatteringClass.DIHEDRAL: (IceLabel.LIKELY, 0.7, "Dihedral: ice-favorable"),
            ScatteringClass.VOLUME_SURFACE: (IceLabel.POSSIBLE, 0.4, "Volume+Surface: weak"),
            ScatteringClass.VOLUME_DIPOLE: (IceLabel.LIKELY, 0.55, "Volume+Dipole: moderate"),
            ScatteringClass.VOLUME_DIHEDRAL: (IceLabel.HIGH_CONFIDENCE, 0.75, "Volume+Dihedral: strong ice"),
            ScatteringClass.RANDOM: (IceLabel.NO_ICE, 0.1, "Random: no ice"),
            ScatteringClass.UNDEFINED: (IceLabel.NO_ICE, 0.0, "Undefined: no data"),
        }

    def compute(self, scattering_class: np.ndarray, **kwargs) -> LabelFunctionResult:
        label = np.full_like(scattering_class, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        confidence = np.zeros_like(scattering_class, dtype=np.float32)
        rules = []

        for sc_class, (lbl, conf, _) in self.zone_labels.items():
            mask = scattering_class == int(sc_class)
            label[mask] = int(lbl)
            confidence[mask] = conf

        label[np.isnan(scattering_class.astype(np.float32))] = IceLabel.NO_ICE
        confidence[np.isnan(scattering_class.astype(np.float32))] = 0.0

        rules_used = list(set(r for _, (_, _, r) in self.zone_labels.items()))
        return LabelFunctionResult(name=self.name, label=label, confidence=confidence, rules=rules_used)
