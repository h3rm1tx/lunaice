from __future__ import annotations

from enum import IntEnum

import numpy as np


class ScatteringClass(IntEnum):
    SURFACE = 1
    DIPOLE = 2
    DIHEDRAL = 3
    VOLUME_SURFACE = 4
    VOLUME_DIPOLE = 5
    VOLUME_DIHEDRAL = 6
    RANDOM = 7
    UNDEFINED = 0


NAMES = {
    ScatteringClass.SURFACE: "Surface Scattering",
    ScatteringClass.DIPOLE: "Dipole Scattering",
    ScatteringClass.DIHEDRAL: "Dihedral / Double-Bounce",
    ScatteringClass.VOLUME_SURFACE: "Volume + Surface",
    ScatteringClass.VOLUME_DIPOLE: "Volume + Dipole",
    ScatteringClass.VOLUME_DIHEDRAL: "Volume + Dihedral",
    ScatteringClass.RANDOM: "Random / High Entropy",
    ScatteringClass.UNDEFINED: "Undefined",
}


class CloudePottierClassifier:
    def __init__(self, entropy_low: float = 0.5, entropy_high: float = 0.9):
        self.entropy_low = entropy_low
        self.entropy_high = entropy_high

    def classify(self, entropy: np.ndarray, alpha_deg: np.ndarray) -> np.ndarray:
        h, w = entropy.shape
        classes = np.full((h, w), ScatteringClass.UNDEFINED, dtype=np.int8)
        low_e = entropy <= self.entropy_low
        mid_e = (entropy > self.entropy_low) & (entropy <= self.entropy_high)
        high_e = entropy > self.entropy_high
        classes[low_e & (alpha_deg <= 42.5)] = ScatteringClass.SURFACE
        classes[low_e & (alpha_deg > 42.5) & (alpha_deg <= 47.5)] = ScatteringClass.DIPOLE
        classes[low_e & (alpha_deg > 47.5)] = ScatteringClass.DIHEDRAL
        classes[mid_e & (alpha_deg <= 42.5)] = ScatteringClass.VOLUME_SURFACE
        classes[mid_e & (alpha_deg > 42.5) & (alpha_deg <= 47.5)] = ScatteringClass.VOLUME_DIPOLE
        classes[mid_e & (alpha_deg > 47.5)] = ScatteringClass.VOLUME_DIHEDRAL
        classes[high_e] = ScatteringClass.RANDOM
        return classes

    @staticmethod
    def class_name(c: int) -> str:
        return NAMES.get(ScatteringClass(c), "Unknown")
