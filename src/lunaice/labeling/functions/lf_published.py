from __future__ import annotations

import numpy as np

from lunaice.labeling.data_models import IceLabel
from lunaice.labeling.functions.base import BaseLabelingFunction, LabelFunctionResult


class LF_PublishedRegions(BaseLabelingFunction):
    def __init__(self, known_regions: list[tuple[str, float, float, float]] | None = None):
        super().__init__("published_regions")
        self.known_regions = known_regions or self._default_regions()

    @staticmethod
    def _default_regions() -> list[tuple[str, float, float, float]]:
        return [
            ("Faustini", 77.8, -87.1, 20000),
            ("Cabeus", -34.0, -84.9, 20000),
            ("Haworth", -4.5, -87.5, 15000),
            ("Shoemaker", 44.9, -88.1, 15000),
            ("Shackleton", 0.0, -89.9, 20000),
            ("de Gerlache", 87.5, -88.5, 10000),
            ("Sverdrup", 108.5, -88.6, 10000),
            ("Slater", 95.8, -88.1, 10000),
            ("Wiechert", 43.8, -84.0, 15000),
            ("Amundsen", 82.8, -84.4, 20000),
        ]

    def compute(self, lon: np.ndarray, lat: np.ndarray, **kwargs) -> LabelFunctionResult:
        label = np.full(lon.shape, fill_value=IceLabel.NO_ICE, dtype=np.int8)
        confidence = np.zeros_like(lon, dtype=np.float32)

        rules_used = []
        for name, region_lon, region_lat, radius_m in self.known_regions:
            dist = self._haversine(lon, lat, region_lon, region_lat)
            inside = dist < radius_m
            label[inside] = np.maximum(label[inside], int(IceLabel.HIGH_CONFIDENCE))
            dist_factor = np.clip(1.0 - dist[inside] / radius_m, 0.3, 0.95)
            confidence[inside] = np.maximum(confidence[inside], dist_factor * 0.8)
            if np.any(inside):
                rules_used.append(f"Inside {name} published candidate region -> HIGH_CONFIDENCE")

        label[np.isnan(lon) | np.isnan(lat)] = IceLabel.NO_ICE
        confidence[np.isnan(lon) | np.isnan(lat)] = 0.0

        if not rules_used:
            rules_used.append("Outside all known published candidate regions -> NO_ICE")

        return LabelFunctionResult(
            name=self.name, label=label, confidence=confidence,
            rules=rules_used,
        )

    @staticmethod
    def _haversine(lon1: np.ndarray, lat1: np.ndarray,
                   lon2: float, lat2: float) -> np.ndarray:
        R = 1737400.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
