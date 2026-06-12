from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lunaice.labeling.data_models import IceLabel, LABEL_NAMES


@dataclass
class LabelRegionMetrics:
    name: str
    n_pixels: int
    label_distribution: dict[str, int]
    mean_confidence: float
    mean_disagreement: float
    high_conf_fraction: float
    ice_favorable_fraction: float


class LabelValidator:
    def __init__(self):
        self.known_regions = self._default_regions()

    @staticmethod
    def _default_regions() -> list[tuple[str, float, float, float]]:
        return [
            ("Faustini_Interior", 77.8, -87.1, 15000),
            ("Faustini_Rim", 77.8, -87.1, 5000),
            ("Cabeus_Interior", -34.0, -84.9, 15000),
            ("Cabeus_Rim", -34.0, -84.9, 5000),
            ("Haworth", -4.5, -87.5, 15000),
            ("Shoemaker", 44.9, -88.1, 15000),
            ("Shackleton_Floor", 0.0, -89.9, 10000),
            ("Shackleton_Rim", 0.0, -89.9, 5000),
            ("de_Gerlache", 87.5, -88.5, 10000),
            ("Amundsen", 82.8, -84.4, 20000),
        ]

    def evaluate_region(
        self,
        labels: np.ndarray,
        confidence: np.ndarray,
        disagreement: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        region_lon: float,
        region_lat: float,
        region_radius_m: float,
        name: str,
    ) -> LabelRegionMetrics:
        dist = self._haversine(lon, lat, region_lon, region_lat)
        mask = dist < region_radius_m
        mask = mask & ~np.isnan(labels.astype(np.float32))

        n_pixels = int(np.sum(mask))
        if n_pixels == 0:
            return LabelRegionMetrics(name=name, n_pixels=0, label_distribution={},
                                       mean_confidence=0.0, mean_disagreement=0.0,
                                       high_conf_fraction=0.0, ice_favorable_fraction=0.0)

        region_labels = labels[mask]
        region_conf = confidence[mask]
        region_dis = disagreement[mask]

        distr = {}
        for label_val in IceLabel:
            count = int(np.sum(region_labels == int(label_val)))
            if count > 0:
                distr[LABEL_NAMES[label_val]] = count

        ice_fav = (region_labels >= int(IceLabel.LIKELY)).sum()
        high_conf = (region_conf >= 0.7).sum()

        return LabelRegionMetrics(
            name=name,
            n_pixels=n_pixels,
            label_distribution=distr,
            mean_confidence=float(np.nanmean(region_conf)),
            mean_disagreement=float(np.nanmean(region_dis.astype(np.float32))),
            high_conf_fraction=high_conf / max(n_pixels, 1),
            ice_favorable_fraction=ice_fav / max(n_pixels, 1),
        )

    @staticmethod
    def _haversine(lon1: np.ndarray, lat1: np.ndarray,
                   lon2: float, lat2: float) -> np.ndarray:
        R = 1737400.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
