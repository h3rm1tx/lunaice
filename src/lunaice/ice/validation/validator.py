from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely import wkt

from lunaice.cube.craters import CraterCatalog

logger = logging.getLogger(__name__)


@dataclass
class RegionMetrics:
    name: str
    n_pixels: int
    mean_ice_prior: float
    std_ice_prior: float
    mean_confidence: float
    mean_cpr: float
    mean_dop: float
    psr_fraction: float
    dominant_scattering: str
    high_confidence_fraction: float


class IceValidator:
    def __init__(self, crater_catalog: Optional[CraterCatalog] = None):
        self.catalog = crater_catalog or CraterCatalog()

    def evaluate_region(
        self,
        ice_prior: np.ndarray,
        confidence: np.ndarray,
        cpr: np.ndarray,
        dop: np.ndarray,
        scattering_class: np.ndarray,
        psr_mask: np.ndarray,
        region_mask: np.ndarray,
        region_name: str = "region",
    ) -> RegionMetrics:
        px = region_mask.astype(bool)
        if px.sum() == 0:
            return RegionMetrics(name=region_name, n_pixels=0, mean_ice_prior=0, std_ice_prior=0,
                                 mean_confidence=0, mean_cpr=0, mean_dop=0, psr_fraction=0,
                                 dominant_scattering="N/A", high_confidence_fraction=0)
        valid_ip = ice_prior[px]
        sc_class = scattering_class[px]
        unique, counts = np.unique(sc_class, return_counts=True)
        dominant_idx = unique[np.argmax(counts)] if len(unique) > 0 else 0
        from lunaice.ice.classification.cloude_pottier import NAMES, ScatteringClass
        dom_name = NAMES.get(ScatteringClass(int(dominant_idx)), "Unknown") if dominant_idx in {c.value for c in ScatteringClass} else "Unknown"
        return RegionMetrics(
            name=region_name,
            n_pixels=int(px.sum()),
            mean_ice_prior=float(np.nanmean(valid_ip)),
            std_ice_prior=float(np.nanstd(valid_ip)),
            mean_confidence=float(np.nanmean(confidence[px])),
            mean_cpr=float(np.nanmean(cpr[px])) if cpr is not None else 0,
            mean_dop=float(np.nanmean(dop[px])) if dop is not None else 0,
            psr_fraction=float(np.nanmean(psr_mask[px])),
            dominant_scattering=dom_name,
            high_confidence_fraction=float(np.nanmean((confidence[px] >= 0.7).astype(float))),
        )

    def known_ice_candidate_regions(self) -> list[tuple[str, float, float, float]]:
        return [
            ("Faustini_Interior", 77.0, -87.2, 15000),
            ("Cabeus_Interior", -35.5, -84.9, 30000),
            ("Haworth_Interior", -5.0, -87.5, 20000),
            ("Shoemaker_Floor", 45.0, -88.1, 15000),
            ("Shackleton_Interior", 0.0, -89.9, 10000),
        ]

    def generate_candidate_polygons(
        self,
        ice_prior: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        threshold: float = 0.6,
        min_area_px: int = 50,
    ) -> gpd.GeoDataFrame:
        mask = (ice_prior >= threshold) & (~np.isnan(ice_prior))
        from scipy.ndimage import label
        labeled, n_features = label(mask)
        features = []
        for feat_id in range(1, n_features + 1):
            feat_mask = labeled == feat_id
            if feat_mask.sum() < min_area_px:
                continue
            ys, xs = np.where(feat_mask)
            if len(xs) < 3:
                continue
            x_min, x_max = float(x_coords[xs].min()), float(x_coords[xs].max())
            y_min, y_max = float(y_coords[ys].min()), float(y_coords[ys].max())
            poly = Polygon([
                (x_min, y_min), (x_max, y_min),
                (x_max, y_max), (x_min, y_max), (x_min, y_min),
            ])
            mean_prior = float(np.nanmean(ice_prior[feat_mask]))
            mean_conf = float(np.nanmean(
                # confidence would need to be passed too
                ice_prior[feat_mask]
            ))
            features.append({
                "geometry": poly,
                "ice_prior_mean": round(mean_prior, 4),
                "area_px": int(feat_mask.sum()),
                "label": f"candidate_{feat_id:04d}",
            })
        return gpd.GeoDataFrame(features, crs=None)
