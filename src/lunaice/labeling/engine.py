from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from lunaice.cube import CubeConfig, PolarisDataCube
from lunaice.labeling.config import LabelingConfig
from lunaice.labeling.data_models import FusedLabelResult, IceLabel, LabelSource
from lunaice.labeling.functions import (
    LF_CPR,
    LF_DOP,
    LF_PSR,
    LF_MultiFrequency,
    LF_Scattering,
    LF_Roughness,
    LF_Illumination,
    LF_PublishedRegions,
)
from lunaice.labeling.fusion.majority import MajorityVoteFusion
from lunaice.labeling.fusion.weighted import WeightedVoteFusion
from lunaice.labeling.fusion.label_model import SnorkelLabelModel
from lunaice.labeling.conflict import ConflictResolver
from lunaice.labeling.validation.validator import LabelValidator
from lunaice.labeling.validation.report import LabelQualityReport
from lunaice.labeling.output.writers import (
    LabelGeoTIFFWriter,
    LabelGeoJSONWriter,
    LabelStatisticsReport,
    DisagreementMapWriter,
)

logger = logging.getLogger("lunaice.labeling")


class LabelingEngine:
    def __init__(self, config: LabelingConfig):
        self.config = config
        self.config.resolve()
        self._setup_logging()

    def _setup_logging(self) -> None:
        level = getattr(logging, self.config.logging_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    def _load_data(self) -> dict:
        logger.info("Loading data from cube: %s", self.config.cube_path)
        cube_cfg = CubeConfig(cube_path=self.config.cube_path)
        cube = PolarisDataCube(cube_cfg)
        ds = cube.ds

        var_names = ["cpr", "dop", "entropy", "alpha_deg", "anisotropy",
                     "sigma_hh", "sigma_hv", "sigma_vv",
                     "elevation", "slope", "incidence_angle",
                     "psr_mask", "illumination", "lon", "lat"]
        v = {}
        for var in var_names:
            if var in ds.data_vars or var in ds.coords:
                arr = ds[var]
                vals = arr.compute().values if hasattr(arr.data, "compute") else arr.values
                v[var] = vals
                logger.debug("  Loaded %s: shape=%s", var, vals.shape)

        h = next(arr.shape[0] for arr in v.values() if arr is not None and arr.ndim >= 2)
        w = next(arr.shape[1] for arr in v.values() if arr is not None and arr.ndim >= 2)

        if "lon" not in v or "lat" not in v:
            xs = ds.x.values if "x" in ds.coords else np.arange(w)
            ys = ds.y.values if "y" in ds.coords else np.arange(h)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            from lunaice.cube.spatial import xy_to_lonlat
            lon_arr, lat_arr = xy_to_lonlat(xx.ravel(), yy.ravel())
            v["lon"] = lon_arr.reshape(h, w)
            v["lat"] = lat_arr.reshape(h, w)

        v["_cube"] = cube
        v["_h"] = h
        v["_w"] = w
        return v

    def _blank(self, h: int, w: int) -> np.ndarray:
        return np.full((h, w), np.nan, dtype=np.float32)

    def _compute_all_lfs(self, v: dict, h: int = 1, w: int = 1) -> dict[str, LabelSource]:
        blank = self._blank(h, w)

        cpr_l = v.get("cpr", blank.copy())
        cpr_s = blank.copy() if "sigma_hh" not in v or v["sigma_hh"] is None else v.get("sigma_hh", blank.copy())
        dop_l = v.get("dop", blank.copy())
        dop_s = blank.copy()
        psr_mask = v.get("psr_mask", None)
        entropy = v.get("entropy", None)
        alpha_deg = v.get("alpha_deg", None)
        slope = v.get("slope", None)
        illumination = v.get("illumination", None)
        lon = v.get("lon", blank.copy())
        lat = v.get("lat", blank.copy())

        sources: dict[str, LabelSource] = {}

        logger.info("LF_1: CPR")
        lf_cpr = LF_CPR(threshold_low=self.config.thresholds.cpr_low,
                         threshold_high=self.config.thresholds.cpr_high)
        cpr_result = lf_cpr.compute(cpr_l=cpr_l, cpr_s=cpr_s)
        sources["cpr"] = LabelSource(name="cpr", label=cpr_result.label,
                                      confidence=cpr_result.confidence, rules=cpr_result.rules)

        logger.info("LF_2: DOP")
        lf_dop = LF_DOP(threshold_low=self.config.thresholds.dop_low,
                         threshold_high=self.config.thresholds.dop_high)
        dop_result = lf_dop.compute(dop_l=dop_l, dop_s=dop_s)
        sources["dop"] = LabelSource(name="dop", label=dop_result.label,
                                      confidence=dop_result.confidence, rules=dop_result.rules)

        logger.info("LF_3: PSR")
        lf_psr = LF_PSR()
        psr_arr = psr_mask if psr_mask is not None else blank
        psr_result = lf_psr.compute(psr_mask=psr_arr)
        sources["psr"] = LabelSource(name="psr", label=psr_result.label,
                                      confidence=psr_result.confidence, rules=psr_result.rules)

        logger.info("LF_4: Multi-Frequency")
        lf_mf = LF_MultiFrequency()
        mf_result = lf_mf.compute(cpr_l=cpr_l, cpr_s=cpr_s)
        sources["multi_frequency"] = LabelSource(name="multi_frequency", label=mf_result.label,
                                                  confidence=mf_result.confidence, rules=mf_result.rules)

        logger.info("LF_5: Scattering")
        lf_scat = LF_Scattering()
        entropy_arr = entropy if entropy is not None else blank
        alpha_arr = alpha_deg if alpha_deg is not None else blank
        from lunaice.ice.classification.cloude_pottier import CloudePottierClassifier
        classifier = CloudePottierClassifier()
        scattering_class = classifier.classify(entropy_arr, alpha_arr)
        scat_result = lf_scat.compute(scattering_class=scattering_class)
        sources["scattering"] = LabelSource(name="scattering", label=scat_result.label,
                                             confidence=scat_result.confidence, rules=scat_result.rules)

        logger.info("LF_6: Roughness")
        lf_rough = LF_Roughness(threshold_low=self.config.thresholds.roughness_low,
                                 threshold_high=self.config.thresholds.roughness_high)
        slope_arr = slope if slope is not None else blank
        rough_result = lf_rough.compute(slope=slope_arr)
        sources["roughness"] = LabelSource(name="roughness", label=rough_result.label,
                                            confidence=rough_result.confidence, rules=rough_result.rules)

        logger.info("LF_7: Illumination")
        lf_illum = LF_Illumination()
        illum_result = lf_illum.compute(illumination=illumination, psr_mask=psr_arr)
        sources["illumination"] = LabelSource(name="illumination", label=illum_result.label,
                                               confidence=illum_result.confidence, rules=illum_result.rules)

        logger.info("LF_8: Published Regions")
        lf_pub = LF_PublishedRegions()
        pub_result = lf_pub.compute(lon=lon, lat=lat)
        sources["published_regions"] = LabelSource(name="published_regions", label=pub_result.label,
                                                    confidence=pub_result.confidence, rules=pub_result.rules)

        return sources

    def _fuse(self, sources: dict[str, LabelSource]) -> FusedLabelResult:
        method = self.config.label_fusion_method
        logger.info("Fusing labels using method: %s", method)

        if method == "majority":
            fusor = MajorityVoteFusion()
        elif method == "snorkel":
            fusor = SnorkelLabelModel()
        else:
            fusor = WeightedVoteFusion(self.config.weights)

        result = fusor.fuse(sources)
        logger.info("Fusion complete: %d sources", result.n_sources)
        return result

    def _resolve_conflicts(self, fused: FusedLabelResult) -> tuple[np.ndarray, np.ndarray]:
        logger.info("Resolving conflicts (threshold: %d)", self.config.conflict_threshold)
        resolver = ConflictResolver(conflict_threshold=self.config.conflict_threshold)
        conflict_result = resolver.analyze(fused)
        resolved_labels = resolver.resolve(fused, conflict_result)
        logger.info("Conflict pixels: %d (%.2f%%)",
                     conflict_result.n_conflict_pixels(),
                     conflict_result.disagreement_rate() * 100)

        valid = ~np.isnan(fused.fused_label.astype(np.float32))
        disagreement_map = np.zeros_like(conflict_result.disagreement_count, dtype=np.float32)
        disagreement_map[valid] = (
            conflict_result.disagreement_count[valid].astype(np.float32)
            / max(fused.n_sources, 1)
        )
        return resolved_labels, disagreement_map

    def _validate(self, fused: FusedLabelResult, resolved_labels: np.ndarray,
                  disagreement_map: np.ndarray, lon: np.ndarray, lat: np.ndarray
                  ) -> list:
        logger.info("Validating labels against known candidate regions")
        validator = LabelValidator()
        metrics = []
        for name, rlon, rlat, radius in validator.known_regions:
            m = validator.evaluate_region(
                labels=resolved_labels,
                confidence=fused.fused_confidence,
                disagreement=disagreement_map,
                lon=lon, lat=lat,
                region_lon=rlon, region_lat=rlat,
                region_radius_m=radius,
                name=name,
            )
            if m.n_pixels > 0:
                metrics.append(m)
                logger.debug("  %s: %d pixels, ice_fav=%.2f%%",
                              name, m.n_pixels, m.ice_favorable_fraction * 100)
        return metrics

    def _write_outputs(self, fused: FusedLabelResult, resolved_labels: np.ndarray,
                       disagreement_map: np.ndarray, metrics: list,
                       lon: np.ndarray, lat: np.ndarray) -> dict:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        gtiff = LabelGeoTIFFWriter(out_dir)
        paths = {}

        logger.info("Writing weak_labels.tif")
        paths["weak_labels"] = str(gtiff.write_labels(resolved_labels, "weak_labels"))

        logger.info("Writing label_confidence.tif")
        paths["label_confidence"] = str(gtiff.write_float(fused.fused_confidence, "label_confidence"))

        logger.info("Writing disagreement_map.tif")
        dis_writer = DisagreementMapWriter(out_dir)
        paths["disagreement_map"] = str(dis_writer.write(disagreement_map))

        logger.info("Writing ice_regions.geojson")
        geo_writer = LabelGeoJSONWriter(out_dir)
        paths["ice_regions"] = str(geo_writer.write_ice_regions(
            resolved_labels, fused.fused_confidence, lon, lat,
        ))

        if self.config.generate_report:
            logger.info("Generating label statistics report")
            stats_writer = LabelStatisticsReport(out_dir)
            paths["label_statistics"] = str(stats_writer.generate(
                resolved_labels, fused.fused_confidence, disagreement_map,
            ))

            logger.info("Generating label quality report")
            total_valid = int(np.sum(~np.isnan(resolved_labels.astype(np.float32))))
            ice_fav = int(np.sum(resolved_labels >= int(IceLabel.LIKELY)) if total_valid > 0 else 0)
            high_conf = int(np.sum(fused.fused_confidence >= 0.7) if total_valid > 0 else 0)
            dist: dict[str, int] = {}
            for label_val in IceLabel:
                cnt = int(np.sum(resolved_labels == int(label_val)))
                from lunaice.labeling.data_models import LABEL_NAMES
                dist[LABEL_NAMES[label_val]] = cnt

            global_stats = {
                "total_pixels": total_valid,
                "ice_favorable_pixels": ice_fav,
                "ice_favorable_pct": 100.0 * ice_fav / max(total_valid, 1),
                "high_confidence_pixels": high_conf,
                "high_confidence_pct": 100.0 * high_conf / max(total_valid, 1),
                "n_sources": fused.n_sources,
                "disagreement_rate": float(np.nanmean(disagreement_map)),
                "fusion_method": self.config.label_fusion_method,
                "label_distribution": dist,
                "label_pcts": {k: 100.0 * v / max(total_valid, 1) for k, v in dist.items()},
            }
            report = LabelQualityReport(out_dir)
            paths["label_quality_report"] = str(report.generate(metrics, global_stats))

        return paths

    def run(self) -> dict:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("POLARIS Labeling Engine v0.1.0")
        logger.info("=" * 60)

        v = self._load_data()
        cube: PolarisDataCube = v.pop("_cube")
        h, w = v.pop("_h"), v.pop("_w")
        lon = v.get("lon", self._blank(h, w))
        lat = v.get("lat", self._blank(h, w))

        sources = self._compute_all_lfs(v, h=h, w=w)

        fused = self._fuse(sources)

        resolved_labels, disagreement_map = self._resolve_conflicts(fused)

        metrics = self._validate(fused, resolved_labels, disagreement_map, lon, lat)

        paths = self._write_outputs(fused, resolved_labels, disagreement_map, metrics, lon, lat)

        elapsed = time.time() - t0
        logger.info("Labeling Engine finished in %.2f s", elapsed)

        cube.close()

        return {
            "sources": {k: v for k, v in sources.items()},
            "fused_label": fused.fused_label,
            "fused_confidence": fused.fused_confidence,
            "resolved_labels": resolved_labels,
            "disagreement_map": disagreement_map,
            "metrics": metrics,
            "output_paths": paths,
            "elapsed_s": elapsed,
        }
