from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from lunaice.cube import CubeConfig, PolarisDataCube
from lunaice.ice.config import IcePriorConfig
from lunaice.ice.indicators.cpr import CPRIndicator
from lunaice.ice.indicators.dop import DOPIndicator
from lunaice.ice.indicators.multi_freq import MultiFrequencyIndicator
from lunaice.ice.indicators.psr import PSRIndicator
from lunaice.ice.indicators.roughness import RoughnessIndicator
from lunaice.ice.indicators.scattering import ScatteringIndicator
from lunaice.ice.scoring.confidence import ConfidenceEngine
from lunaice.ice.scoring.fusion import IcePriorScore
from lunaice.ice.classification.cloude_pottier import CloudePottierClassifier
from lunaice.ice.validation.report import ValidationReport
from lunaice.ice.validation.validator import IceValidator
from lunaice.ice.output.geotiff import IceGeoTIFFWriter
from lunaice.ice.output.geojson import CandidateGeoJSONWriter
from lunaice.ice.output.visualization import IceVisualizationEngine

logger = logging.getLogger("lunaice.ice")


class IcePriorEngine:
    def __init__(self, config: IcePriorConfig):
        self.config = config
        self.config.resolve()
        self._setup_logging()

    def _setup_logging(self) -> None:
        level = getattr(logging, self.config.logging_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    def run(self) -> dict:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("POLARIS Ice Prior Engine v0.1.0")

        cube_cfg = CubeConfig(cube_path=self.config.cube_path)
        cube = PolarisDataCube(cube_cfg)
        ds = cube.ds

        logger.info("Loading variables from cube: %s", list(ds.data_vars))
        v = {}
        for var in ["cpr", "dop", "entropy", "alpha_deg", "anisotropy",
                     "sigma_hh", "sigma_hv", "sigma_vv",
                     "elevation", "slope", "incidence_angle",
                     "psr_mask"]:
            if var in ds.data_vars:
                v[var] = ds[var].compute().values if hasattr(ds[var].data, "compute") else ds[var].values
                logger.debug("  Loaded %s: shape=%s", var, v[var].shape)
        cpr_l = v.get("cpr", None)
        dop_l = v.get("dop", None)
        entropy = v.get("entropy", None)
        alpha_deg = v.get("alpha_deg", None)
        anisotropy = v.get("anisotropy", None)
        slope = v.get("slope", None)
        psr_mask = v.get("psr_mask", None)
        sigma_hh = v.get("sigma_hh", None)

        h = next(arr.shape[0] for arr in v.values() if arr is not None)
        w = next(arr.shape[1] for arr in v.values() if arr is not None)
        blank = np.full((h, w), np.nan, dtype=np.float32)

        if cpr_l is None:
            cpr_l = blank.copy()
        cpr_s = blank.copy()
        if sigma_hh is not None:
            cpr_s = cpr_l.copy()
        dop_s = blank.copy()

        indicators = {}

        logger.info("Computing indicator: CPR")
        cpr_ind = CPRIndicator(threshold=self.config.cpr_threshold)
        indicators["cpr"] = cpr_ind.compute(cpr_l=cpr_l, cpr_s=cpr_s)
        del cpr_ind

        logger.info("Computing indicator: DOP")
        dop_ind = DOPIndicator(threshold=self.config.dop_threshold)
        indicators["dop"] = dop_ind.compute(dop_l=dop_l, dop_s=dop_s)
        del dop_ind

        logger.info("Computing indicator: PSR")
        psr_ind = PSRIndicator()
        indicators["psr"] = psr_ind.compute(psr_mask=psr_mask if psr_mask is not None else blank)
        del psr_ind

        logger.info("Computing indicator: Roughness")
        rough_ind = RoughnessIndicator(percentile=self.config.roughness_percentile)
        indicators["roughness"] = rough_ind.compute(slope=slope if slope is not None else blank)
        del rough_ind

        logger.info("Computing indicator: Multi-Frequency")
        mf_ind = MultiFrequencyIndicator()
        indicators["multi_frequency"] = mf_ind.compute(
            cpr_l=cpr_l, cpr_s=cpr_s,
            sigma_hh_l=sigma_hh if sigma_hh is not None else blank,
            sigma_hh_s=blank,
        )
        del mf_ind

        logger.info("Computing indicator: Scattering")
        scat_ind = ScatteringIndicator()
        indicators["scattering"] = scat_ind.compute(
            entropy=entropy if entropy is not None else blank,
            alpha_deg=alpha_deg if alpha_deg is not None else blank,
            anisotropy=anisotropy if anisotropy is not None else blank,
        )
        del scat_ind

        logger.info("Fusing indicators -> Ice Prior Score")
        fusion = IcePriorScore(self.config.weights)
        ice_prior = fusion.compute(indicators)

        logger.info("Computing confidence")
        conf_engine = ConfidenceEngine()
        indicator_score_map = {k: v.score for k, v in indicators.items()}
        confidence = conf_engine.compute(ice_prior, indicator_score_map)

        logger.info("Classifying scattering type")
        classifier = CloudePottierClassifier(
            entropy_low=self.config.thresholds.entropy_low,
            entropy_high=self.config.thresholds.entropy_high,
        )
        scattering_class = classifier.classify(
            entropy if entropy is not None else blank,
            alpha_deg if alpha_deg is not None else blank,
        )

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Writing GeoTIFF outputs")
        gtiff = IceGeoTIFFWriter(out_dir)
        gtiff.write(ice_prior, "ice_prior")
        gtiff.write(confidence, "confidence")
        gtiff.write_int(scattering_class, "scattering_class")
        for key, result in indicators.items():
            gtiff.write(result.score, f"indicator_{key}")

        logger.info("Generating candidate polygons")
        xs = ds.x.values if "x" in ds.coords else np.arange(w)
        ys = ds.y.values if "y" in ds.coords else np.arange(h)
        validator = IceValidator()
        gdf = validator.generate_candidate_polygons(ice_prior, xs, ys)
        geo_writer = CandidateGeoJSONWriter(out_dir)
        geo_writer.write(gdf)

        logger.info("Validating against known regions")
        metrics = []
        for name, lon, lat, radius in validator.known_ice_candidate_regions():
            from lunaice.cube.spatial import lonlat_to_xy
            cx, cy = lonlat_to_xy(np.array([lon]), np.array([lat]))
            cx, cy = float(cx[0]), float(cy[0])
            dx = int(radius / abs(float(ds.x.values[1] - ds.x.values[0]))) if "x" in ds.coords else 10
            dy = int(radius / abs(float(ds.y.values[1] - ds.y.values[0]))) if "y" in ds.coords else 10
            cx_idx = int((cx - float(ds.x.values[0])) / float(ds.x.values[1] - ds.x.values[0])) if "x" in ds.coords else w // 2
            cy_idx = int((float(ds.y.values[0]) - cy) / abs(float(ds.y.values[1] - ds.y.values[0]))) if "y" in ds.coords else h // 2
            region_mask = np.zeros((h, w), dtype=np.float32)
            x_s = max(0, cx_idx - dx)
            x_e = min(w, cx_idx + dx)
            y_s = max(0, cy_idx - dy)
            y_e = min(h, cy_idx + dy)
            region_mask[y_s:y_e, x_s:x_e] = 1.0
            rm = validator.evaluate_region(
                ice_prior, confidence, cpr_l, dop_l, scattering_class,
                psr_mask if psr_mask is not None else blank, region_mask, name,
            )
            metrics.append(rm)

        report = ValidationReport(out_dir)
        if self.config.generate_report:
            global_stats = {
                "mean_ice_prior": float(np.nanmean(ice_prior)),
                "max_ice_prior": float(np.nanmax(ice_prior)),
                "mean_confidence": float(np.nanmean(confidence)),
                "candidate_regions": len(gdf) if not gdf.empty else 0,
                "valid_pixels": int(np.sum(~np.isnan(ice_prior))),
            }
            report.generate(metrics, global_stats)
        else:
            logger.info("Report generation skipped")

        if self.config.generate_plots:
            logger.info("Generating visualizations")
            viz = IceVisualizationEngine(out_dir)
            viz.histogram(ice_prior, "Ice Prior Score Distribution", "ice_prior_histogram.png")
            viz.histogram(confidence, "Confidence Distribution", "confidence_histogram.png")
            viz.heatmap(ice_prior, "Ice Prior Score Map", "ice_prior_heatmap.png")
            viz.feature_distribution(
                {k: v.score for k, v in indicators.items()},
                "indicator_distributions.png",
            )

        elapsed = time.time() - t0
        logger.info("Ice Prior Engine finished in %.2f s", elapsed)

        result = {
            "ice_prior": ice_prior,
            "confidence": confidence,
            "scattering_class": scattering_class,
            "indicators": {k: v.score for k, v in indicators.items()},
            "metrics": metrics,
            "candidate_gdf": gdf,
            "elapsed_s": elapsed,
        }
        cube.close()
        return result
