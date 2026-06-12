from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from lunaice.config import DFSARConfig, ProcessingConfig, SpeckleFilterConfig
from lunaice.io.reader import DFSARReader
from lunaice.io.writer import GeoTIFFWriter, ReportWriter, ZarrWriter
from lunaice.models.schemas import DecompositionProducts, ProcessingSummary
from lunaice.processing.calibration import PolarimetricCalibrator, RadiometricCalibrator
from lunaice.processing.polarimetry import (
    BackscatterCoefficient,
    CircularPolarizationRatio,
    CloudePottierDecomposition,
    CoherencyMatrixBuilder,
    DegreeOfPolarization,
)
from lunaice.processing.speckle import SpeckleFilter

logger = logging.getLogger("lunaice.pipeline")


class Pipeline:
    def __init__(self, config: DFSARConfig):
        self.config = config
        self.config.resolve()
        self._setup_logging()

    def _setup_logging(self) -> None:
        level = getattr(logging, self.config.logging_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    def run(self) -> ProcessingSummary:
        t0 = time.time()
        errors: list[str] = []
        warnings: list[str] = []
        products_generated: list[str] = []
        proc = self.config.processing

        logger.info("=" * 60)
        logger.info("LUNAICE DFSAR Processing Pipeline v0.1.0")
        logger.info("Input: %s", self.config.input_file)
        logger.info("Band: %s", self.config.band)

        try:
            reader = DFSARReader(self.config.input_file)
            data = reader.read_slc()
            logger.info("Loaded SLC data: shape=%s, quad_pol=%s", data.shape, data.is_quad_pol)
        except Exception as e:
            logger.critical("Failed to read input: %s", e)
            errors.append(f"ReadError: {e}")
            return ProcessingSummary(
                input_file=self.config.input_file,
                output_dir=self.config.output_dir,
                products_generated=[],
                processing_time_s=time.time() - t0,
                config_snapshot=self._config_snapshot(),
                errors=errors,
            )

        if proc.radiometric_calibration:
            try:
                rad_cal = RadiometricCalibrator(
                    cal_constant=data.metadata.calibration_constant if data.metadata else 1.0
                )
                data = rad_cal.calibrate(data)
                products_generated.append("radiometric_calibration")
            except Exception as e:
                warnings.append(f"RadiometricCalibrationWarning: {e}")
                logger.warning("Radiometric calibration failed: %s", e)

        if proc.polarimetric_calibration and data.is_quad_pol:
            try:
                pol_cal = PolarimetricCalibrator()
                data = pol_cal.calibrate(data)
                products_generated.append("polarimetric_calibration")
            except Exception as e:
                warnings.append(f"PolarimetricCalibrationWarning: {e}")
                logger.warning("Polarimetric calibration failed: %s", e)

        if proc.speckle_filter:
            try:
                filt = SpeckleFilter(proc.speckle_filter)
                data = filt.apply(data)
                products_generated.append(f"speckle_filter_{proc.speckle_filter.method}")
            except Exception as e:
                warnings.append(f"SpeckleFilterWarning: {e}")
                logger.warning("Speckle filtering failed: %s", e)

        products = DecompositionProducts()

        if proc.generate_coherency_matrix and data.is_quad_pol:
            try:
                t3_builder = CoherencyMatrixBuilder(
                    multilook_range=proc.multilook_range,
                    multilook_azimuth=proc.multilook_azimuth,
                )
                t3 = t3_builder.build(data)
                products_generated.append("coherency_matrix_T3")
            except Exception as e:
                warnings.append(f"T3Warning: {e}")
                logger.warning("T3 generation failed: %s", e)
                t3 = None
        else:
            t3 = None

        if proc.generate_cloude_pottier and t3 is not None:
            try:
                cp = CloudePottierDecomposition()
                prod_cp = cp.decompose(t3)
                for attr in ["entropy", "alpha_deg", "anisotropy",
                             "lambda_1", "lambda_2", "lambda_3",
                             "alpha_1", "alpha_2", "alpha_3"]:
                    setattr(products, attr, getattr(prod_cp, attr))
                products_generated.append("cloude_pottier_H_alpha_A")
            except Exception as e:
                warnings.append(f"CloudePottierWarning: {e}")
                logger.warning("Cloude-Pottier decomposition failed: %s", e)

        if proc.generate_cpr:
            try:
                cpr_comp = CircularPolarizationRatio()
                products = cpr_comp.compute(data, products)
                products_generated.append("cpr")
            except Exception as e:
                warnings.append(f"CPRWarning: {e}")
                logger.warning("CPR computation failed: %s", e)

        if proc.generate_dop:
            try:
                dop_comp = DegreeOfPolarization()
                products = dop_comp.compute(data, products)
                products_generated.append("dop")
            except Exception as e:
                warnings.append(f"DOPWarning: {e}")
                logger.warning("DOP computation failed: %s", e)

        if proc.generate_backscatter:
            try:
                inc = data.metadata.incidence_angle_deg if data.metadata else 30.0
                bsc = BackscatterCoefficient(incidence_angle_deg=inc)
                products = bsc.compute(data, products)
                products_generated.append("backscatter_coefficients")
            except Exception as e:
                warnings.append(f"BackscatterWarning: {e}")
                logger.warning("Backscatter computation failed: %s", e)

        try:
            self._write_outputs(products, products_generated)
        except Exception as e:
            errors.append(f"OutputError: {e}")
            logger.error("Output writing failed: %s", e)

        elapsed = time.time() - t0
        logger.info("Pipeline finished in %.2f s", elapsed)
        logger.info("Products: %s", products_generated)
        return ProcessingSummary(
            input_file=self.config.input_file,
            output_dir=self.config.output_dir,
            products_generated=products_generated,
            processing_time_s=elapsed,
            config_snapshot=self._config_snapshot(),
            errors=errors,
            warnings=warnings,
        )

    def _write_outputs(self, products: DecompositionProducts, generated: list[str]) -> None:
        out_dir = Path(self.config.output_dir)
        gtiff = GeoTIFFWriter(out_dir)
        zarr_w = ZarrWriter(out_dir)
        report = ReportWriter(out_dir)

        single_band_map = {
            "entropy": products.entropy,
            "alpha_deg": products.alpha_deg,
            "anisotropy": products.anisotropy,
            "cpr": products.cpr,
            "dop": products.dop,
            "span": products.span,
            "sigma_hh": products.sigma_hh,
            "sigma_hv": products.sigma_hv,
            "sigma_vv": products.sigma_vv,
            "gamma_hh": products.gamma_hh,
            "gamma_hv": products.gamma_hv,
            "gamma_vv": products.gamma_vv,
            "odd_bounce": products.odd_bounce,
            "double_bounce": products.double_bounce,
            "volume_scattering": products.volume_scattering,
        }
        for name, arr in single_band_map.items():
            if arr is not None:
                gtiff.write_band(arr, name)

        if any(getattr(products, f) is not None for f in
               ["entropy", "alpha_deg", "anisotropy", "cpr", "dop"]):
            zarr_w.write_product_cube(products)

        report.write_statistics(products)

    def _config_snapshot(self) -> dict:
        return {
            "input_file": self.config.input_file,
            "output_dir": self.config.output_dir,
            "band": self.config.band,
            "processing": {
                "radiometric_calibration": self.config.processing.radiometric_calibration,
                "polarimetric_calibration": self.config.processing.polarimetric_calibration,
                "speckle_filter": {
                    "method": self.config.processing.speckle_filter.method,
                    "window_size": self.config.processing.speckle_filter.window_size,
                } if self.config.processing.speckle_filter else None,
                "multilook_range": self.config.processing.multilook_range,
                "multilook_azimuth": self.config.processing.multilook_azimuth,
                "generate_cloude_pottier": self.config.processing.generate_cloude_pottier,
                "generate_cpr": self.config.processing.generate_cpr,
                "generate_dop": self.config.processing.generate_dop,
            },
        }
