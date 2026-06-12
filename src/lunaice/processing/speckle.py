from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import generic_filter, uniform_filter

from lunaice.models.schemas import PolarimetricData
from lunaice.config import SpeckleFilterConfig

logger = logging.getLogger(__name__)


class SpeckleFilter:
    def __init__(self, config: SpeckleFilterConfig):
        self.config = config

    def _refined_lee(self, intensity: np.ndarray) -> np.ndarray:
        w = self.config.window_size
        n_looks = self.config.n_looks
        half = w // 2
        padded = np.pad(intensity, half, mode="reflect")
        mu = uniform_filter(padded, size=w)[half:-half, half:-half]
        sq = uniform_filter(padded ** 2, size=w)[half:-half, half:-half]
        var = np.maximum(sq - mu ** 2, 0)
        sigma_v = mu * np.sqrt(1.0 / n_looks) if n_looks > 0 else np.zeros_like(mu)
        var_x = np.maximum(var - sigma_v ** 2, 0)
        weights = var_x / (var_x + sigma_v ** 2 + 1e-10)
        return mu + weights * (intensity - mu)

    def _boxcar(self, intensity: np.ndarray) -> np.ndarray:
        w = self.config.window_size
        return uniform_filter(intensity, size=w)

    def _lee_sigma(self, intensity: np.ndarray) -> np.ndarray:
        w = self.config.window_size
        n_sigma = self.config.damping_factor
        half = w // 2
        padded = np.pad(intensity, half, mode="reflect")
        h, w_im = intensity.shape
        result = np.empty_like(intensity)
        for i in range(h):
            for j in range(w_im):
                patch = padded[i : i + w, j : j + w]
                mu_p = patch.mean()
                sigma_p = patch.std()
                lower = mu_p - n_sigma * sigma_p
                upper = mu_p + n_sigma * sigma_p
                mask = (patch >= lower) & (patch <= upper)
                selected = patch[mask]
                result[i, j] = selected.mean() if selected.size > 0 else mu_p
        return result

    def _idani(self, intensity: np.ndarray) -> np.ndarray:
        w = self.config.window_size
        half = w // 2
        padded = np.pad(intensity, half, mode="reflect")
        h, w_im = intensity.shape
        result = np.empty_like(intensity)
        n_looks = max(self.config.n_looks, 1)
        enl = n_looks
        for i in range(h):
            for j in range(w_im):
                patch = padded[i : i + w, j : j + w]
                mu_p = patch.mean()
                var_p = patch.var()
                cu = np.sqrt(var_p) / (mu_p + 1e-10)
                ci = 1.0 / np.sqrt(enl)
                cmax = np.sqrt(1 + 2.0 / enl)
                if cu <= ci:
                    result[i, j] = mu_p
                elif cu >= cmax:
                    result[i, j] = intensity[i, j]
                else:
                    damp = (cu - ci) / (cmax - ci)
                    result[i, j] = mu_p * (1 - damp) + intensity[i, j] * damp
        return result

    def _bilateral(self, intensity: np.ndarray) -> np.ndarray:
        w = self.config.window_size
        sigma_s = w / 3.0
        sigma_r = intensity.std() * self.config.damping_factor
        half = w // 2
        padded = np.pad(intensity, half, mode="reflect")
        h, w_im = intensity.shape
        result = np.empty_like(intensity)
        yi, xi = np.ogrid[-half : half + 1, -half : half + 1]
        spatial_kernel = np.exp(-(yi ** 2 + xi ** 2) / (2 * sigma_s ** 2))
        for i in range(h):
            for j in range(w_im):
                patch = padded[i : i + w, j : j + w]
                range_kernel = np.exp(-((patch - intensity[i, j]) ** 2) / (2 * sigma_r ** 2 + 1e-10))
                kernel = spatial_kernel * range_kernel
                result[i, j] = np.sum(patch * kernel) / (np.sum(kernel) + 1e-10)
        return result

    def _filter_channel(self, channel: np.ndarray) -> np.ndarray:
        intensity = np.abs(channel)
        intensity_smooth = getattr(self, f"_{self.config.method}")(intensity)
        phase = np.angle(channel)
        return intensity_smooth * np.exp(1j * phase)

    def apply(self, data: PolarimetricData) -> PolarimetricData:
        logger.info("Applying %s speckle filter (window=%d)", self.config.method, self.config.window_size)
        for pol in ["hh", "hv", "vh", "vv"]:
            arr = getattr(data, pol, None)
            if arr is not None:
                setattr(data, pol, self._filter_channel(arr))
        logger.info("Speckle filtering complete")
        return data
