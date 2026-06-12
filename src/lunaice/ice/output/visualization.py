from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

FIGSIZE = (10, 6)


class IceVisualizationEngine:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def histogram(
        self,
        data: np.ndarray,
        title: str,
        filename: str,
        bins: int = 100,
        xlabel: str = "Value",
        color: str = "#64b5f6",
    ) -> Path:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        valid = data[~np.isnan(data)]
        ax.hist(valid, bins=bins, color=color, alpha=0.8, edgecolor="none")
        ax.set_title(title, color="#e0e6ed")
        ax.set_xlabel(xlabel, color="#94a3b8")
        ax.set_ylabel("Frequency", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#0a0e17")
        ax.spines["bottom"].set_color("#1a2744")
        ax.spines["top"].set_color("#1a2744")
        ax.spines["left"].set_color("#1a2744")
        ax.spines["right"].set_color("#1a2744")
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved histogram: %s", path)
        return path

    def heatmap(
        self,
        data: np.ndarray,
        title: str,
        filename: str,
        cmap: str = "inferno",
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
        cbar.ax.tick_params(colors="#94a3b8")
        cbar.outline.set_edgecolor("#1a2744")
        ax.set_title(title, color="#e0e6ed", fontsize=14)
        ax.tick_params(colors="#94a3b8")
        fig.patch.set_facecolor("#0a0e17")
        ax.set_facecolor("#111827")
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved heatmap: %s", path)
        return path

    def feature_distribution(
        self,
        indicators: dict[str, np.ndarray],
        filename: str = "feature_distribution.png",
    ) -> Path:
        n = len(indicators)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * min(n, 3), 8))
        axes = axes.flatten() if n > 1 else [axes]
        colors = ["#64b5f6", "#81c784", "#ffb74d", "#f06292", "#ba68c8", "#4dd0e1"]
        for ax, (name, arr), color in zip(axes, indicators.items(), colors):
            valid = arr[~np.isnan(arr)]
            ax.hist(valid, bins=80, color=color, alpha=0.8, edgecolor="none")
            ax.set_title(name.replace("_", " ").title(), color="#e0e6ed")
            ax.tick_params(colors="#94a3b8")
            ax.set_facecolor("#111827")
        for ax in axes[len(indicators):]:
            ax.set_visible(False)
        fig.patch.set_facecolor("#0a0e17")
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved feature distribution: %s", path)
        return path
