from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lunaice.labeling.data_models import IceLabel, LABEL_NAMES


class LabelGeoTIFFWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, data: np.ndarray, name: str, dtype: str = "float32",
              profile: dict | None = None) -> Path:
        import rasterio
        from rasterio.profiles import DefaultGTiffProfile

        h, w = data.shape
        prof = DefaultGTiffProfile()
        prof.update(
            driver="GTiff", height=h, width=w, count=1,
            dtype=dtype, compress="LZW",
        )
        if profile:
            prof.update(profile)

        path = self.output_dir / f"{name}.tif"
        with rasterio.open(str(path), "w", **prof) as dst:
            if dtype == "int16" or dtype == "int32":
                dst.write(data.astype(np.dtype(dtype)), 1)
            else:
                dst.write(data.astype(np.float32), 1)
        return path

    def write_labels(self, labels: np.ndarray, name: str = "weak_labels",
                     profile: dict | None = None) -> Path:
        return self.write(labels, name, dtype="int16", profile=profile)

    def write_float(self, data: np.ndarray, name: str,
                    profile: dict | None = None) -> Path:
        return self.write(data, name, dtype="float32", profile=profile)


class LabelGeoJSONWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_ice_regions(self, labels: np.ndarray, confidence: np.ndarray,
                          lon: np.ndarray, lat: np.ndarray,
                          min_confidence: float = 0.5,
                          name: str = "ice_regions") -> Path:
        import geopandas as gpd
        from shapely.geometry import Point

        ice_fav = labels >= int(IceLabel.LIKELY)
        high_conf = confidence >= min_confidence
        mask = ice_fav & high_conf

        ys, xs = np.where(mask)
        if len(ys) == 0:
            features: list[dict] = []
        else:
            features = []
            for y, x in zip(ys, xs):
                if y < len(lat) and x < len(lon):
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [float(lon[y, x]) if lon.ndim == 2 else float(lon[x]),
                                            float(lat[y, x]) if lat.ndim == 2 else float(lat[y])],
                        },
                        "properties": {
                            "label": int(labels[y, x]),
                            "label_name": LABEL_NAMES.get(IceLabel(int(labels[y, x])), "Unknown"),
                            "confidence": float(confidence[y, x]),
                        },
                    })

        geojson = {"type": "FeatureCollection", "features": features}
        path = self.output_dir / f"{name}.geojson"
        with open(path, "w") as f:
            json.dump(geojson, f, indent=2)
        return path


class LabelStatisticsReport:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, labels: np.ndarray, confidence: np.ndarray,
                 disagreement: np.ndarray | None = None) -> Path:
        valid = ~np.isnan(labels.astype(np.float32))
        total = int(np.sum(valid))
        if total == 0:
            stats = {"error": "No valid pixels"}
        else:
            dist: dict[str, int] = {}
            for label_val in IceLabel:
                cnt = int(np.sum(labels[valid] == int(label_val)))
                dist[LABEL_NAMES[label_val]] = cnt

            ice_fav = int(np.sum(labels[valid] >= int(IceLabel.LIKELY)))
            high_conf = int(np.sum(confidence[valid] >= 0.7))

            stats = {
                "total_pixels": total,
                "label_distribution": dist,
                "label_pcts": {k: 100.0 * v / total for k, v in dist.items()},
                "mean_confidence": float(np.nanmean(confidence[valid])),
                "ice_favorable_pixels": ice_fav,
                "ice_favorable_pct": 100.0 * ice_fav / total,
                "high_confidence_pixels": high_conf,
                "high_confidence_pct": 100.0 * high_conf / total,
            }
            if disagreement is not None:
                d_valid = disagreement[valid]
                stats["mean_disagreement"] = float(np.nanmean(d_valid.astype(np.float32)))
                stats["disagreement_rate"] = float(np.sum(d_valid > 0)) / total

        html = self._build_html(stats)
        path = self.output_dir / "label_statistics.html"
        path.write_text(html)
        return path

    def _build_html(self, stats: dict) -> str:
        if "error" in stats:
            return f"<html><body><h1>Error</h1><p>{stats['error']}</p></body></html>"

        rows = ""
        for name, count in stats.get("label_distribution", {}).items():
            pct = stats.get("label_pcts", {}).get(name, 0)
            rows += f"<tr><td>{name}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>\n"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>POLARIS Label Statistics</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 2em; background: #f5f5f5; }}
.container {{ max-width: 900px; margin: auto; background: white; padding: 2em; border-radius: 8px; }}
h1 {{ color: #1a1a2e; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #1a1a2e; color: white; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1em; }}
.card {{ background: #e8f4f8; padding: 1em; border-radius: 8px; text-align: center; }}
.card h3 {{ margin: 0; font-size: 0.9em; color: #666; }}
.card p {{ font-size: 1.5em; font-weight: bold; margin: 0.5em 0; }}
</style>
</head>
<body>
<div class="container">
<h1>POLARIS Label Statistics</h1>
<div class="summary">
    <div class="card"><h3>Total Pixels</h3><p>{stats['total_pixels']:,}</p></div>
    <div class="card"><h3>Ice Fav %</h3><p>{stats['ice_favorable_pct']:.1f}%</p></div>
    <div class="card"><h3>High Conf %</h3><p>{stats['high_confidence_pct']:.1f}%</p></div>
    <div class="card"><h3>Mean Conf</h3><p>{stats['mean_confidence']:.3f}</p></div>
</div>
<h2>Label Distribution</h2>
<table><tr><th>Label</th><th>Count</th><th>%</th></tr>
{rows}</table>
</div></body></html>"""


class DisagreementMapWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, disagreement: np.ndarray, name: str = "disagreement_map",
              profile: dict | None = None) -> Path:
        return LabelGeoTIFFWriter(self.output_dir).write(
            disagreement.astype(np.float32), name, dtype="float32", profile=profile,
        )
