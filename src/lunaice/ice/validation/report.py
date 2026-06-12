from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from lunaice.ice.validation.validator import RegionMetrics

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #0a0e17; color: #e0e6ed; }}
  h1 {{ color: #64b5f6; border-bottom: 2px solid #1a2744; padding-bottom: 0.5rem; }}
  h2 {{ color: #81c784; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #1a2744; padding: 0.5rem; text-align: left; }}
  th {{ background: #1a2744; color: #64b5f6; }}
  tr:nth-child(even) {{ background: #111827; }}
  .summary {{ background: #1a2744; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
  .metric {{ display: inline-block; margin: 0.5rem 2rem 0.5rem 0; }}
  .metric-value {{ font-size: 1.5rem; font-weight: bold; color: #81c784; }}
  .metric-label {{ font-size: 0.8rem; color: #94a3b8; }}
  .footer {{ margin-top: 3rem; color: #64748b; font-size: 0.85rem; border-top: 1px solid #1a2744; padding-top: 1rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>Generated: {timestamp}</p>
{content}
<div class="footer">LUNAICE / POLARIS &mdash; Physics-Based Ice Prior Engine</div>
</body>
</html>"""


class ValidationReport:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        metrics: list[RegionMetrics],
        global_stats: Optional[dict] = None,
        filename: str = "validation_report.html",
    ) -> Path:
        content = []
        if global_stats:
            content.append('<div class="summary">')
            for key, val in global_stats.items():
                label = key.replace("_", " ").title()
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                content.append(
                    f'<div class="metric"><div class="metric-value">{val_str}</div>'
                    f'<div class="metric-label">{label}</div></div>'
                )
            content.append("</div>")
        content.append("<h2>Region Validation Results</h2>")
        content.append("<table><tr>"
            "<th>Region</th><th>Pixels</th><th>Mean Ice Prior</th><th>Std Ice Prior</th>"
            "<th>Mean Confidence</th><th>Mean CPR</th><th>Mean DOP</th>"
            "<th>PSR Fraction</th><th>Dominant Scattering</th><th>High Conf. Fraction</th>"
            "</tr>")
        for m in sorted(metrics, key=lambda x: x.mean_ice_prior, reverse=True):
            content.append(f"<tr><td>{m.name}</td><td>{m.n_pixels}</td>"
                f"<td>{m.mean_ice_prior:.4f}</td><td>{m.std_ice_prior:.4f}</td>"
                f"<td>{m.mean_confidence:.4f}</td><td>{m.mean_cpr:.4f}</td>"
                f"<td>{m.mean_dop:.4f}</td><td>{m.psr_fraction:.3f}</td>"
                f"<td>{m.dominant_scattering}</td><td>{m.high_confidence_fraction:.3f}</td></tr>")
        content.append("</table>")
        html = HTML_TEMPLATE.format(
            title="POLARIS Ice Prior Validation Report",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content="\n".join(content),
        )
        path = self.output_dir / filename
        path.write_text(html)
        logger.info("Validation report written to %s", path)
        return path
