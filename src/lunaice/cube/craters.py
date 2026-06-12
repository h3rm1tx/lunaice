from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from lunaice.cube.spatial import lonlat_to_xy

logger = logging.getLogger(__name__)

_CRATER_CATALOG_BUILTIN = Path(__file__).parent / "data" / "craters.csv"


@dataclass
class CraterRecord:
    name: str
    lon_deg: float
    lat_deg: float
    diameter_km: float
    x_m: float = 0.0
    y_m: float = 0.0

    def __post_init__(self):
        if self.x_m == 0.0 and self.y_m == 0.0:
            x, y = lonlat_to_xy(np.array([self.lon_deg]), np.array([self.lat_deg]))
            self.x_m = float(x[0])
            self.y_m = float(y[0])


class CraterCatalog:
    def __init__(self, catalog_path: Optional[str | Path] = None):
        self._records: dict[str, CraterRecord] = {}
        path = Path(catalog_path) if catalog_path else _CRATER_CATALOG_BUILTIN
        if path.exists():
            self._load(path)
        else:
            self._load_builtin()

    def _load_builtin(self) -> None:
        builtin = [
            CraterRecord(name="Faustini", lon_deg=77.0, lat_deg=-87.2, diameter_km=42.0),
            CraterRecord(name="Shoemaker", lon_deg=45.0, lat_deg=-88.1, diameter_km=51.0),
            CraterRecord(name="Haworth", lon_deg=-5.0, lat_deg=-87.5, diameter_km=51.0),
            CraterRecord(name="Cabeus", lon_deg=-35.5, lat_deg=-84.9, diameter_km=98.0),
            CraterRecord(name="Shackleton", lon_deg=0.0, lat_deg=-89.9, diameter_km=21.0),
            CraterRecord(name="Amundsen", lon_deg=83.0, lat_deg=-84.4, diameter_km=103.0),
            CraterRecord(name="Scott", lon_deg=45.0, lat_deg=-82.1, diameter_km=108.0),
            CraterRecord(name="Hermite-A", lon_deg=-85.0, lat_deg=86.0, diameter_km=20.0),
            CraterRecord(name="Byrgius C", lon_deg=65.0, lat_deg=-24.0, diameter_km=13.0),
        ]
        for r in builtin:
            self._records[r.name.lower()] = r
        logger.info("Loaded %d built-in crater records", len(self._records))

    def _load(self, path: Path) -> None:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = CraterRecord(
                    name=row["name"],
                    lon_deg=float(row["lon_deg"]),
                    lat_deg=float(row["lat_deg"]),
                    diameter_km=float(row["diameter_km"]),
                )
                self._records[r.name.lower()] = r
        logger.info("Loaded %d crater records from %s", len(self._records), path)

    def lookup(self, name: str) -> Optional[CraterRecord]:
        return self._records.get(name.lower())

    def list_craters(self) -> list[CraterRecord]:
        return list(self._records.values())

    def search_near(
        self, x_m: float, y_m: float, radius_m: float
    ) -> list[CraterRecord]:
        results = []
        for r in self._records.values():
            dist = np.sqrt((r.x_m - x_m) ** 2 + (r.y_m - y_m) ** 2)
            if dist <= radius_m:
                results.append(r)
        return results
