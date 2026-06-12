from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

from lunaice.models.schemas import (
    CalibrationConstants,
    FrequencyBand,
    Metadata,
    PolarimetricData,
    PolarizationMode,
    ProcessingLevel,
)

logger = logging.getLogger(__name__)


def read_pds4_label(label_path: str | Path) -> dict:
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"PDS4 label not found: {label_path}")
    tree = ET.parse(label_path)
    root = tree.getroot()
    ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}
    info = {}
    ident = root.find(".//pds:Identification_Area", ns)
    if ident is not None:
        pid = ident.find("pds:logical_identifier", ns)
        if pid is not None:
            info["product_id"] = pid.text
    obs = root.find(".//pds:Observation_Area", ns)
    if obs is not None:
        time = obs.find(".//pds:start_date_time", ns)
        if time is not None:
            info["acquisition_time"] = time.text
    band_info = root.find(".//pds:Radar_Frequency_Band", ns)
    if band_info is not None:
        info["frequency_band"] = band_info.text
    else:
        band_attrib = root.find(".//pds:Band", ns)
        if band_attrib is not None:
            info["frequency_band"] = band_attrib.text
    params = root.findall(".//pds:Parameter", ns)
    for p in params:
        name_el = p.find("pds:name", ns)
        val_el = p.find("pds:value", ns)
        if name_el is not None and val_el is not None:
            info[name_el.text.lower().replace(" ", "_")] = val_el.text
    return info


def _load_slc_channel(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix in {".npy", ".npz"}:
        return np.load(path)
    raw = np.fromfile(path, dtype=np.complex64)
    return raw


class DFSARReader:
    def __init__(
        self,
        input_path: str | Path,
        label_path: Optional[str | Path] = None,
    ):
        self.input_path = Path(input_path)
        self.label_path = Path(label_path) if label_path else None
        self._metadata: Optional[Metadata] = None
        self._cal: Optional[CalibrationConstants] = None

    def read_metadata(self) -> Metadata:
        if self._metadata is not None:
            return self._metadata
        label_data = {}
        if self.label_path and self.label_path.exists():
            label_data = read_pds4_label(self.label_path)
        else:
            xml_candidates = list(self.input_path.parent.glob("*.xml")) + list(
                self.input_path.parent.glob("*.lbl*")
            )
            if xml_candidates:
                label_data = read_pds4_label(xml_candidates[0])
                logger.info("Found PDS4 label: %s", xml_candidates[0])
        self._metadata = Metadata.from_dict(label_data)
        return self._metadata

    def read_calibration(self) -> CalibrationConstants:
        if self._cal is not None:
            return self._cal
        met = self.read_metadata()
        raw = met.cal_params or {}
        self._cal = CalibrationConstants(
            k_hh=float(raw.get("cal_const_hh", 1.0)),
            k_hv=float(raw.get("cal_const_hv", 1.0)),
            k_vh=float(raw.get("cal_const_vh", 1.0)),
            k_vv=float(raw.get("cal_const_vv", 1.0)),
            phase_offset_hh_vv=float(raw.get("phase_hh_vv", 0.0)),
            phase_offset_hv_vh=float(raw.get("phase_hv_vh", 0.0)),
            cross_talk_hv=float(raw.get("crosstalk_hv", 0.0)),
            cross_talk_vh=float(raw.get("crosstalk_vh", 0.0)),
            channel_imbalance_amp=float(raw.get("ch_imbalance_amp", 1.0)),
            channel_imbalance_phase=float(raw.get("ch_imbalance_phase", 0.0)),
        )
        return self._cal

    def read_slc(self) -> PolarimetricData:
        met = self.read_metadata()
        cal = self.read_calibration()
        data = PolarimetricData(metadata=met, calibration=cal)
        pattern = str(self.input_path)
        if self.input_path.is_dir():
            files = {
                "hh": list(self.input_path.glob("*HH*")) + list(self.input_path.glob("*hh*")),
                "hv": list(self.input_path.glob("*HV*")) + list(self.input_path.glob("*hv*")),
                "vh": list(self.input_path.glob("*VH*")) + list(self.input_path.glob("*vh*")),
                "vv": list(self.input_path.glob("*VV*")) + list(self.input_path.glob("*vv*")),
            }
        else:
            stem = self.input_path.stem
            parent = self.input_path.parent
            files = {
                "hh": [parent / f"{stem}_HH{self.input_path.suffix}"],
                "hv": [parent / f"{stem}_HV{self.input_path.suffix}"],
                "vh": [parent / f"{stem}_VH{self.input_path.suffix}"],
                "vv": [parent / f"{stem}_VV{self.input_path.suffix}"],
            }
        for pol, candidates in files.items():
            valid = [c for c in candidates if c.exists()]
            if valid:
                ch_data = _load_slc_channel(valid[0])
                setattr(data, pol, ch_data.reshape(met.looks_azimuth, -1) if ch_data.ndim == 1 else ch_data)
                logger.info("Loaded channel %s: %s", pol.upper(), valid[0])
        data.data_id = met.product_id or data.data_id
        data.validate()
        return data
