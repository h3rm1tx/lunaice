# LUNAICE

**Lunar Subsurface Ice Detection using Chandrayaan-2 DFSAR Polarimetric Data**

Production-grade polarimetric SAR processing pipeline for Chandrayaan-2 Dual-Frequency Synthetic Aperture Radar (DFSAR) L-band and S-band data.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Input      │────▶│  Calibration      │────▶│  Speckle Filter  │
│ (PDS4 SLC)   │     │ (Radiometric +    │     │ (Refined Lee /   │
│              │     │  Polarimetric)    │     │  Boxcar / Idani) │
└─────────────┘     └──────────────────┘     └──────────────────┘
                                                       │
                                                       ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Output      │◀────│  Decomposition    │◀────│  Coherency T3    │
│ (GeoTIFF +   │     │ (H/α/A, CPR,     │     │  Matrix Builder   │
│  Zarr + PNG) │     │  DOP, σ°, γ°)    │     │                  │
└─────────────┘     └──────────────────┘     └──────────────────┘
```

## Installation

```bash
pip install -e ".[all]"
```

## Quick Start

```bash
lunaice process -i /path/to/L1A_SLC.h5 -o output/ --band L
```

## Processing Pipeline

| Step | Component | Description |
|------|-----------|-------------|
| 1 | DFSARReader | Loads PDS4 SLC quad-pol data + metadata |
| 2 | RadiometricCalibrator | Applies calibration constant (K) per channel |
| 3 | PolarimetricCalibrator | Corrects cross-talk, channel imbalance, phase offsets |
| 4 | SpeckleFilter | Refined Lee / Boxcar / Lee-Sigma / IDAN / Bilateral |
| 5 | CoherencyMatrixBuilder | Pauli-based T3 matrix generation |
| 6 | CloudePottierDecomposition | H/α/A eigenvalue decomposition |
| 7 | CircularPolarizationRatio | SC → circular basis → CPR |
| 8 | DegreeOfPolarization | Stokes-based DOP |
| 9 | BackscatterCoefficients | σ°, γ°, span |
| 10 | GeoTIFFWriter / ZarrWriter | Output products |

## Output Products

- **GeoTIFFs**: Individual layers for H, α, A, CPR, DOP, σ°, γ°, span
- **Polarimetric Feature Cube**: Zarr-based multi-variable xarray Dataset
- **Statistics**: JSON with per-band min/max/mean/std/percentiles
- **Visualization**: PNG quick-looks (RGB composites)

## Polarimetric Products

| Product | Range | Physical Meaning |
|---------|-------|-----------------|
| Entropy (H) | [0, 1] | Scattering randomness |
| Alpha (α) | [0°, 90°] | Dominant scattering mechanism |
| Anisotropy (A) | [0, 1] | Secondary scattering relative importance |
| CPR | ≥ 0 | Circular polarization ratio (ice indicator) |
| DOP | [0, 1] | Degree of polarization |
| σ° | dB | Backscatter coefficient (HH/HV/VV) |
| γ° | dB | Incidence-angle normalized σ° |

## DFSAR Literature Basis

This implementation follows the calibration and decomposition methodology described in:

- Bhiravarasu et al. (2021) *Chandrayaan-2 DFSAR: Performance Characterization and Initial Results*, PSJ 2, 134
- Cloude & Pottier (1996) *A review of target decomposition theorems in radar polarimetry*, IEEE TGRS
- Lee & Pottier (2009) *Polarimetric Radar Imaging: From Basics to Applications*, CRC Press
- Chakraborty et al. (2024) *Subsurface ice detection in lunar PSRs using DFSAR*
- Sun et al. (2018) *Polarimetric calibration without external targets*
