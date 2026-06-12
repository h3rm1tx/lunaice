# LUNAICE Architecture

## Overview

LUNAICE is a modular, configurable polarimetric SAR processing pipeline designed specifically for Chandrayaan-2 DFSAR data. The architecture follows a functional pipeline pattern where each processing step is an independent component that transforms the `PolarimetricData` object through the processing chain.

## Class Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     DFSARConfig                              │
│  - input_file: str                                           │
│  - output_dir: str                                           │
│  - band: str (L/S)                                           │
│  - processing: ProcessingConfig                              │
│  + from_yaml() -> DFSARConfig                                │
│  + resolve() -> None                                         │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Pipeline                                │
│  - config: DFSARConfig                                       │
│  + run() -> ProcessingSummary                                │
└─────────────────────────────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   DFSARReader     │  │ RadiometricCal   │  │ PolarimetricCal  │
│ + read_slc()      │  │  ibrator         │  │  ibrator         │
│ + read_metadata() │  │ + calibrate()    │  │ + calibrate()    │
└──────────────────┘  └──────────────────┘  └──────────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │  SpeckleFilter    │
                                          │  (5 algorithms)   │
                                          │  + apply()        │
                                          └──────────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │ CoherencyMatrix   │
                                          │  Builder          │
                                          │ + build() -> T3   │
                                          └──────────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │CloudePottierDecomp│
                                          │ + decompose()     │
                                          │  -> H/α/A/λ       │
                                          └──────────────────┘
                                                    │
          ┌────────────────────┬────────────────────┼────────────────────┐
          ▼                    ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ CircularPol.     │  │ DegreeOfPol.     │  │ BackscatterCoeff │  │   Output Writers  │
│  Ratio (CPR)     │  │  (DOP)           │  │  (σ°, γ°)        │  │  GeoTIFF / Zarr   │
└──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

## Data Flow

1. **Input Stage**: `DFSARReader` loads SLC complex data + PDS4 metadata
2. **Calibration Stage**: Radiometric (DN → σ°) + Polarimetric (phase/cross-talk correction)
3. **Filtering Stage**: Speckle reduction (5 algorithms available)
4. **Decomposition Stage**:
   - Pauli vector → T3 coherency matrix
   - Eigenvalue decomposition → H, α, A, λ₁, λ₂, λ₃
   - Circular basis transform → CPR
   - Stokes parameters → DOP
5. **Output Stage**: GeoTIFF (per-band), Zarr (multi-variable cube), JSON statistics

## Key Design Decisions

### Config-Driven
All processing parameters are externalized in YAML. No hardcoded values.

### Streaming-Aware
Zarr chunked storage enables out-of-core processing of large scenes.

### Type-Safe
Full Python 3.12 type hints via `dataclass` schemas with validation.

### Graceful Degradation
Pipeline continues with warnings if individual processing steps fail; partial results are still written.

## Data Models

### PolarimetricData
The core data container holding complex SLC arrays (HH, HV, VH, VV) with attached Metadata and CalibrationConstants.

### CoherencyMatrix
9-element T3 matrix stored as individual `np.ndarray` components (t11..t33).

### DecompositionProducts
Container for all derived products (H, α, A, CPR, DOP, σ°, γ°, span, λ₁, λ₂, λ₃, α₁, α₂, α₃).

## Processing Modes

| Mode | Bands | Polarization | Products |
|------|-------|-------------|----------|
| Quad-pol L | L | HH, HV, VH, VV | Full (all products) |
| Quad-pol S | S | HH, HV, VH, VV | Full (all products) |
| Dual-pol | L/S | HH, HV or VV, VH | Subset (CPR, DOP, σ°) |
| Hybrid | L/S | Compact | Subset (DOP, Stokes) |
