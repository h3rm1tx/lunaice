# LUNAICE API Reference

## Configuration

### `DFSARConfig`
Top-level configuration for the pipeline.

```python
from lunaice import DFSARConfig
config = DFSARConfig(
    input_file="/data/SLC.h5",
    output_dir="/output/",
    band="L",
    processing=ProcessingConfig(...),
)
```

### `ProcessingConfig`
```python
ProcessingConfig(
    radiometric_calibration=True,
    polarimetric_calibration=True,
    speckle_filter=SpeckleFilterConfig(...),
    multilook_range=1,
    multilook_azimuth=1,
    generate_coherency_matrix=True,
    generate_cloude_pottier=True,
    generate_cpr=True,
    generate_dop=True,
    generate_backscatter=True,
)
```

### `SpeckleFilterConfig`
| Field | Default | Options |
|-------|---------|---------|
| method | "refined_lee" | refined_lee, boxcar, lee_sigma, idani, bilateral |
| window_size | 7 | Odd integer >= 3 |
| damping_factor | 1.0 | float |
| n_looks | 4 | int |

## I/O

### `DFSARReader(input_path, label_path=None)`
- `read_metadata()` -> `Metadata`
- `read_calibration()` -> `CalibrationConstants`
- `read_slc()` -> `PolarimetricData`

### `GeoTIFFWriter(output_dir, crs="EPSG:4326")`
- `write_band(array, name)` -> `Path`
- `write_multiband(bands_dict, name)` -> `Path`

### `ZarrWriter(output_dir, chunks=(256,256))`
- `write_product_cube(products)` -> `Path`

### `ReportWriter(output_dir)`
- `write_summary(summary_dict)` -> `Path`
- `write_statistics(products)` -> `Path`

## Processing

### `RadiometricCalibrator(cal_constant=1.0)`
- `calibrate(data)` -> `PolarimetricData`

### `PolarimetricCalibrator(co_pol_phase=-50.0, cross_pol_phase=-5.0, ...)`
- `calibrate(data)` -> `PolarimetricData`

### `SpeckleFilter(config: SpeckleFilterConfig)`
- `apply(data)` -> `PolarimetricData`

### `CoherencyMatrixBuilder(multilook_range=1, multilook_azimuth=1)`
- `build(data)` -> `CoherencyMatrix`

### `CloudePottierDecomposition()`
- `decompose(t3)` -> `DecompositionProducts`

### `CircularPolarizationRatio()`
- `compute(data, products)` -> `DecompositionProducts`

### `DegreeOfPolarization()`
- `compute(data, products)` -> `DecompositionProducts`

### `BackscatterCoefficient(incidence_angle_deg=30.0)`
- `compute(data, products)` -> `DecompositionProducts`

## CLI

```bash
lunaice process -i <input> -o <output_dir> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--band` | L | Frequency band (L or S) |
| `-c, --config` | None | YAML config file |
| `--no-radiometric` | True | Skip radiometric calibration |
| `--no-polarimetric` | True | Skip polarimetric calibration |
| `--no-speckle` | True | Skip speckle filtering |
| `--speckle-method` | refined_lee | Speckle filter algorithm |
| `--speckle-window` | 7 | Filter kernel size |
| `--no-cloude` | True | Skip H/α/A decomposition |
| `--no-cpr` | True | Skip CPR computation |
| `--no-dop` | True | Skip DOP computation |
| `--no-backscatter` | True | Skip backscatter coefficients |
| `--multilook-range` | 1 | Range multilook factor |
| `--multilook-azimuth` | 1 | Azimuth multilook factor |
| `-v, --verbose` | False | DEBUG-level logging |
