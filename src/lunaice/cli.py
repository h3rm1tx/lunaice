from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lunaice._version import __version__
from lunaice.config import DFSARConfig
from lunaice.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lunaice",
        description="LUNAICE: Lunar Subsurface Ice Detection using Chandrayaan-2 DFSAR Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lunaice process -i data/L1A_SLC.h5 -o output/ --band L
  lunaice process -i data/SLC/ --config configs/processing.yaml -v
  lunaice cube build -c configs/cube.yaml
  lunaice cube info -c polaris_cube.zarr
  lunaice cube query pixel --x 1000 --y -2000 -c polaris_cube.zarr
  lunaice ice prior -c configs/ice.yaml
  lunaice ice scatter -c polaris_cube.zarr
        """,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True, help="Sub-command")

    # --- process ---
    proc = sub.add_parser("process", help="Run the DFSAR polarimetric processing pipeline")
    proc.add_argument("-i", "--input", required=True, help="Input SLC file or directory")
    proc.add_argument("-o", "--output", default="output", help="Output directory")
    proc.add_argument("--band", choices=["L", "S"], default="L", help="Frequency band")
    proc.add_argument("-c", "--config", help="YAML configuration file (overrides defaults)")
    proc.add_argument("--no-radiometric", action="store_false", dest="radiometric", help="Skip radiometric calibration")
    proc.add_argument("--no-polarimetric", action="store_false", dest="polarimetric", help="Skip polarimetric calibration")
    proc.add_argument("--no-speckle", action="store_false", dest="speckle", help="Skip speckle filtering")
    proc.add_argument("--speckle-method", default="refined_lee", choices=["refined_lee", "boxcar", "lee_sigma", "idani", "bilateral"])
    proc.add_argument("--speckle-window", type=int, default=7, help="Speckle filter window size")
    proc.add_argument("--no-cloude", action="store_false", dest="cloude", help="Skip Cloude-Pottier decomposition")
    proc.add_argument("--no-cpr", action="store_false", dest="cpr", help="Skip CPR computation")
    proc.add_argument("--no-dop", action="store_false", dest="dop", help="Skip DOP computation")
    proc.add_argument("--no-backscatter", action="store_false", dest="backscatter", help="Skip backscatter coefficients")
    proc.add_argument("--multilook-range", type=int, default=1, help="Multi-looking range factor")
    proc.add_argument("--multilook-azimuth", type=int, default=1, help="Multi-looking azimuth factor")
    proc.add_argument("--output-dtype", default="float32", choices=["float32", "float64"], help="Output data type")
    proc.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    proc.add_argument("--overwrite", action="store_true", help="Overwrite existing output")

    # --- cube ---
    cube = sub.add_parser("cube", help="POLARIS Data Cube operations")
    cube_sub = cube.add_subparsers(dest="cube_command", required=True)

    cube_build = cube_sub.add_parser("build", help="Build the POLARIS data cube from sources")
    cube_build.add_argument("-c", "--config", required=True, help="Cube YAML configuration file")
    cube_build.add_argument("--overwrite", action="store_true", help="Overwrite existing cube")

    cube_info = cube_sub.add_parser("info", help="Show cube metadata and summary")
    cube_info.add_argument("-c", "--cube", required=True, help="Path to the Zarr cube")

    cube_list = cube_sub.add_parser("list", help="List variables in the cube")
    cube_list.add_argument("-c", "--cube", required=True, help="Path to the Zarr cube")

    cube_pixel = cube_sub.add_parser("pixel", help="Query a single pixel")
    cube_pixel.add_argument("--x", type=float, required=True, help="X coordinate (m)")
    cube_pixel.add_argument("--y", type=float, required=True, help="Y coordinate (m)")
    cube_pixel.add_argument("-c", "--cube", required=True, help="Path to the Zarr cube")
    cube_pixel.add_argument("--var", nargs="*", help="Variables to query (default: all)")

    cube_patch = cube_sub.add_parser("patch", help="Extract a rectangular patch")
    cube_patch.add_argument("--x", type=float, required=True, help="Center X (m)")
    cube_patch.add_argument("--y", type=float, required=True, help="Center Y (m)")
    cube_patch.add_argument("--size", type=float, default=1000.0, help="Patch size in meters")
    cube_patch.add_argument("-c", "--cube", required=True, help="Path to the Zarr cube")
    cube_patch.add_argument("--var", nargs="*", help="Variables to extract")
    cube_patch.add_argument("-o", "--output", help="Output path (save as GeoTIFF or Zarr)")

    cube_crater = cube_sub.add_parser("crater", help="Query by crater name")
    cube_crater.add_argument("--name", required=True, help="Crater name")
    cube_crater.add_argument("--buffer", type=float, default=0.0, help="Buffer around crater (m)")
    cube_crater.add_argument("-c", "--cube", required=True, help="Path to the Zarr cube")
    cube_crater.add_argument("--var", nargs="*", help="Variables to extract")
    cube_crater.add_argument("-o", "--output", help="Output path to save patch")

    cube_pyramid = cube_sub.add_parser("pyramid", help="Build multi-resolution pyramids")
    cube_pyramid.add_argument("-c", "--cube", required=True, help="Path to the Zarr cube")
    cube_pyramid.add_argument("--levels", type=int, default=5, help="Number of pyramid levels")

    cube_craters_list = cube_sub.add_parser("craters", help="List known craters in catalog")

    # --- ice ---
    ice = sub.add_parser("ice", help="POLARIS Ice Prior Engine operations")
    ice_sub = ice.add_subparsers(dest="ice_command", required=True)

    ice_prior = ice_sub.add_parser("prior", help="Run the ice prior engine")
    ice_prior.add_argument("-c", "--config", required=True, help="Ice engine YAML configuration")
    ice_prior.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    ice_scatter = ice_sub.add_parser("scatter", help="Classify scattering mechanisms from cube")
    ice_scatter.add_argument("-c", "--cube", required=True, help="Path to the Zarr cube")
    ice_scatter.add_argument("-o", "--output", default="scattering_class.tif", help="Output GeoTIFF")

    return parser


def cmd_process(args: argparse.Namespace) -> int:
    if args.config:
        config = DFSARConfig.from_yaml(args.config)
        config.input_file = args.input
        config.output_dir = args.output
        if args.verbose:
            config.logging_level = "DEBUG"
    else:
        from lunaice.config import ProcessingConfig, SpeckleFilterConfig
        speckle = None
        if args.speckle:
            speckle = SpeckleFilterConfig(method=args.speckle_method, window_size=args.speckle_window)
        proc = ProcessingConfig(
            radiometric_calibration=args.radiometric,
            polarimetric_calibration=args.polarimetric,
            speckle_filter=speckle,
            multilook_range=args.multilook_range,
            multilook_azimuth=args.multilook_azimuth,
            generate_cloude_pottier=args.cloude,
            generate_cpr=args.cpr,
            generate_dop=args.dop,
            generate_backscatter=args.backscatter,
            output_dtype=args.output_dtype,
        )
        config = DFSARConfig(
            input_file=args.input,
            output_dir=args.output,
            band=args.band,
            processing=proc,
            verbose=args.verbose,
            overwrite=args.overwrite,
        )
        if args.verbose:
            config.logging_level = "DEBUG"

    pipeline = Pipeline(config)
    summary = pipeline.run()
    print(json.dumps({
        "status": "success" if not summary.errors else "partial_failure",
        "products": summary.products_generated,
        "time_s": round(summary.processing_time_s, 2),
        "errors": summary.errors,
        "warnings": summary.warnings,
    }, indent=2))
    return 1 if summary.errors else 0


def cmd_cube(args: argparse.Namespace) -> int:
    if args.cube_command == "build":
        from lunaice.cube import CubeConfig, CubeBuilder
        config = CubeConfig.from_yaml(args.config)
        if args.overwrite:
            config.overwrite = True
        builder = CubeBuilder(config)
        path = builder.build()
        print(json.dumps({"status": "built", "path": path}, indent=2))
        return 0

    elif args.cube_command == "info":
        from lunaice.cube import CubeConfig, PolarisDataCube
        cfg = CubeConfig(cube_path=args.cube)
        cube = PolarisDataCube(cfg)
        print(cube.summary())
        cube.close()
        return 0

    elif args.cube_command == "list":
        from lunaice.cube import CubeConfig, PolarisDataCube
        cfg = CubeConfig(cube_path=args.cube)
        cube = PolarisDataCube(cfg)
        info = cube.info()
        print("Variables:")
        for v in info["variables"]:
            print(f"  {v}")
        print(f"\nCoords: {info['coords']}")
        print(f"Shape: {info['shape']}")
        cube.close()
        return 0

    elif args.cube_command == "pixel":
        from lunaice.cube import CubeConfig, PolarisDataCube
        from lunaice.cube.queries import PixelQuery, QueryAPI
        cfg = CubeConfig(cube_path=args.cube)
        cube = PolarisDataCube(cfg)
        api = QueryAPI(cube)
        result = api.pixel(PixelQuery(x=args.x, y=args.y, variables=args.var))
        print(json.dumps(result, indent=2, default=str))
        cube.close()
        return 0

    elif args.cube_command == "patch":
        from lunaice.cube import CubeConfig, PolarisDataCube
        from lunaice.cube.queries import PatchQuery, QueryAPI
        cfg = CubeConfig(cube_path=args.cube)
        cube = PolarisDataCube(cfg)
        api = QueryAPI(cube)
        patch = api.patch(PatchQuery(x_center=args.x, y_center=args.y, size_m=args.size, variables=args.var))
        if args.output:
            out = Path(args.output)
            if out.suffix == ".zarr":
                patch.to_zarr(str(out), mode="w")
            else:
                patch.to_netcdf(str(out))
            print(f"Saved to {out}")
        else:
            print(patch)
        cube.close()
        return 0

    elif args.cube_command == "crater":
        from lunaice.cube import CubeConfig, PolarisDataCube
        from lunaice.cube.queries import CraterQuery, QueryAPI
        cfg = CubeConfig(cube_path=args.cube)
        cube = PolarisDataCube(cfg)
        api = QueryAPI(cube)
        patch = api.crater(CraterQuery(crater_name=args.name, buffer_m=args.buffer, variables=args.var))
        if args.output:
            out = Path(args.output)
            if out.suffix == ".zarr":
                patch.to_zarr(str(out), mode="w")
            else:
                patch.to_netcdf(str(out))
            print(f"Saved to {out}")
        else:
            print(patch)
        cube.close()
        return 0

    elif args.cube_command == "pyramid":
        from lunaice.cube import CubeConfig, PolarisDataCube
        from lunaice.cube.pyramid import PyramidBuilder
        cfg = CubeConfig(cube_path=args.cube)
        cube = PolarisDataCube(cfg)
        builder = PyramidBuilder(cube)
        paths = builder.build(levels=args.levels)
        print(json.dumps({"pyramids": paths}, indent=2))
        cube.close()
        return 0

    elif args.cube_command == "craters":
        from lunaice.cube.craters import CraterCatalog
        cat = CraterCatalog()
        print(f"{'Name':<20} {'Lon':>8} {'Lat':>8} {'Diam(km)':>10} {'X(m)':>12} {'Y(m)':>12}")
        print("-" * 70)
        for r in sorted(cat.list_craters(), key=lambda x: x.name):
            print(f"{r.name:<20} {r.lon_deg:>8.1f} {r.lat_deg:>8.1f} {r.diameter_km:>10.1f} {r.x_m:>12.0f} {r.y_m:>12.0f}")
        return 0

    return 1


def cmd_ice(args: argparse.Namespace) -> int:
    if args.ice_command == "prior":
        from lunaice.ice import IcePriorConfig, IcePriorEngine
        config = IcePriorConfig.from_yaml(args.config)
        if args.overwrite:
            config.overwrite = True
        engine = IcePriorEngine(config)
        result = engine.run()
        print(json.dumps({
            "status": "success",
            "output_dir": config.output_dir,
            "elapsed_s": round(result["elapsed_s"], 2),
            "mean_ice_prior": round(float(np.nanmean(result["ice_prior"])), 4),
            "max_ice_prior": round(float(np.nanmax(result["ice_prior"])), 4),
            "candidate_regions": len(result["candidate_gdf"]) if not result["candidate_gdf"].empty else 0,
        }, indent=2))
        return 0

    elif args.ice_command == "scatter":
        from lunaice.cube import CubeConfig, PolarisDataCube
        from lunaice.ice.classification.cloude_pottier import CloudePottierClassifier
        cfg = CubeConfig(cube_path=args.cube)
        cube = PolarisDataCube(cfg)
        ds = cube.ds
        entropy = ds["entropy"].compute().values if "entropy" in ds else None
        alpha = ds["alpha_deg"].compute().values if "alpha_deg" in ds else None
        if entropy is None or alpha is None:
            print("Error: cube must contain 'entropy' and 'alpha_deg'")
            return 1
        classifier = CloudePottierClassifier()
        sc = classifier.classify(entropy, alpha)
        import rasterio
        profile = {"driver": "GTiff", "height": sc.shape[0], "width": sc.shape[1],
                   "count": 1, "dtype": "int16", "compress": "LZW"}
        with rasterio.open(args.output, "w", **profile) as dst:
            dst.write(sc.astype(np.int16), 1)
        print(f"Saved scattering class map to {args.output}")
        cube.close()
        return 0

    return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "process":
        return cmd_process(args)
    elif args.command == "cube":
        return cmd_cube(args)
    elif args.command == "ice":
        return cmd_ice(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
