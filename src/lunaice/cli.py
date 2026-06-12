from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lunaice import __version__
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
  lunaice process -i data/L1A_SLC.h5 --no-speckle --multilook 2 2
        """,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True, help="Sub-command")

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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "process":
        return cmd_process(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
