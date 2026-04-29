from __future__ import annotations

import argparse
from pathlib import Path

from gcp_pipeline import DEFAULT_CONFIDENCE_THRESHOLD, PipelineConfig, run_pipeline


DEFAULT_WEIGHTS_PATH = r"D:\gcp\yolov8 obb\runs\train_8cls_obb_s_local_kmpfix\weights\best.pt"
DEFAULT_INPUT_PATH = r"D:\gcp\slice\raw_val"
DEFAULT_OUTPUT_DIR = r"D:\gcp\gcp_pipeline_outputs"
DEFAULT_PYTHON_EXE = r"D:\Anaconda_envs\envs\gcp\python.exe"
DEFAULT_DETECT_SCRIPT = r"D:\gcp\detect_gcp_hourglass.py"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GCP large-image to global-coordinate pipeline.")
    parser.add_argument("--input", dest="input_path", default=DEFAULT_INPUT_PATH, help="Input image path or directory")
    parser.add_argument("--weights", dest="weights_path", default=DEFAULT_WEIGHTS_PATH, help="Path to best.pt")
    parser.add_argument("--output-dir", dest="output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--python-exe", dest="python_exe", default=DEFAULT_PYTHON_EXE, help="Python executable for the black-box fine locator")
    parser.add_argument("--detect-script", dest="detect_script_path", default=DEFAULT_DETECT_SCRIPT, help="Path to detect_gcp_hourglass.py")
    parser.add_argument("--device", dest="device", default="0", help="Inference device, for example 0 or cpu")
    parser.add_argument("--slice-size", dest="slice_size", type=int, default=1024, help="Slice size in pixels")
    parser.add_argument("--overlap", dest="overlap", type=float, default=0.2, help="Slice overlap ratio")
    parser.add_argument(
        "--conf",
        dest="conf",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold. Values below 0.85 are clamped to 0.85 by the pipeline.",
    )
    parser.add_argument("--roi-pad", dest="roi_pad", type=int, default=8, help="Extra ROI padding in pixels")
    parser.add_argument("--max-images", dest="max_images", type=int, default=None, help="Optional limit on image count")
    parser.add_argument(
        "--prefer-manual-slicing",
        dest="prefer_sahi",
        action="store_false",
        help="Disable SAHI and use manual Ultralytics slice inference only",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        input_path=args.input_path,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        python_exe=args.python_exe,
        detect_script_path=args.detect_script_path,
        device=args.device,
        slice_size=args.slice_size,
        overlap=args.overlap,
        conf=args.conf,
        roi_pad=args.roi_pad,
        max_images=args.max_images,
        prefer_sahi=args.prefer_sahi,
    )

    results = run_pipeline(config)
    print("========== GCP Pipeline Complete ==========")
    print(f"images={len(results)}")
    if results:
        print(f"output_root={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
