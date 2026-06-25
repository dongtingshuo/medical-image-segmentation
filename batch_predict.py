import argparse
from pathlib import Path

from src.batch_prediction import batch_predict
from src.utils import load_config


def parse_extensions(value):
    return [item.strip().lower().lstrip(".") for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run segmentation inference on a directory of images.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/final_model.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--extensions", default="jpg,jpeg,png")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--model-name", default=None, help="Optional manual model override; normally leave unset.")
    args = parser.parse_args()

    config = load_config(args.config)
    threshold = args.threshold
    if threshold is None:
        threshold = float(config.get("inference", {}).get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    output_dir = args.output or Path(config.get("paths", {}).get("output_dir", "outputs")) / "batch_predictions"
    summary = batch_predict(
        input_dir=args.input_dir,
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
        threshold=threshold,
        device=args.device,
        recursive=args.recursive,
        extensions=parse_extensions(args.extensions),
        continue_on_error=not args.stop_on_error,
        model_name_override=args.model_name,
    )

    print("Batch prediction complete")
    print(f"Images: {summary['total_images']}")
    print(f"Succeeded: {summary['succeeded']}")
    print(f"Failed: {summary['failed']}")
    print(f"CSV: {summary['csv_path']}")
    print(f"Summary: {summary['summary_path']}")


if __name__ == "__main__":
    main()
