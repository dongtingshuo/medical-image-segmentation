import argparse
from pathlib import Path

from src.inference import predict_file
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/unet.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    config = load_config(args.config)
    threshold = args.threshold
    if threshold is None:
        threshold = float(config.get("inference", {}).get("threshold", 0.5))
    output_dir = args.output or Path(config.get("paths", {}).get("output_dir", "outputs")) / "samples"
    result = predict_file(
        image_path=args.image,
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
        threshold=threshold,
        device=args.device,
    )
    print(f"Device: {result['device']}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Inference time: {result['inference_time']:.4f}s")
    print(f"Lesion area ratio: {result['lesion_ratio']:.6f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
