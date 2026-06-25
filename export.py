import argparse

from src.export import export_model
from src.utils import load_config


def parse_formats(value):
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Export a trained segmentation model to ONNX and/or TorchScript.")
    parser.add_argument("--config", default="configs/final_model.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="exports")
    parser.add_argument("--formats", default="torchscript,onnx", help="Comma-separated formats: torchscript,onnx")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fixed-batch", action="store_true", help="Disable dynamic batch axis for ONNX export.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {args.batch_size}")

    config = load_config(args.config)
    manifest = export_model(
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        formats=parse_formats(args.formats),
        device=args.device,
        opset=args.opset,
        batch_size=args.batch_size,
        dynamic_batch=not args.fixed_batch,
    )

    print("Export complete")
    print(f"Manifest: {manifest['manifest_path']}")
    for name, item in manifest["formats"].items():
        print(f"{name}: {item['path']} ({item['bytes']} bytes, sha256={item['sha256']})")


if __name__ == "__main__":
    main()
