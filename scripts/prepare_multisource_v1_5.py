import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multisource_data import prepare_multisource_dataset  # noqa: E402


def parse_triplet(value):
    parts = str(value).split(":", maxsplit=2)
    if len(parts) != 3:
        raise ValueError("Dataset inputs must use NAME:IMAGES_DIR:MASKS_DIR format.")
    return parts[0], Path(parts[1]), Path(parts[2])


def main():
    parser = argparse.ArgumentParser(description="Prepare deduplicated multi-source v1.5 segmentation data.")
    parser.add_argument("--source", action="append", required=True, help="NAME:IMAGES_DIR:MASKS_DIR; first is primary.")
    parser.add_argument("--benchmark", action="append", required=True, help="NAME:IMAGES_DIR:MASKS_DIR")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--ssim-threshold", type=float, default=0.95)
    args = parser.parse_args()

    result = prepare_multisource_dataset(
        [parse_triplet(value) for value in args.source],
        [parse_triplet(value) for value in args.benchmark],
        args.output_root,
        phash_distance=args.phash_distance,
        ssim_threshold=args.ssim_threshold,
    )
    summary = {"accepted": result["accepted"], "manifest": str(result["manifest"])}
    (Path(args.output_root) / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
