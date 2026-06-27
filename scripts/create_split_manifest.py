import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multisource_data import _attach_strata, discover_pairs, sample_features  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Create a contrast/lesion-ratio subgroup manifest for one split.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="benchmark")
    args = parser.parse_args()

    rows = []
    for stem, image_path, mask_path in discover_pairs(args.images_dir, args.masks_dir):
        features = sample_features(image_path, mask_path)
        rows.append(
            {
                "source": args.source,
                "stem": stem,
                "status": "accepted",
                "contrast": features["contrast"],
                "lesion_ratio": features["lesion_ratio"],
            }
        )
    _attach_strata(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} subgroup records to {output}")


if __name__ == "__main__":
    main()
