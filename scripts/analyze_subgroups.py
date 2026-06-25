import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_failure_cases import build_loader  # noqa: E402
from scripts.search_threshold import load_model  # noqa: E402
from src.analysis.failure_cases import collect_failure_case_records, flatten_failure_record  # noqa: E402
from src.subgroup_analysis import (  # noqa: E402
    attach_subgroups,
    image_contrast,
    summarize_subgroups,
    write_subgroup_outputs,
)
from src.utils import get_device, load_config  # noqa: E402


def read_image_contrast(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image for subgroup analysis: {path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_contrast(image_rgb)


def main():
    parser = argparse.ArgumentParser(description="Run lesion-size and image-contrast subgroup analysis.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test", "external"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs/subgroup_analysis")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {args.threshold}")
    config = load_config(args.config)
    device = get_device(args.device or config.get("device", "auto"))
    dataset, loader = build_loader(config, args.split)
    model = load_model(config, args.checkpoint, device)
    records = collect_failure_case_records(model, loader, dataset, device, threshold=args.threshold)
    flat_records = []
    for record in records:
        image_path, _ = dataset.pairs[int(record["sample_index"])]
        flat = flatten_failure_record(record)
        flat["image_contrast"] = read_image_contrast(image_path)
        flat["split"] = args.split
        flat_records.append(flat)

    enriched = attach_subgroups(flat_records)
    summary = summarize_subgroups(enriched)
    outputs = write_subgroup_outputs(enriched, summary, args.output_dir)
    print(f"Analyzed {len(enriched)} samples from split `{args.split}`.")
    print(f"Saved subgroup report to {outputs['markdown']}")


if __name__ == "__main__":
    main()
