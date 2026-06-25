import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.search_threshold import load_model  # noqa: E402
from src.analysis.failure_cases import (  # noqa: E402
    collect_failure_case_records,
    save_failure_case_visuals,
    select_failure_cases,
    write_failure_case_outputs,
)
from src.dataset import SkinLesionDataset, get_val_transforms  # noqa: E402
from src.utils import data_path, get_device, load_config  # noqa: E402


def build_loader(config, split):
    images_value = data_path(config, f"{split}_images_dir")
    masks_value = data_path(config, f"{split}_masks_dir")
    if not images_value or not masks_value:
        raise ValueError(f"Config does not define {split}_images_dir and {split}_masks_dir.")
    dataset = SkinLesionDataset(images_value, masks_value, transform=get_val_transforms(config))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def main():
    parser = argparse.ArgumentParser(description="Rank and export worst segmentation failure cases.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test", "external"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--sort-by", default="dice", choices=["dice", "iou", "precision", "recall", "false_positive", "false_negative"])
    parser.add_argument("--output-dir", default="outputs/failure_cases")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {args.threshold}")

    config = load_config(args.config)
    device = get_device(args.device or config.get("device", "auto"))
    dataset, loader = build_loader(config, args.split)
    model = load_model(config, args.checkpoint, device)
    records = collect_failure_case_records(model, loader, dataset, device, threshold=args.threshold)
    selected = select_failure_cases(records, top_k=args.top_k, sort_by=args.sort_by)
    output_dir = Path(args.output_dir)
    save_failure_case_visuals(selected, dataset, output_dir / "visuals")
    outputs = write_failure_case_outputs(records, selected, output_dir, split=args.split, threshold=args.threshold)
    print(f"Analyzed {len(records)} samples from split `{args.split}` at threshold {args.threshold:.3f}.")
    print(f"Saved failure analysis report to {outputs['markdown']}")


if __name__ == "__main__":
    main()
