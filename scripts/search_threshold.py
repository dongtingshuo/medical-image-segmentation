import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SkinLesionDataset, get_val_transforms  # noqa: E402
from src.inference import build_model_from_config  # noqa: E402
from src.threshold_search import (  # noqa: E402
    best_threshold,
    collect_probability_batches,
    parse_thresholds,
    summarize_threshold_predictions,
    write_threshold_search_outputs,
)
from src.utils import (  # noqa: E402
    checkpoint_model_config,
    data_path,
    get_device,
    load_checkpoint,
    load_checkpoint_payload,
    load_config,
)


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


def load_model(config, checkpoint_path, device):
    checkpoint = load_checkpoint_payload(checkpoint_path, device=device)
    model = build_model_from_config(config, checkpoint=checkpoint).to(device)
    expected_model_config = checkpoint_model_config(checkpoint) or config.get("model", {})
    load_checkpoint(
        checkpoint_path,
        model,
        device,
        expected_model_config=expected_model_config,
        checkpoint=checkpoint,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Search the best segmentation threshold on a configured split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test", "external"])
    parser.add_argument("--thresholds", help="Comma-separated thresholds, for example: 0.3,0.35,0.4")
    parser.add_argument("--start", type=float, default=0.3)
    parser.add_argument("--stop", type=float, default=0.7)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--best-metric", default="dice", choices=["dice", "iou", "precision", "recall"])
    parser.add_argument("--output-dir", default="outputs/threshold_search")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(args.device or config.get("device", "auto"))
    thresholds = parse_thresholds(args.thresholds, start=args.start, stop=args.stop, step=args.step)
    _, loader = build_loader(config, args.split)
    model = load_model(config, args.checkpoint, device)
    probability_batches, mask_batches = collect_probability_batches(model, loader, device)
    rows = summarize_threshold_predictions(probability_batches, mask_batches, thresholds)
    result = write_threshold_search_outputs(rows, args.output_dir, best_metric=args.best_metric, split=args.split)
    best = best_threshold(rows, args.best_metric)
    print(f"Best threshold by {args.best_metric}: {best['threshold']:.3f}")
    print(f"Dice={best['dice']:.6f} IoU={best['iou']:.6f} Precision={best['precision']:.6f} Recall={best['recall']:.6f}")
    print(f"Saved threshold search report to {result['markdown']}")


if __name__ == "__main__":
    main()
