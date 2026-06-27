import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cross_validation import read_folds  # noqa: E402
from src.dataset import SkinLesionDataset, get_val_transforms  # noqa: E402
from src.inference import build_model_from_config  # noqa: E402
from src.oof import write_oof_outputs  # noqa: E402
from src.utils import (  # noqa: E402
    checkpoint_model_config,
    get_device,
    load_checkpoint,
    load_checkpoint_payload,
    load_config,
)


def parse_member(value):
    parts = value.split(":", maxsplit=3)
    if len(parts) != 4:
        raise ValueError("OOF members must use ARCHITECTURE:FOLD:CONFIG:CHECKPOINT format.")
    architecture, fold, config_path, checkpoint_path = parts
    return architecture, int(fold), Path(config_path), Path(checkpoint_path)


def _subset_dataset(config, images_dir, masks_dir, stems):
    dataset = SkinLesionDataset(images_dir, masks_dir, transform=get_val_transforms(config))
    pairs = {image_path.stem: (image_path, mask_path) for image_path, mask_path in dataset.pairs}
    missing = sorted(set(stems) - set(pairs))
    if missing:
        raise ValueError(f"Fold references missing source samples: {missing[:10]}")
    dataset.pairs = [pairs[stem] for stem in stems]
    return dataset


@torch.no_grad()
def predict_member(member, fold, images_dir, masks_dir, device):
    architecture, _, config_path, checkpoint_path = member
    config = load_config(config_path)
    expected_name = f"teacher-{architecture}-fold{fold['fold']}"
    if config.get("experiment_name") != expected_name:
        raise ValueError(
            f"OOF member config does not match its declared fold: expected={expected_name}, "
            f"actual={config.get('experiment_name')}"
        )
    dataset = _subset_dataset(config, images_dir, masks_dir, fold["val_ids"])
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 4)),
        shuffle=False,
        num_workers=int(config.get("training", {}).get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    checkpoint = load_checkpoint_payload(checkpoint_path, device=device)
    model = build_model_from_config(config, checkpoint=checkpoint).to(device)
    expected = checkpoint_model_config(checkpoint) or config.get("model", {})
    load_checkpoint(checkpoint_path, model, device, expected_model_config=expected, checkpoint=checkpoint)
    model.eval()
    predictions = {}
    offset = 0
    for images, _ in loader:
        probabilities = torch.sigmoid(model(images.to(device, non_blocking=True))).cpu().numpy().astype(np.float16)
        batch_stems = fold["val_ids"][offset : offset + len(probabilities)]
        for stem, probability in zip(batch_stems, probabilities):
            predictions[stem] = probability[0]
        offset += len(probabilities)
    if offset != len(fold["val_ids"]):
        raise RuntimeError(f"Incomplete inference for {architecture} fold {fold['fold']}.")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, config


def main():
    parser = argparse.ArgumentParser(description="Generate leak-free dual-architecture OOF distillation targets.")
    parser.add_argument("--member", action="append", required=True, help="ARCH:FOLD:CONFIG:CHECKPOINT")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-architectures", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    members = [parse_member(value) for value in args.member]
    architectures = sorted({member[0] for member in members})
    if len(architectures) != args.expected_architectures:
        raise ValueError(f"Expected {args.expected_architectures} OOF architectures, got {architectures}")
    folds = read_folds(args.folds)
    folds_by_index = {int(fold["fold"]): fold for fold in folds}
    expected_keys = {(architecture, int(fold["fold"])) for architecture in architectures for fold in folds}
    actual_keys = {(architecture, fold_index) for architecture, fold_index, _, _ in members}
    if actual_keys != expected_keys or len(actual_keys) != len(members):
        raise ValueError(
            f"Each architecture must provide every fold exactly once. missing={sorted(expected_keys - actual_keys)} "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    device = get_device(args.device)
    predictions = {architecture: {} for architecture in architectures}
    resize_modes = set()
    for member in members:
        architecture, fold_index, _, _ = member
        fold_predictions, config = predict_member(
            member, folds_by_index[fold_index], args.images_dir, args.masks_dir, device
        )
        predictions[architecture].update(fold_predictions)
        resize_modes.add(str(config.get("data", {}).get("resize_mode", "stretch")))
    if len(resize_modes) != 1:
        raise ValueError(f"OOF members must use one resize mode, got {sorted(resize_modes)}")

    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    result = write_oof_outputs(
        predictions,
        folds,
        manifest_rows,
        args.images_dir,
        args.masks_dir,
        args.output_root,
        resize_mode=resize_modes.pop(),
    )
    print(json.dumps(result["coverage"], indent=2))
    print(f"Soft masks: {result['soft_masks_dir']}")
    print(f"Hard-example weights: {result['weights']}")


if __name__ == "__main__":
    main()
