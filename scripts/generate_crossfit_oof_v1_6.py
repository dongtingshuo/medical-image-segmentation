import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_oof_targets import parse_member, predict_member  # noqa: E402
from src.cross_validation import read_folds  # noqa: E402
from src.oof import resize_target, restore_probability, write_soft_mask  # noqa: E402
from src.utils import get_device  # noqa: E402
from src.v16 import select_crossfit_family_weights  # noqa: E402


def _load_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["stem"]: row for row in csv.DictReader(handle) if row.get("stem")}


def main():
    parser = argparse.ArgumentParser(description="Generate target-domain cross-fit OOF predictions for v1.6.")
    parser.add_argument("--member", action="append", required=True, help="ARCH:FOLD:CONFIG:CHECKPOINT")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", default="none", choices=["none", "flip", "multiscale_flip"])
    args = parser.parse_args()

    members = [parse_member(value) for value in args.member]
    architectures = sorted({member[0] for member in members})
    if architectures != ["segformer", "unetpp"]:
        raise ValueError(f"v1.6 requires exactly segformer and unetpp teachers, got {architectures}")
    folds = read_folds(args.folds)
    expected = {(architecture, int(fold["fold"])) for architecture in architectures for fold in folds}
    actual = {(architecture, fold) for architecture, fold, _, _ in members}
    if actual != expected or len(actual) != len(members):
        raise ValueError("Each v1.6 architecture must provide each target-domain fold exactly once.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.manifest)
    image_paths = {path.stem: path for path in Path(args.images_dir).iterdir() if path.is_file()}
    mask_paths = {path.stem: path for path in Path(args.masks_dir).iterdir() if path.is_file()}
    device = get_device(args.device)
    by_architecture = {architecture: {} for architecture in architectures}
    resize_modes = set()
    fold_by_id = {int(fold["fold"]): fold for fold in folds}
    for member in members:
        architecture, fold_index, _, _ = member
        prediction, config = predict_member(
            member, fold_by_id[fold_index], args.images_dir, args.masks_dir, device, tta=args.tta
        )
        by_architecture[architecture].update(prediction)
        resize_modes.add(str(config.get("data", {}).get("resize_mode", "stretch")))
    if len(resize_modes) != 1:
        raise ValueError(f"v1.6 OOF members must share one resize mode, got {sorted(resize_modes)}")
    stems = [stem for fold in folds for stem in fold["val_ids"]]
    if len(stems) != len(set(stems)) or set(stems) != set(by_architecture["unetpp"]) or set(stems) != set(by_architecture["segformer"]):
        raise ValueError("Cross-fit OOF predictions must cover every target validation sample exactly once per architecture.")
    if any(manifest.get(stem, {}).get("source") != "isic17" for stem in stems):
        raise ValueError("Cross-fit OOF validation samples must come exclusively from ISIC17 training data.")
    resize_mode = resize_modes.pop()
    raw_probabilities = {
        architecture: np.asarray([by_architecture[architecture][stem] for stem in stems], dtype=np.float32)[:, None]
        for architecture in architectures
    }
    for architecture, values in raw_probabilities.items():
        np.save(output_root / f"{architecture}_crossfit.npy", values.astype(np.float16))
    # Store soft masks at original image geometry for student augmentation synchronization.
    soft_dir = output_root / "soft_masks"
    for stem in stems:
        image = cv2.imread(str(image_paths[stem]), cv2.IMREAD_COLOR)
        averaged = 0.5 * (by_architecture["unetpp"][stem] + by_architecture["segformer"][stem])
        write_soft_mask(
            soft_dir / f"{stem}.png",
            restore_probability(averaged, image.shape[:2], resize_mode=resize_mode),
        )
    # Metrics require common geometry, so resize each prediction/target to the base 384 square used by all teachers.
    model_targets = []
    for stem in stems:
        image = cv2.imread(str(image_paths[stem]), cv2.IMREAD_COLOR)
        raw_mask = cv2.imread(str(mask_paths[stem]), cv2.IMREAD_GRAYSCALE)
        model_targets.append(resize_target((raw_mask > 127).astype(np.float32), (384, 384), resize_mode=resize_mode))
    model_targets = np.asarray(model_targets, dtype=np.float32)[:, None]
    selected, rows = select_crossfit_family_weights(raw_probabilities, model_targets)
    decision = {
        "family_weights": {"unetpp": selected["unetpp_weight"], "segformer": selected["segformer_weight"]},
        "oof_metrics": selected,
        "search": rows,
        "stems": stems,
        "resize_mode": resize_mode,
        "tta": args.tta,
    }
    (output_root / "family_selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    np.save(output_root / "targets_384.npy", model_targets.astype(np.uint8))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
