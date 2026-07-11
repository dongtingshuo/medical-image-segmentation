"""v1.6 data and cross-fit helpers kept separate from the v1.5 workflow."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.cross_validation import create_stratified_kfold_splits
from src.ensemble_v15 import macro_metrics
from src.multisource_data import read_manifest

HAM_LICENSE = "CC BY-NC-SA 4.0"
HAM_LABEL_QUALITY = "reviewed"


def _stable_bucket(value, modulo=10):
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8], 16) % int(modulo)


def read_ham_metadata(path):
    rows = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row.get("image_id", "")).strip()
            lesion_id = str(row.get("lesion_id", "")).strip()
            if image_id:
                rows[image_id] = {"group_id": lesion_id or image_id}
    if not rows:
        raise ValueError(f"HAM10000 metadata contains no image_id values: {path}")
    return rows


def decorate_v16_manifest(manifest_path, ham_metadata_path):
    metadata = read_ham_metadata(ham_metadata_path)
    rows = read_manifest(manifest_path)
    fields = list(rows[0]) if rows else []
    for field in ["label_quality", "group_id", "license", "stage"]:
        if field not in fields:
            fields.append(field)
    for row in rows:
        source = row.get("source", "")
        original = row.get("original_stem", "")
        if source == "ham10000":
            item = metadata.get(original)
            if item is None and row.get("status") == "accepted":
                raise ValueError(f"HAM10000 manifest sample is absent from metadata: {original}")
            row.update(
                {
                    "label_quality": HAM_LABEL_QUALITY,
                    "group_id": item["group_id"] if item else "",
                    "license": HAM_LICENSE,
                    "stage": "pretrain",
                }
            )
        else:
            row.update(
                {
                    "label_quality": "expert",
                    "group_id": original or row.get("stem", ""),
                    "license": "dataset-provided terms",
                    "stage": "finetune",
                }
            )
    with Path(manifest_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def create_ham_pretrain_split(records, validation_fraction=0.10):
    accepted = [row for row in records if row.get("status") == "accepted" and row.get("source") == "ham10000"]
    if not accepted:
        raise ValueError("No accepted HAM10000 samples are available for pretraining.")
    groups = {str(row["group_id"]) for row in accepted}
    val_groups = {group for group in groups if _stable_bucket(group) < round(float(validation_fraction) * 10)}
    if not val_groups:
        val_groups.add(sorted(groups)[0])
    train_ids = sorted(row["stem"] for row in accepted if row["group_id"] not in val_groups)
    val_ids = sorted(row["stem"] for row in accepted if row["group_id"] in val_groups)
    if not train_ids or not val_ids:
        raise ValueError("HAM10000 group split must contain both train and validation samples.")
    return {"train_ids": train_ids, "val_ids": val_ids, "validation_groups": sorted(val_groups)}


def create_target_domain_folds(records, k=5, seed=42):
    accepted = [row for row in records if row.get("status") == "accepted" and row.get("stem")]
    isic17 = [row for row in accepted if row.get("source") == "isic17"]
    if len(isic17) < int(k):
        raise ValueError("Need at least k accepted ISIC17 training samples for target-domain folds.")
    target_folds = create_stratified_kfold_splits(isic17, k=k, seed=seed)
    # HAM10000 is pretraining-only. It must not re-enter target-domain teacher folds.
    all_ids = {row["stem"] for row in accepted if row.get("source") != "ham10000"}
    folds = []
    for fold in target_folds:
        val_ids = sorted(fold["val_ids"])
        train_ids = sorted(all_ids - set(val_ids))
        folds.append({"fold": int(fold["fold"]), "train_ids": train_ids, "val_ids": val_ids})
    validate_target_domain_folds(folds, records)
    return folds


def validate_target_domain_folds(folds, records):
    isic17_ids = {
        row["stem"]
        for row in records
        if row.get("status") == "accepted" and row.get("source") == "isic17" and row.get("stem")
    }
    seen = []
    for fold in folds:
        train_ids, val_ids = set(fold["train_ids"]), set(fold["val_ids"])
        if train_ids & val_ids:
            raise ValueError(f"Target fold {fold['fold']} overlaps train and validation samples.")
        if not val_ids <= isic17_ids:
            raise ValueError(f"Target fold {fold['fold']} validation includes a non-ISIC17 sample.")
        seen.extend(val_ids)
    if sorted(seen) != sorted(isic17_ids):
        raise ValueError("Target-domain folds must partition accepted ISIC17 samples exactly once.")
    return True


def write_v16_folds(path, folds, metadata=None):
    payload = {"metadata": metadata or {}, "folds": folds}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def select_crossfit_family_weights(probabilities, targets):
    """Choose U-Net++/SegFormer mixture by target-domain OOF Dice then boundary F1."""
    unet = np.asarray(probabilities["unetpp"], dtype=np.float32)
    segformer = np.asarray(probabilities["segformer"], dtype=np.float32)
    rows = []
    for weight in np.arange(0.0, 1.0001, 0.05):
        combined = weight * unet + (1.0 - weight) * segformer
        metrics = macro_metrics(combined, targets, threshold=0.5)
        rows.append({"unetpp_weight": float(round(weight, 2)), "segformer_weight": float(round(1.0 - weight, 2)), **metrics})
    return max(rows, key=lambda row: (row["dice"], row["boundary_f1"], row["unetpp_weight"])), rows
