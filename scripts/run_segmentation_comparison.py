import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_toy_segmentation_data import create_toy_segmentation_data
from src.losses import build_loss
from src.metrics import dice_score, iou_score, precision_score, recall_score
from src.model_factory import get_model
from src.utils import load_config, set_seed


DISCLAIMER_ZH = "本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。"
DISCLAIMER_EN = (
    "This project is intended only for medical image segmentation experiments and engineering workflow "
    "validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world "
    "medical decision-making."
)


def _cfg_get(config, section, key, default=None):
    return config.get(section, {}).get(key, config.get(key, default))


def _ensure_toy_data(config):
    data_cfg = config.get("data", {})
    images_dir = Path(data_cfg.get("image_dir", "examples/toy_segmentation_demo/images"))
    masks_dir = Path(data_cfg.get("mask_dir", "examples/toy_segmentation_demo/masks"))
    if not images_dir.exists() or not masks_dir.exists() or not list(images_dir.glob("*.png")):
        create_toy_segmentation_data(
            output_dir=images_dir.parent,
            num_samples=int(data_cfg.get("num_samples", 12)),
            image_size=int(data_cfg.get("image_size", 128)),
            seed=int(config.get("seed", 42)),
        )
    return images_dir, masks_dir


class ToySegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=128):
        self.images = sorted(Path(images_dir).glob("*.png"))
        self.masks_dir = Path(masks_dir)
        self.image_size = int(image_size)
        if not self.images:
            raise ValueError(f"No toy images found in {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks_dir / image_path.name
        image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return image, mask


def _build_model(model_cfg):
    model_name = model_cfg.get("name", model_cfg.get("model_name", "unet"))
    kwargs = {
        "in_channels": int(model_cfg.get("in_channels", 3)),
        "out_channels": int(model_cfg.get("out_channels", 1)),
        "base_channels": int(model_cfg.get("base_channels", 8)),
        "encoder_name": model_cfg.get("encoder", model_cfg.get("encoder_name", "resnet34")),
        "encoder_weights": model_cfg.get("encoder_weights", None),
    }
    return get_model(model_name, **kwargs)


def _train_and_evaluate(model, loss_fn, train_loader, val_loader, epochs, lr, max_batches, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        for batch_index, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, masks)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered during comparison training.")
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            if batch_index + 1 >= max_batches:
                break

    model.eval()
    metric_totals = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    count = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            metric_totals["dice"] += float(dice_score(logits, masks).cpu())
            metric_totals["iou"] += float(iou_score(logits, masks).cpu())
            metric_totals["precision"] += float(precision_score(logits, masks).cpu())
            metric_totals["recall"] += float(recall_score(logits, masks).cpu())
            count += 1
    metrics = {key: value / max(count, 1) for key, value in metric_totals.items()}
    p = metrics["precision"]
    r = metrics["recall"]
    metrics["f1"] = (2.0 * p * r) / (p + r + 1e-7)
    metrics["loss"] = last_loss
    return metrics


def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "loss", "status", "dice", "iou", "precision", "recall", "f1", "train_loss", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(rows, output_path, config):
    table_rows = []
    for row in rows:
        table_rows.append(
            f"| {row['model']} | {row['loss']} | {row['status']} | {row['dice']:.4f} | {row['iou']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['note']} |"
        )
    table = "\n".join(table_rows)
    model_names = ", ".join([m.get("name", m.get("model_name", "unknown")) for m in config.get("models", [])])
    loss_names = ", ".join([l.get("name", "unknown") for l in config.get("losses", [])])
    output_path.write_text(
        f"""# Experiment Objective / 实验目的

中文：
本报告使用 toy 数据执行轻量模型和损失函数对比，用于验证医学图像分割实验框架的配置、训练、评估和报告流程。

English:
This report uses toy data for lightweight model and loss comparison to validate the configuration, training, evaluation, and reporting workflow of the segmentation framework.

## Model Variants / 模型变体

中文：
本次配置包含以下模型变体：{model_names}。高级模型依赖不可用时会被标记为 skipped。

English:
The configuration includes these model variants: {model_names}. Advanced models are marked as skipped when their optional dependencies are unavailable.

## Loss Functions / 损失函数

中文：
本次配置包含以下损失函数：{loss_names}。

English:
The configuration includes these loss functions: {loss_names}.

## Data Setup / 数据设置

中文：
数据来自 `examples/toy_segmentation_demo`，由脚本生成椭圆区域和二值 mask，不代表真实医学图像分布。

English:
The data comes from `examples/toy_segmentation_demo`, where synthetic elliptical regions and binary masks are generated by script. It does not represent real medical image distributions.

## Metrics / 评估指标

中文：
报告记录 Dice、IoU、Precision、Recall 和 F1，用于检查流程是否稳定运行。

English:
The report records Dice, IoU, Precision, Recall, and F1 to check whether the workflow runs stably.

## Comparison Table / 对比表格

| Model / 模型 | Loss / 损失 | Status / 状态 | Dice | IoU | Precision | Recall | F1 | Note / 备注 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
{table}

## Preliminary Analysis / 初步分析

中文：
这些结果来自极小规模 toy 数据和 smoke training，只能说明工程流程可以运行，不能作为真实医学性能结论。

English:
These results come from very small toy data and smoke training. They only indicate that the engineering workflow runs and must not be interpreted as real medical performance.

## Current Limitations / 当前限制

中文：
toy 数据简单、样本量小、训练轮数少，高级模型可能因可选依赖缺失而跳过。

English:
The toy data is simple, the sample size is small, the training is short, and advanced models may be skipped when optional dependencies are missing.

## Medical Disclaimer / 医学免责声明

中文：
{DISCLAIMER_ZH}

English:
{DISCLAIMER_EN}
""",
        encoding="utf-8",
    )


def run_comparison(config_path):
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    output_dir = Path(_cfg_get(config, "paths", "output_dir", "outputs/comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir, masks_dir = _ensure_toy_data(config)
    data_cfg = deepcopy(config.get("data", {}))
    data_cfg["image_size"] = int(data_cfg.get("image_size", 128))
    dataset = ToySegmentationDataset(images_dir, masks_dir, image_size=data_cfg["image_size"])
    val_dataset = ToySegmentationDataset(images_dir, masks_dir, image_size=data_cfg["image_size"])
    batch_size = int(_cfg_get(config, "training", "batch_size", 4))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cpu")
    rows = []
    epochs = int(_cfg_get(config, "training", "epochs", 1))
    lr = float(_cfg_get(config, "training", "lr", 1e-3))
    max_batches = int(_cfg_get(config, "training", "max_batches", 2))
    for model_cfg in config.get("models", [{"name": "unet"}]):
        model_name = model_cfg.get("name", model_cfg.get("model_name", "unet"))
        for loss_cfg in config.get("losses", [{"name": "bce_dice"}]):
            loss_name = loss_cfg.get("name", "bce_dice")
            row = {
                "model": model_name,
                "loss": loss_name,
                "status": "ok",
                "dice": 0.0,
                "iou": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "train_loss": 0.0,
                "note": "",
            }
            try:
                model = _build_model(model_cfg)
                loss_fn = build_loss({"loss": loss_cfg})
                metrics = _train_and_evaluate(model, loss_fn, train_loader, val_loader, epochs, lr, max_batches, device)
                row.update(
                    {
                        "dice": metrics["dice"],
                        "iou": metrics["iou"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "train_loss": metrics["loss"],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["status"] = "skipped"
                row["note"] = str(exc).replace("\n", " ")[:180]
            rows.append(row)
    serializable_rows = [{**row} for row in rows]
    (output_dir / "comparison_results.json").write_text(
        json.dumps(serializable_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(rows, output_dir / "comparison_results.csv")
    _write_report(rows, output_dir / "comparison_report.md", config)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run lightweight segmentation model/loss comparison.")
    parser.add_argument("--config", default="configs/demo_comparison.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = run_comparison(args.config)
    print(f"Comparison outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
