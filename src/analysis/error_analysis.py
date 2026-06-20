from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _as_bool(mask):
    return np.asarray(mask).squeeze() > 0


def analyze_segmentation_errors(pred_mask, true_mask, small_object_ratio=0.02):
    pred = _as_bool(pred_mask)
    true = _as_bool(true_mask)
    total = int(true.size)
    pred_area = int(pred.sum())
    true_area = int(true.sum())
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    tn = int((~pred & ~true).sum())
    eps = 1e-7
    dice = (2.0 * tp + eps) / (pred_area + true_area + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    boundary_error = False
    if pred.shape == true.shape and pred.size > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        pred_boundary = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        true_boundary = cv2.morphologyEx(true.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
        boundary_union = int((pred_boundary | true_boundary).sum())
        boundary_intersection = int((pred_boundary & true_boundary).sum())
        boundary_mismatch = 1.0 - ((boundary_intersection + eps) / (boundary_union + eps))
        boundary_error = bool(boundary_union > 0 and boundary_mismatch > 0.45)
    else:
        boundary_mismatch = 0.0

    true_ratio = true_area / max(total, 1)
    flags = {
        "over_segmentation": bool(fp > 0 and pred_area > true_area * 1.15),
        "under_segmentation": bool(fn > 0 and pred_area < true_area * 0.85),
        "small_object_miss": bool(true_area > 0 and true_ratio <= small_object_ratio and recall < 0.5),
        "boundary_error": boundary_error,
        "empty_prediction": bool(pred_area == 0),
        "empty_ground_truth": bool(true_area == 0),
    }
    return {
        "areas": {
            "prediction": pred_area,
            "ground_truth": true_area,
            "false_positive": fp,
            "false_negative": fn,
        },
        "metrics": {
            "dice": float(dice),
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(recall),
            "specificity": float(specificity),
            "boundary_mismatch": float(boundary_mismatch),
        },
        "error_flags": flags,
    }


def summarize_error_records(records):
    summary = {
        "num_samples": len(records),
        "error_counts": {
            "over_segmentation": 0,
            "under_segmentation": 0,
            "small_object_miss": 0,
            "boundary_error": 0,
            "empty_prediction": 0,
            "empty_ground_truth": 0,
        },
        "mean_metrics": {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
        },
    }
    if not records:
        return summary
    for record in records:
        for key in summary["error_counts"]:
            summary["error_counts"][key] += int(record["error_flags"].get(key, False))
        for key in summary["mean_metrics"]:
            summary["mean_metrics"][key] += float(record["metrics"].get(key, 0.0))
    for key in summary["mean_metrics"]:
        summary["mean_metrics"][key] /= len(records)
    return summary


def write_error_analysis_report(summary, records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(records):
        flags = ", ".join([key for key, value in record["error_flags"].items() if value]) or "none"
        rows.append(
            f"| {index} | {record['metrics']['dice']:.4f} | {record['metrics']['iou']:.4f} | "
            f"{record['metrics']['precision']:.4f} | {record['metrics']['recall']:.4f} | {flags} |"
        )
    table = "\n".join(rows)
    output_path.write_text(
        f"""# Error Analysis Overview / 错误分析概述

中文：
本报告基于 toy 或 mock mask 对分割错误进行结构化统计，用于验证错误分析流程。

English:
This report summarizes segmentation errors from toy or mock masks to validate the analysis workflow.

## Error Types / 错误类型

中文：
当前统计过分割、欠分割、小目标漏检、边界误差、空预测和空真实 mask。

English:
The current analysis tracks over-segmentation, under-segmentation, small-object misses, boundary errors, empty predictions, and empty ground-truth masks.

## Metric Summary / 指标摘要

| Metric / 指标 | Value / 数值 |
| --- | ---: |
| Samples / 样本数 | {summary['num_samples']} |
| Dice | {summary['mean_metrics']['dice']:.4f} |
| IoU | {summary['mean_metrics']['iou']:.4f} |
| Precision | {summary['mean_metrics']['precision']:.4f} |
| Recall / Sensitivity | {summary['mean_metrics']['recall']:.4f} |
| Specificity | {summary['mean_metrics']['specificity']:.4f} |

## Common Failure Patterns / 常见失败模式

| Error Type / 错误类型 | Count / 数量 |
| --- | ---: |
| Over-segmentation / 过分割 | {summary['error_counts']['over_segmentation']} |
| Under-segmentation / 欠分割 | {summary['error_counts']['under_segmentation']} |
| Small-object miss / 小目标漏检 | {summary['error_counts']['small_object_miss']} |
| Boundary error / 边界误差 | {summary['error_counts']['boundary_error']} |
| Empty prediction / 空预测 | {summary['error_counts']['empty_prediction']} |
| Empty ground truth / 空真实 mask | {summary['error_counts']['empty_ground_truth']} |

| Sample / 样本 | Dice | IoU | Precision | Recall | Flags / 标记 |
| ---: | ---: | ---: | ---: | ---: | --- |
{table}

## Current Limitations / 当前限制

中文：
该分析基于简化规则和 toy 数据，仅用于流程验证；真实数据上的错误分析需要结合图像质量、标注规范和任务协议。

English:
This analysis uses simplified rules and toy data only for workflow validation. Error analysis on real datasets should consider image quality, annotation rules, and task protocols.

## Medical Disclaimer / 医学免责声明

中文：
本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。

English:
This project is intended only for medical image segmentation experiments and engineering workflow validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world medical decision-making.
""",
        encoding="utf-8",
    )
