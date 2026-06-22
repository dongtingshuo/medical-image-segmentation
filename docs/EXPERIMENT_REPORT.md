# Experiment Report: Medical Image Segmentation Framework / 实验报告：医学图像分割框架

## Project Scope / 项目范围

中文：
本项目关注医学图像二分类分割中的模型结构、损失函数、指标评估、可视化分析和可复现实验流程。当前任务以皮肤病灶图像分割为主要示例。

English:
This project focuses on model architectures, loss functions, metric evaluation, visualization analysis, and reproducible workflows for binary medical image segmentation. Skin lesion segmentation is used as the primary example task.

## Model Variants / 模型变体

中文：
项目支持手写 U-Net、Attention U-Net，以及基于 `segmentation-models-pytorch` 的 U-Net++、DeepLabV3+ 和 FPN 等可选高性能模型。

English:
The project supports a hand-written U-Net, Attention U-Net, and optional high-capacity models such as U-Net++, DeepLabV3+, and FPN through `segmentation-models-pytorch`.

## Loss Functions / 损失函数

中文：
当前支持 BCE Loss、Dice Loss、BCE + Dice Loss 和 Focal Loss。所有 loss 均面向二分类分割 logits 输入，Dice Loss 内部执行 sigmoid。

English:
The current loss functions include BCE Loss, Dice Loss, BCE + Dice Loss, and Focal Loss. All losses are designed for binary segmentation logits, and Dice Loss applies sigmoid internally.

## Evaluation Metrics / 评估指标

中文：
项目使用 Dice、IoU、Precision、Recall/Sensitivity、Specificity 和 Boundary F1 等指标分析分割结果。Boundary F1 用于衡量给定像素容差内的边界一致性。

English:
The project analyzes segmentation outputs with Dice, IoU, Precision, Recall/Sensitivity, Specificity, and Boundary F1. Boundary F1 measures contour agreement within a specified pixel tolerance.

## Comparison Workflow / 对比实验流程

中文：
轻量对比流程使用 toy 数据和 smoke training，仅用于验证模型/loss 组合、指标统计和报告生成是否正常。

English:
The lightweight comparison workflow uses toy data and smoke training only to validate model/loss combinations, metric aggregation, and report generation.

```bash
python scripts/create_toy_segmentation_data.py
python scripts/run_segmentation_comparison.py --config configs/demo_comparison.yaml
```

## Visualization Workflow / 可视化流程

中文：
可视化流程生成预测叠加图、误检图、漏检图和并排对比图，默认基于 toy 数据和 mock prediction。

English:
The visualization workflow generates prediction overlays, false-positive maps, false-negative maps, and side-by-side comparisons based on toy data and mock predictions by default.

```bash
python scripts/run_visualization_demo.py
```

## Error Analysis Workflow / 错误分析流程

中文：
错误分析流程统计过分割、欠分割、小目标漏检、边界误差、空预测和空真实 mask 等情况，并输出 JSON 和 Markdown 报告。

English:
The error analysis workflow summarizes over-segmentation, under-segmentation, small-object misses, boundary errors, empty predictions, and empty ground-truth masks, then exports JSON and Markdown reports.

```bash
python scripts/run_error_analysis.py
```

## Result Interpretation / 结果解释

中文：
demo comparison 使用 toy 数据或 mock 数据，只用于验证流程。完整性能评估需要真实医学图像数据集、稳定训练配置和更严格的评估协议。

English:
The demo comparison uses toy or mock data only to validate the workflow. Full performance evaluation requires real medical image datasets, stable training configurations, and stricter evaluation protocols.

## Current Limitations / 当前限制

中文：
toy demo 不代表真实医学性能；小样本结果不稳定；完整训练需要真实数据集和 GPU 资源；mask 质量会显著影响评估；简化指标可能不同于特定挑战赛协议。

English:
The toy demo does not represent real medical performance; small-sample results are unstable; full training requires real datasets and GPU resources; mask quality strongly affects evaluation; simplified metrics may differ from challenge-specific protocols.

## Future Work / 后续工作

中文：
后续可支持更多 segmentation backbone、增加公开数据集准备指南、改进边界感知指标、加入交叉验证流程、集成实验追踪，并提供可选 Docker 环境。

English:
Future work may support richer segmentation backbones, add public dataset preparation guides, improve boundary-aware metrics, add cross-validation workflows, integrate experiment tracking, and provide an optional Docker environment.

## Medical Disclaimer / 医学免责声明

中文：
本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。

English:
This project is intended only for medical image segmentation experiments and engineering workflow validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world medical decision-making.
