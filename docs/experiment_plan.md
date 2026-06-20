# Experiment Plan / 实验计划

## Purpose / 目的

中文：
本计划用于组织模型结构、损失函数、数据增强和高性能配置的对比实验。

English:
This plan organizes comparison experiments for model architectures, loss functions, augmentation settings, and high-capacity configurations.

## Experiment 1: U-Net Baseline / 实验 1：U-Net 基线

中文：
使用手写 U-Net 和 BCE + Dice Loss 建立可解释基线。

English:
Use the hand-written U-Net with BCE + Dice Loss to establish an interpretable baseline.

## Experiment 2: Attention U-Net / 实验 2：Attention U-Net

中文：
验证注意力门控结构对前景区域定位的影响。

English:
Evaluate how attention gates affect foreground localization.

## Experiment 3: Augmentation Comparison / 实验 3：数据增强对比

中文：
比较 `unet.yaml` 与 `unet_no_aug.yaml`，分析增强对验证指标和稳定性的影响。

English:
Compare `unet.yaml` with `unet_no_aug.yaml` to analyze the effect of augmentation on validation metrics and stability.

## Experiment 4: Loss Comparison / 实验 4：损失函数对比

中文：
比较 BCE、Dice 和 BCE + Dice Loss 对 Dice、IoU、Precision 和 Recall 的影响。

English:
Compare BCE, Dice, and BCE + Dice Loss in terms of Dice, IoU, Precision, and Recall.

## Experiment 5: High-Capacity Models / 实验 5：高容量模型

中文：
比较 U-Net++、DeepLabV3+ 和可选 FPN 配置。预训练 encoder 仅在正式训练配置中启用。

English:
Compare U-Net++, DeepLabV3+, and optional FPN configurations. Pretrained encoders are enabled only in formal training configurations.

## Current Limitations / 当前限制

中文：
轻量 demo 仅用于验证流程。正式结论需要真实数据集、稳定训练设置和明确的数据划分。

English:
The lightweight demo is only for workflow validation. Formal conclusions require real datasets, stable training settings, and explicit data splits.

## Medical Disclaimer / 医学免责声明

中文：
本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。

English:
This project is intended only for medical image segmentation experiments and engineering workflow validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world medical decision-making.
