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
比较 BCE、Dice 和 BCE + Dice Loss 对 Dice、IoU、Precision、Recall/Specificity 和 Boundary F1 的影响。

English:
Compare BCE, Dice, and BCE + Dice Loss in terms of Dice, IoU, Precision, Recall/Specificity, and Boundary F1.

## Experiment 5: High-Capacity Models / 实验 5：高容量模型

中文：
比较 U-Net++、DeepLabV3+ 和可选 FPN 配置。预训练 encoder 仅在正式训练配置中启用。

English:
Compare U-Net++, DeepLabV3+, and optional FPN configurations. Pretrained encoders are enabled only in formal training configurations.

## Experiment 6: Repeated Seeds and Independent Test / 实验 6：多随机种子与独立测试

中文：
在数据划分固定时至少使用 3 个随机种子训练，报告均值与标准差。模型选择完成后，使用未参与调参的 test split 评估。

English:
Train with at least three random seeds on a fixed split and report mean and standard deviation. After model selection, evaluate on a test split that was not used for tuning.

## Current Limitations / 当前限制

中文：
轻量 demo 仅用于验证流程。当前真实实验仅报告单一验证集结果；多随机种子、独立测试和外部验证尚未完成。

English:
The lightweight demo is only for workflow validation. Current real-data experiments report one validation split only; repeated seeds, independent testing, and external validation remain incomplete.

## Medical Disclaimer / 医学免责声明

中文：
本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。

English:
This project is intended only for medical image segmentation experiments and engineering workflow validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world medical decision-making.
