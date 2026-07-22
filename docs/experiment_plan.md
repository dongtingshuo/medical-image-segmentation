# Completed Experiment Plan / 已完成实验计划

## Purpose / 目的

中文：
本计划用于组织模型结构、损失函数、数据增强和高性能配置的对比实验。

English:
This plan organizes comparison experiments for model architectures, loss functions, augmentation settings, and high-capacity configurations.

This document is retained as the historical experiment plan. The planned engineering and Kaggle research scope concluded in v1.6.0; new training is not scheduled under the current project.

本文作为历史实验计划保留。计划内工程与 Kaggle 研究范围已在 v1.6.0 完成，当前项目不再安排新训练。

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

## Completion Status / 完成状态

| Scope | Status | Evidence |
| --- | --- | --- |
| Handwritten U-Net baseline | Complete | README baseline results and curve archive |
| Attention/augmentation/loss comparison tooling | Complete as engineering workflow | Configs, toy demos, and tests; no clinical claim |
| U-Net++ high-accuracy default | Complete | v1.0.0 verified checkpoint |
| Repeated seeds and independent evaluation | Complete | v1.1 validation/test/external reports |
| Cross-validation and encoder comparison | Complete | v1.2 reports |
| Low-contrast variants | Complete | v1.3 report |
| High-capacity/TTA candidates | Complete | v1.4 report |
| Multi-source teachers and students | Complete, not published | v1.5 final evaluation |
| Target-domain generalization | Complete, not published | v1.6 final evaluation |

Repeated-seed, independent ISIC 2017 test, and ISIC 2018 external evaluations are complete. v1.5 and v1.6 did not pass the locked publication gate, so the verified v1.0.0 checkpoint remains the default. The remaining limitation is that all reported results are dataset-level engineering evidence rather than clinical validation.

多随机种子、ISIC 2017 独立测试和 ISIC 2018 外部评估均已完成。v1.5 与 v1.6 未通过锁定发布门槛，因此已验证的 v1.0.0 checkpoint 继续作为默认模型。剩余限制是全部结果仍属于数据集级工程证据，而非临床验证。

All historical training curves are indexed in [`TRAINING_CURVES.md`](TRAINING_CURVES.md). Additional model training requires a new project scope, independent-validation rationale, and explicit GPU budget.

全部历史训练曲线见 [`TRAINING_CURVES.md`](TRAINING_CURVES.md)。新增模型训练需要重新立项，并明确独立验证依据和 GPU 预算。

## Medical Disclaimer / 医学免责声明

中文：
本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。

English:
This project is intended only for medical image segmentation experiments and engineering workflow validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world medical decision-making.
