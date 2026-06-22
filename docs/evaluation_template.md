# Evaluation Template / 评估模板

## Experiment Configuration / 实验配置

中文：
本节记录模型、输入尺寸、训练参数和损失函数，便于复现实验设置。

English:
This section records the model, input size, training parameters, and loss function for reproducibility.

- experiment_name:
- data_source:
- split:
- seed:
- model_name:
- image_size:
- batch_size:
- epochs:
- lr:
- loss_name:
- augmentation_enabled:
- checkpoint_sha256:
- best_epoch:

## Results / 结果

中文：
请填写验证集或测试集上的分割指标，并注明数据划分来源。

English:
Fill in segmentation metrics on the validation or test split and state the source of the data split.

| Metric / 指标 | Value / 数值 |
| --- | --- |
| Dice | |
| IoU | |
| Precision | |
| Recall | |
| Specificity | |
| Boundary F1 | |
| Loss at selected epoch | |

State whether the row is from validation, internal test, or external test data. Report multiple-seed mean and standard deviation when those runs exist; otherwise mark them as not available.

请明确结果来自验证集、内部测试集还是外部测试集。如已完成多随机种子实验，报告均值和标准差；否则标记为不可用。

## Visualization Analysis / 可视化分析

中文：
请放入原图、真实 mask、预测 mask、叠加图，并说明典型失败案例。

English:
Include the original image, ground-truth mask, predicted mask, overlay, and a short description of typical failure cases.

## Current Limitations / 当前限制

中文：
单次实验结果可能受数据划分、mask 质量、输入尺寸和训练轮数影响，不应过度解读。

English:
Single-run results may be affected by data split, mask quality, input size, and training length, and should not be over-interpreted.

## Medical Disclaimer / 医学免责声明

中文：
本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。

English:
This project is intended only for medical image segmentation experiments and engineering workflow validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world medical decision-making.
