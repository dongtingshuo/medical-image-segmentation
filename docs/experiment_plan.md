# Experiment Plan

## 实验 1：手写 U-Net 基线

- 模型：U-Net
- Loss：BCE + Dice
- 指标：Dice、IoU、Precision、Recall
- 目的：建立可解释基线。

## 实验 2：Attention U-Net

- 模型：Attention U-Net
- Loss：BCE + Dice
- 目的：验证注意力门控对病灶区域定位的帮助。

## 实验 3：数据增强对比

- 配置：`unet.yaml` vs `unet_no_aug.yaml`
- 目的：分析增强对泛化能力的影响。

## 实验 4：Loss 对比

- 配置：`unet_bce.yaml`、`unet_dice.yaml`、`unet_bce_dice.yaml`
- 目的：比较不同损失函数对 Dice / IoU 的影响。

## 实验 5：高性能模型

- 模型：U-Net++ / DeepLabV3+
- Encoder：efficientnet-b3
- Encoder weights：imagenet
- 目的：追求更高 Dice / IoU。
