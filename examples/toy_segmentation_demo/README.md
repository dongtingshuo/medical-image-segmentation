# Toy Segmentation Demo / 轻量分割演示

## Demo Purpose / 演示目的

中文：
本目录提供一个轻量级 toy segmentation demo，用于验证数据读取、mask 对齐、模型/loss 对比、可视化和错误分析流程。

English:
This directory provides a lightweight toy segmentation demo for validating data loading, mask alignment, model/loss comparison, visualization, and error analysis workflows.

## Files / 文件说明

中文：
`images/` 保存合成 RGB 图像，`masks/` 保存对应的二值 mask。文件名保持一致，便于按 stem 匹配。

English:
`images/` stores synthetic RGB images, and `masks/` stores the corresponding binary masks. Filenames are aligned by stem for matching.

## How to Generate Data / 如何生成数据

中文：
运行以下命令可重新生成少量 toy 图像和 mask。

English:
Run the following command to regenerate a small set of toy images and masks.

```bash
python scripts/create_toy_segmentation_data.py
```

## How to Run / 如何运行

中文：
可以基于 toy 数据运行模型/loss 对比、可视化 demo 和错误分析 demo。

English:
The toy data can be used to run model/loss comparison, visualization demo, and error analysis demo.

```bash
python scripts/run_segmentation_comparison.py --config configs/demo_comparison.yaml
python scripts/run_visualization_demo.py
python scripts/run_error_analysis.py
```

## Current Limitations / 当前限制

中文：
toy 数据由简单椭圆形区域生成，只用于工程流程验证，不代表真实医学图像分布或真实性能。

English:
The toy data is generated from simple elliptical regions and is intended only for engineering workflow validation. It does not represent real medical image distributions or real performance.

## Medical Disclaimer / 医学免责声明

中文：
本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。

English:
This project is intended only for medical image segmentation experiments and engineering workflow validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world medical decision-making.
