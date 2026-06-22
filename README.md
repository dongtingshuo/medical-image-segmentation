# Medical Image Segmentation

[![CI](https://github.com/dongtingshuo/medical-image-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/dongtingshuo/medical-image-segmentation/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/dongtingshuo/medical-image-segmentation)](https://github.com/dongtingshuo/medical-image-segmentation/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**English:** Skin lesion segmentation with improved U-Net models.  
**中文：** 基于 U-Net 改进模型的皮肤病灶图像分割系统。

This repository implements a PyTorch-based binary medical image segmentation pipeline for skin lesion images. It supports U-Net, Attention U-Net, U-Net++, DeepLabV3+, Kaggle GPU training, local CPU/CUDA inference, metric evaluation, prediction visualization, and a Gradio Web Demo.

本项目实现了一个基于 PyTorch 的皮肤病灶二分类医学图像分割流程，支持 U-Net、Attention U-Net、U-Net++、DeepLabV3+、Kaggle GPU 训练、本地 CPU/CUDA 推理、指标评估、预测可视化和 Gradio Web Demo。

## Overview / 项目概述

The task is pixel-level binary segmentation. Given an RGB skin lesion image, the model predicts a one-channel lesion mask. Models output logits; sigmoid and thresholding are applied only during metrics and inference.

本任务是像素级二分类分割。输入为 RGB 皮肤病灶图像，输出为单通道病灶 mask。所有模型输出 logits，只有在指标计算和推理阶段才执行 sigmoid 和 threshold。

The final default inference model is:

最终默认推理模型为：

```text
U-Net++ + EfficientNet-B3 encoder
Checkpoint: checkpoints/best_model.pth
Config: configs/final_model.yaml
```

## Features / 功能

- Binary skin lesion segmentation / 皮肤病灶二分类分割
- Handwritten U-Net baseline / 手写 U-Net 基线模型
- Attention U-Net / Attention U-Net 改进结构
- U-Net++ and DeepLabV3+ high-accuracy models / U-Net++ 与 DeepLabV3+ 高精度模型
- Kaggle GPU training workflow / Kaggle GPU 训练流程
- CPU/CUDA automatic local inference / 本地 CPU/CUDA 自动选择
- Dice, IoU, Precision, Recall, Specificity, and Boundary F1 evaluation / Dice、IoU、Precision、Recall、Specificity 与 Boundary F1 评估
- Model comparison and loss comparison / 模型对比与损失函数对比
- Prediction visualization with masks and overlays / mask 与叠加图可视化
- Overlay visualization and false-positive/false-negative maps / 叠加可视化与误检、漏检图
- Segmentation error analysis / 分割错误分析
- Lightweight toy segmentation demo / 轻量分割演示
- Gradio Web Demo / Gradio Web 演示
- Training sanity checks / 训练前质量检查
- Medical disclaimer for non-clinical use / 非临床用途医学免责声明

## Project Structure / 项目结构

```text
configs/                 YAML configs for local and Kaggle runs
src/                     Dataset, models, losses, metrics, trainer, visualization
scripts/                 Dataset check, small-batch overfit, quick train
notebooks/               Kaggle training notebook
docs/                    Technical report and experiment documents
models/                  Versioned model manifest and artifact metadata
outputs/                 Local outputs, curves, samples, sanity checks
checkpoints/             Model checkpoints
tests/                   Unit tests independent of real datasets
train.py                 Training entry point
evaluate.py              Evaluation entry point
predict.py               Single-image prediction
app.py                   Gradio Web Demo
```

## Documentation / 文档

English:
The main project documents describe training, evaluation, result interpretation, and lightweight reproducible demos.

中文：
主要文档说明训练、评估、结果解释和轻量可复现实验 demo。

- [docs/report.md](docs/report.md)
- [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md)
- [FINAL_GUIDE.md](FINAL_GUIDE.md)
- [DATASET.md](DATASET.md)
- [MODEL_CARD.md](MODEL_CARD.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [CHANGELOG.md](CHANGELOG.md)
- [examples/toy_segmentation_demo/README.md](examples/toy_segmentation_demo/README.md)

## System Architecture / 系统架构

```mermaid
flowchart TD
    A["YAML Config"] --> B["SkinLesionDataset"]
    B --> C["Albumentations Transforms"]
    C --> D["Model Factory"]
    D --> E1["U-Net"]
    D --> E2["Attention U-Net"]
    D --> E3["U-Net++ / DeepLabV3+ / FPN"]
    E1 --> F["Trainer"]
    E2 --> F
    E3 --> F
    F --> G["Checkpoints"]
    F --> H["Metrics CSV"]
    F --> I["Curves and Samples"]
    G --> J["evaluate.py"]
    G --> K["predict.py"]
    G --> L["Gradio Demo"]
```

## Environment Setup / 环境配置

Python 3.10-3.12 is supported. Direct dependencies are pinned for reproducible installation.

支持 Python 3.10-3.12，直接依赖已锁定版本，用于可重复安装。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` already includes the dependency for the final U-Net++ model. If you use a minimal environment, install it manually:

`requirements.txt` 已包含最终 U-Net++ 模型所需依赖。若使用精简环境，可手动安装：

```bash
pip install segmentation-models-pytorch
```

Run tests:

运行测试：

```bash
python -m pytest -q
```

GitHub Actions runs the CPU test suite and command-line smoke tests on Python 3.10 and 3.12. Training uses deterministic algorithms and seeded DataLoader workers by default; set `reproducibility.deterministic: false` only when throughput is more important than exact repeatability.

GitHub Actions 会在 Python 3.10 和 3.12 上执行 CPU 测试与命令行检查。训练默认启用确定性算法和 DataLoader worker 随机种子；只有在吞吐量优先时才建议设置 `reproducibility.deterministic: false`。

## Quick Demo / 快速演示

English:
The quick demo uses synthetic toy data and runs on CPU. It validates the experiment workflow without downloading large datasets or model weights.

中文：
快速演示使用合成 toy 数据，可在 CPU 上运行，用于验证实验流程，不下载大型数据集或模型权重。

Generate toy data:

生成 toy 数据：

```bash
python scripts/create_toy_segmentation_data.py
```

Run model and loss comparison:

运行模型和损失函数对比：

```bash
python scripts/run_segmentation_comparison.py --config configs/demo_comparison.yaml
```

Run visualization demo:

运行可视化 demo：

```bash
python scripts/run_visualization_demo.py
```

Run error analysis demo:

运行错误分析 demo：

```bash
python scripts/run_error_analysis.py
```

## Dataset Format / 数据集格式

Images and masks must share the same filename stem.

图像和 mask 必须使用相同文件名 stem。

```text
data/
  images/
    train/
      ISIC_0000001.jpg
    val/
      ISIC_0000002.jpg
  masks/
    train/
      ISIC_0000001.png
    val/
      ISIC_0000002.png
```

All dataset paths are passed through YAML configs or command-line arguments. Source code does not hard-code local or Kaggle data paths.

所有数据路径均通过 YAML 配置或命令行参数传入，源码不硬编码本地或 Kaggle 数据路径。

The reported experiments use the ISIC 2017 train/validation segmentation split through a Kaggle mirror. Dataset provenance, split counts, licensing boundaries, and validation status are documented in [DATASET.md](DATASET.md). The medical dataset is not redistributed by this repository.

已报告实验使用 Kaggle 镜像中的 ISIC 2017 train/validation 分割数据。数据来源、划分数量、授权边界和评估状态详见 [DATASET.md](DATASET.md)。本仓库不再分发医疗数据。

## Kaggle Training / Kaggle 训练

Training was completed in a Kaggle GPU environment. The high-accuracy run used Tesla P100 with a PyTorch CUDA compatibility preparation step.

训练已在 Kaggle GPU 环境完成。高精度训练使用 Tesla P100，并通过兼容性脚本处理 PyTorch CUDA 运行问题。

Kaggle dependency preparation:

Kaggle 依赖准备：

```bash
pip install -r requirements-kaggle.txt
python scripts/kaggle_prepare_gpu.py --install-if-needed
```

Pre-training checks:

训练前检查：

```bash
python -m pytest -q
python scripts/check_dataset.py --config configs/kaggle_debug.yaml
python scripts/overfit_small_batch.py --config configs/kaggle_debug.yaml
python scripts/quick_train.py --config configs/kaggle_debug.yaml
```

Baseline training:

基线模型训练：

```bash
python train.py --config configs/kaggle_unet.yaml
```

High-accuracy training:

高精度模型训练：

```bash
python train.py --config configs/kaggle_high_accuracy.yaml
```

## Local Inference / 本地推理

The final default model files are:

最终默认模型文件：

```text
Checkpoint: checkpoints/best_model.pth
Config: configs/final_model.yaml
```

Model checkpoint files (`*.pth`, `*.pt`, `*.ckpt`) are intentionally excluded from Git. Download `best_model.pth` from the Kaggle output or a project release, then place it at `checkpoints/best_model.pth` before running local inference.

模型权重文件（`*.pth`、`*.pt`、`*.ckpt`）不会提交到 Git。请从 Kaggle 输出或项目 Release 下载 `best_model.pth`，并放到 `checkpoints/best_model.pth` 后再进行本地推理。

Verified Release / 已验证 Release：

```bash
mkdir -p checkpoints
curl -L \
  https://github.com/dongtingshuo/medical-image-segmentation/releases/download/v1.0.0/best_model.pth \
  -o checkpoints/best_model.pth
shasum -a 256 checkpoints/best_model.pth
```

Expected SHA256 / 期望校验值：

```text
4b04ccd5f4fbdad492a91ea9866d31b9329a886e74464ddf42fffa1854f76577
```

See [MODEL_CARD.md](MODEL_CARD.md) and [models/model_manifest.yaml](models/model_manifest.yaml) for model scope and machine-readable metadata. Checkpoints are loaded with PyTorch `weights_only=True`; do not use untrusted weight files.

模型适用范围和机器可读元数据见 [MODEL_CARD.md](MODEL_CARD.md) 与 [models/model_manifest.yaml](models/model_manifest.yaml)。checkpoint 使用 PyTorch `weights_only=True` 加载，请勿使用来源不可信的权重文件。

Single-image prediction:

单图预测：

```bash
python predict.py \
  --config configs/final_model.yaml \
  --checkpoint checkpoints/best_model.pth \
  --image path/to/image.jpg \
  --output outputs/samples \
  --threshold 0.5 \
  --device auto
```

Gradio Demo:

Gradio 演示：

```bash
python app.py
```

The project supports CPU/CUDA automatic device selection. If CUDA is unavailable, local prediction and the Gradio Demo run on CPU.

项目支持 CPU/CUDA 自动选择。若 CUDA 不可用，本地预测和 Gradio Demo 会使用 CPU。

## Evaluation / 评估

Evaluation command:

评估命令：

```bash
python evaluate.py \
  --config configs/final_model.yaml \
  --checkpoint checkpoints/best_model.pth \
  --split val \
  --threshold 0.5
```

Metrics:

指标含义：

- Dice: overlap between predicted mask and ground truth.
- IoU: intersection-over-union of predicted and ground-truth masks.
- Precision: proportion of predicted lesion pixels that are correct.
- Recall: proportion of ground-truth lesion pixels recovered by prediction.
- Specificity: proportion of background pixels correctly rejected.
- Boundary F1: boundary agreement within a configurable pixel tolerance.

`evaluate.py --split test` is supported when `test_images_dir` and `test_masks_dir` are explicitly defined in the YAML config. The repository does not claim test-set results until those paths are supplied and evaluated.

当 YAML 明确定义 `test_images_dir` 和 `test_masks_dir` 时，可使用 `evaluate.py --split test`。在未提供并评估该划分前，仓库不声明测试集结果。

## Results / 实验结果

The following results are read from:

以下结果来自：

```text
kaggle_outputs/baseline_unet/outputs/experiment_results.csv
kaggle_outputs/high_accuracy/outputs/experiment_results.csv
```

The full Kaggle outputs are kept outside Git tracking. A small set of representative curves, prediction samples, and sanity-check images is stored under `docs/assets/` for repository documentation.

完整 Kaggle 输出不纳入 Git 跟踪。仓库文档使用 `docs/assets/` 中的少量代表性曲线、预测样例和数据检查图。

| Experiment | Model | Encoder | Image Size | Batch Size | Best Epoch | Loss | Val Loss at Best Dice Epoch | Dice | IoU | Precision | Recall | Training Time | Inference Time |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| U-Net baseline | U-Net | None | 256 | 8 | 27 | BCE + Dice | 0.186221 | 0.839209 | 0.749852 | 0.904178 | 0.836919 | 11m 54s | Not available |
| High accuracy model | U-Net++ | EfficientNet-B3, ImageNet | 384 | 8 | 4 | BCE + Dice | 0.153719 | 0.872120 | 0.792033 | 0.905242 | 0.881161 | 18m 26s | Not available |

Metric improvement of the high-accuracy model over baseline:

高精度模型相对 baseline 的提升：

| Metric | Baseline | High Accuracy | Difference |
| --- | ---: | ---: | ---: |
| Dice | 0.839209 | 0.872120 | +0.032911 |
| IoU | 0.749852 | 0.792033 | +0.042181 |
| Precision | 0.904178 | 0.905242 | +0.001064 |
| Recall | 0.836919 | 0.881161 | +0.044241 |

## Visualization / 可视化结果

Training curves:

训练曲线：

```text
docs/assets/results/baseline_unet_training_curves.png
docs/assets/results/high_accuracy_training_curves.png
```

![Baseline U-Net training curves](docs/assets/results/baseline_unet_training_curves.png)

![High accuracy training curves](docs/assets/results/high_accuracy_training_curves.png)

Prediction samples:

预测样例：

```text
docs/assets/samples/baseline_unet/sample_000_overlay.png
docs/assets/samples/baseline_unet/sample_001_overlay.png
docs/assets/samples/high_accuracy/sample_000_overlay.png
docs/assets/samples/high_accuracy/sample_001_overlay.png
```

![Baseline U-Net prediction sample 0](docs/assets/samples/baseline_unet/sample_000_overlay.png)

![Baseline U-Net prediction sample 1](docs/assets/samples/baseline_unet/sample_001_overlay.png)

![High accuracy prediction sample 0](docs/assets/samples/high_accuracy/sample_000_overlay.png)

![High accuracy prediction sample 1](docs/assets/samples/high_accuracy/sample_001_overlay.png)

Sanity check report:

数据检查报告：

```text
docs/assets/sanity_check/dataset_check_report.md
docs/assets/sanity_check/dataset_overlay_00.png
docs/assets/sanity_check/dataset_overlay_01.png
```

High-accuracy dataset check summary:

高精度训练数据检查摘要：

```text
Train images/masks: 2000 / 2000
Val images/masks: 150 / 150
Invalid binary masks: 0
Image/mask size mismatches: 0
Mean foreground ratio: 0.192484
All-black/all-white masks: none reported
```

## Model Selection / 模型选择

The final default model is the high-accuracy U-Net++ model with an EfficientNet-B3 encoder. It provides higher Dice, IoU, and Recall than the U-Net baseline while keeping Precision at a similar level.

最终默认模型采用 U-Net++ + EfficientNet-B3。该模型在 Dice、IoU 和 Recall 上均优于 U-Net baseline，同时 Precision 基本保持一致。

Use:

使用：

```text
configs/final_model.yaml
checkpoints/best_model.pth
```

## Current Limitations / 当前限制

- Current reported metrics are based on the available Kaggle validation split. / 当前指标基于已有 Kaggle 验证集划分。
- The v1.0.0 checkpoint did not persist complete package versions or a source commit; the manifest records these fields as unavailable. / v1.0.0 checkpoint 未保存完整依赖版本和源码 commit，manifest 已将这些字段标记为不可用。
- No external test, cross-validation, subgroup, or clinical validation has been completed. / 尚未完成外部测试、交叉验证、子组分析或临床验证。
- Inference time was not persisted in the Kaggle output files and is therefore reported as `Not available`. / Kaggle 输出文件未持久化推理时间，因此该字段记为 `Not available`。
- The high-accuracy model shows mild overfitting after the best epoch. / 高精度模型在最佳 epoch 后出现轻微过拟合。
- Toy demo results do not represent real medical performance. / toy demo 结果不代表真实医学性能。
- Full training requires real medical datasets and GPU resources. / 完整训练需要真实医学图像数据集和 GPU 资源。
- This project is not intended for clinical diagnosis or medical decision-making. / 本项目不用于临床诊断或医学决策。

## Medical Disclaimer / 医学免责声明

English:
This project is intended only for medical image segmentation experiments and engineering workflow validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world medical decision-making.

中文：
本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。

## Future Work / 后续改进

- Add external test set evaluation and cross-validation.
- Compare additional encoders such as EfficientNet-B4, ResNet50, and ConvNeXt variants.
- Evaluate Focal + Dice loss for small-lesion recall.
- Add post-processing options for boundary smoothing or small false-positive removal.
- Export ONNX or TorchScript models for deployment.
- Add batch prediction and structured report export.

## License / 许可证

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

本项目基于 MIT License 开源，详情见 [LICENSE](LICENSE)。
