# Final User Guide / 最终使用手册

This guide describes how to reproduce the training workflow, download Kaggle outputs, run local prediction, and start the Gradio Demo.

本手册说明如何复现训练流程、下载 Kaggle 输出、本地预测以及启动 Gradio Demo。

## Project Status / 项目状态

The project completed its planned engineering and Kaggle research scope in v1.6.0 and is now maintenance-only. No new training is scheduled. The verified v1.0.0 checkpoint remains the default because v1.5 and v1.6 did not pass the locked independent-test release gate.

项目已在 v1.6.0 完成计划内工程与 Kaggle 研究范围，现进入维护状态，不再安排新训练。由于 v1.5 和 v1.6 未通过锁定的独立测试发布门槛，已验证的 v1.0.0 checkpoint 继续作为默认模型。

Historical results and all 49 training curves are indexed in [`docs/TRAINING_CURVES.md`](docs/TRAINING_CURVES.md).

历史结果与全部 49 条训练曲线见 [`docs/TRAINING_CURVES.md`](docs/TRAINING_CURVES.md)。

## 1. Environment Setup / 环境配置

Use Python 3.10-3.12. `requirements.txt` pins direct dependency versions; Kaggle uses `requirements-kaggle.txt` so that its GPU-specific PyTorch build remains under explicit control.

使用 Python 3.10-3.12。`requirements.txt` 已锁定直接依赖版本；Kaggle 使用 `requirements-kaggle.txt`，避免覆盖由 GPU 兼容性脚本管理的 PyTorch build。

Create an environment:

创建环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` already includes the final U-Net++ dependency. If a minimal environment is used, install:

`requirements.txt` 已包含最终 U-Net++ 模型依赖。若使用精简环境，安装：

```bash
pip install segmentation-models-pytorch
```

Run tests:

运行测试：

```bash
python -m pytest -q
```

Device behavior:

设备行为：

- `device: auto` uses CUDA when available.
- If CUDA is unavailable, prediction and Gradio Demo use CPU.
- Kaggle formal training configs use `require_gpu: true`.

## 2. Dataset Preparation / 数据集准备

Expected local layout:

推荐本地目录：

```text
data/
  images/
    train/
    val/
  masks/
    train/
    val/
```

Images and masks must share filename stems:

图像和 mask 需要同名 stem：

```text
data/images/train/ISIC_0000001.jpg
data/masks/train/ISIC_0000001.png
```

All paths are configured through YAML files. Do not hard-code dataset paths in source code.

所有路径通过 YAML 配置传入，不在源码中硬编码数据路径。

## 3. Kaggle Training Workflow / Kaggle 训练流程

In Kaggle Notebook settings, enable GPU:

在 Kaggle Notebook 设置中开启 GPU：

```text
Settings -> Accelerator -> GPU
```

Install Kaggle-specific dependencies and verify CUDA compatibility:

安装 Kaggle 依赖并检查 CUDA 兼容性：

```bash
pip install -r requirements-kaggle.txt
python scripts/kaggle_prepare_gpu.py --install-if-needed
```

The compatibility script is required when Kaggle assigns Tesla P100 and the default PyTorch build cannot run CUDA kernels on that GPU.

当 Kaggle 分配 Tesla P100 且默认 PyTorch build 无法运行 CUDA kernel 时，该兼容性脚本是必要步骤。

## 4. Training Sanity Checks / 训练前检查

Run:

执行：

```bash
python -m pytest -q
python scripts/check_dataset.py --config configs/kaggle_debug.yaml
python scripts/overfit_small_batch.py --config configs/kaggle_debug.yaml
python scripts/quick_train.py --config configs/kaggle_debug.yaml
```

Pass criteria:

通过标准：

- Image and mask counts match.
- Filename stems match.
- Masks are binary after thresholding.
- No all-black or all-white masks are reported.
- Overlay samples align with lesion regions.
- Small-batch overfit shows decreasing loss and increasing Dice.
- Quick train finishes without NaN.

## 5. Full Training / 完整训练

Baseline U-Net:

基线 U-Net：

```bash
python train.py --config configs/kaggle_unet.yaml
```

High-accuracy model:

高精度模型：

```bash
python train.py --config configs/kaggle_high_accuracy.yaml
```

Resume interrupted training:

断点续训：

```bash
python train.py \
  --config configs/kaggle_high_accuracy.yaml \
  --resume /kaggle/working/checkpoints/last_model.pth
```

Use `last_model.pth` for resume. Use `best_model.pth` for evaluation, prediction, and Gradio Demo.

使用 `last_model.pth` 断点续训。评估、预测和 Gradio Demo 使用 `best_model.pth`。

Actual completed high-accuracy configuration:

实际完成的高精度配置：

```text
Model: U-Net++
Encoder: EfficientNet-B3
Encoder weights: ImageNet
Image size: 384
Batch size: 8
Optimizer: AdamW
Scheduler: CosineAnnealingLR
Loss: BCE + Dice
Mixed precision: enabled
Early stopping: enabled
```

## 6. Research Workflow v1.2 / 研究增强流程 v1.2

Use this workflow when you want stronger experimental evidence beyond one train/validation run. It is designed for Kaggle GPU and a small research budget. The current v1.2 run has completed successfully and is documented in `docs/releases/v1.2.0.md`.

当需要比单次 train/validation 更强的实验依据时，使用该流程。它面向 Kaggle GPU 和小预算研究训练。当前 v1.2 运行已成功完成，结果记录在 `docs/releases/v1.2.0.md`。

Run the full Kaggle research script:

运行完整 Kaggle 研究脚本：

```bash
python notebooks/kaggle_research_v1_2.py
```

The script performs:

脚本执行：

- `pytest`
- ISIC data preparation
- dataset sanity check
- 3-fold cross-validation
- EfficientNet-B3 vs ResNet34 encoder comparison
- threshold search
- subgroup analysis on internal test and external validation data
- statistical summary generation
- research artifact packaging

Main outputs:

主要输出：

```text
/kaggle/working/research_v1_2/cross_validation/cross_validation_summary.md
/kaggle/working/research_v1_2/encoder_comparison/encoder_comparison_summary.md
/kaggle/working/research_v1_2/subgroup_analysis_test/subgroup_summary.md
/kaggle/working/research_v1_2/subgroup_analysis_external/subgroup_summary.md
/kaggle/working/research_v1_2/statistics_cv_encoder/statistical_analysis.md
/kaggle/working/release_artifacts/medical-segmentation-research-artifacts-v1.2.zip
```

Completed local v1.2 output directory:

本地已下载的 v1.2 输出目录：

```text
kaggle_outputs/research_v1_2/medical-segmentation-research-artifacts-v1.2/
```

Sanitized release artifact:

清理版 Release 产物：

```text
release_artifacts/medical-segmentation-research-artifacts-v1.2.zip
SHA256: 68f8d417d8df21434666f6cfd438c0972a9849ebd6801b275cfbc4e7ab131843
```

The sanitized package excludes materialized fold data and medical images. Use this package for GitHub Release assets rather than the raw Kaggle zip.

清理版产物排除了 materialized fold data 和医学图像。用于 GitHub Release 时应使用该清理版，而不是原始 Kaggle zip。

Key v1.2 results:

v1.2 关键结果：

| Item | Result |
| --- | --- |
| 3-fold CV Dice | 0.907006 ± 0.003104 |
| 3-fold CV IoU | 0.841579 ± 0.003732 |
| Best encoder in comparison | EfficientNet-B3 |
| EfficientNet-B3 validation Dice | 0.870200 |
| ResNet34 validation Dice | 0.857985 |
| v1.2 selected threshold | 0.55 |
| Main weakness from subgroup analysis | Low-contrast images |

Do not replace the default local inference model with v1.2 outputs. The v1.2 workflow did not publish a new checkpoint, and its threshold-search result does not materially exceed the existing default model workflow. Keep using `configs/final_model.yaml` and `checkpoints/best_model.pth`.

不要用 v1.2 输出替换默认本地推理模型。v1.2 流程没有发布新 checkpoint，其阈值搜索结果也没有显著超过现有默认模型流程。继续使用 `configs/final_model.yaml` 和 `checkpoints/best_model.pth`。

## 7. Low-Contrast Workflow v1.3 / 低对比度专项流程 v1.3

Use v1.3 to test whether low-contrast augmentation and loss changes improve low-contrast segmentation.

使用 v1.3 验证低对比度增强和 loss 改动是否能改善低对比度图像分割。

Debug run:

调试运行：

```bash
python notebooks/kaggle_low_contrast_v1_3.py --debug
```

Full run:

完整运行：

```bash
python notebooks/kaggle_low_contrast_v1_3.py
```

Restart after Kaggle time limit:

Kaggle 时间到后继续：

```bash
python notebooks/kaggle_low_contrast_v1_3.py
```

Re-submit the same script. The workflow skips variants with `completed.json` and resumes unfinished variants from their local `checkpoints/last_model.pth`.

重新提交同一个脚本即可。流程会跳过已有 `completed.json` 的 variant，并从未完成 variant 的 `checkpoints/last_model.pth` 继续。

Variants:

实验变体：

```text
control_bce_dice
contrast_aug_bce_dice
contrast_aug_focal_dice
contrast_aug_tversky
```

Main outputs:

主要输出：

```text
/kaggle/working/research_v1_3_low_contrast/comparison/low_contrast_comparison.csv
/kaggle/working/research_v1_3_low_contrast/comparison/low_contrast_comparison.md
/kaggle/working/research_v1_3_low_contrast/execution_manifest.json
/kaggle/working/release_artifacts/medical-segmentation-low-contrast-artifacts-v1.3.zip
```

Completed local download:

已完成本地下载：

```text
kaggle_outputs/low_contrast_v1_3/research_v1_3_low_contrast/comparison/low_contrast_comparison.csv
kaggle_outputs/low_contrast_v1_3/research_v1_3_low_contrast/comparison/low_contrast_comparison.md
kaggle_outputs/low_contrast_v1_3/research_v1_3_low_contrast/execution_manifest.json
kaggle_outputs/low_contrast_v1_3/release_artifacts/medical-segmentation-low-contrast-artifacts-v1.3.zip
```

Completed v1.3 decision:

已完成 v1.3 决策：

| Item | Result |
| --- | --- |
| Best variant | `contrast_aug_bce_dice` |
| Internal test Dice | `0.864766` |
| Internal test IoU | `0.786815` |
| Internal low-contrast Dice delta | `+0.000800` |
| Internal low-contrast Recall delta | `+0.009289` |
| External Dice | `0.924386` |
| External low-contrast Recall delta | `+0.023732` |
| Replace default model | `No` |

Do not replace the default model. The completed comparison report did not meet the configured replacement criterion of at least `+0.02` internal low-contrast Dice improvement or `+0.03` internal low-contrast Recall improvement.

不要替换默认模型。已完成对比报告没有达到预设替换标准，即 internal 低对比度 Dice 至少提升 `+0.02` 或 internal 低对比度 Recall 至少提升 `+0.03`。

## 8. Downloading Kaggle Outputs / 下载 Kaggle 输出

Required files:

需要下载的文件：

```text
checkpoints/best_model.pth
checkpoints/last_model.pth
outputs/metrics.csv
outputs/experiment_results.csv
outputs/curves/training_curves.png
outputs/samples/
outputs/sanity_check/
```

Current downloaded output directories:

当前已下载输出目录：

```text
kaggle_outputs/baseline_unet/
kaggle_outputs/high_accuracy/
kaggle_outputs/research_v1_2/
```

These full output directories are local runtime artifacts and are intentionally ignored by Git. Representative public documentation images are stored in:

这些完整输出目录属于本地运行产物，已被 Git 忽略。用于公开文档展示的代表性图片位于：

```text
docs/assets/results/
docs/assets/samples/
docs/assets/sanity_check/
```

Final model checkpoint copied to:

最终模型 checkpoint 已复制到：

```text
checkpoints/best_model.pth
```

Checkpoint files are ignored by Git. For a fresh clone, download `best_model.pth` from the Kaggle output or project release and place it in the path above.

checkpoint 文件已被 Git 忽略。全新 clone 后，需要从 Kaggle 输出或项目 Release 下载 `best_model.pth`，并放到上述路径。

Direct verified download / 直接验证下载：

```bash
mkdir -p checkpoints
curl -L \
  https://github.com/dongtingshuo/medical-image-segmentation/releases/download/v1.0.0/best_model.pth \
  -o checkpoints/best_model.pth
shasum -a 256 checkpoints/best_model.pth
```

Expected SHA256 / 期望 SHA256：

```text
4b04ccd5f4fbdad492a91ea9866d31b9329a886e74464ddf42fffa1854f76577
```

Windows PowerShell verification / Windows PowerShell 校验：

```powershell
Get-FileHash checkpoints/best_model.pth -Algorithm SHA256
```

See `MODEL_CARD.md` and `models/model_manifest.yaml` before using the artifact.

使用权重前请阅读 `MODEL_CARD.md` 和 `models/model_manifest.yaml`。

Final model config:

最终模型配置：

```text
configs/final_model.yaml
```

## 9. Local Prediction / 本地预测

Use the final model:

使用最终模型：

```bash
python predict.py \
  --config configs/final_model.yaml \
  --checkpoint checkpoints/best_model.pth \
  --image path/to/image.jpg \
  --output outputs/samples \
  --device auto
```

Outputs:

输出：

```text
*_image.png
*_pred_mask.png
*_overlay.png
*_lesion_ratio.txt
```

## 10. Batch Prediction / 批量预测

Run inference on a folder:

对一个图片目录执行推理：

```bash
python batch_predict.py \
  --config configs/final_model.yaml \
  --checkpoint checkpoints/best_model.pth \
  --input-dir path/to/images \
  --output outputs/batch_predictions \
  --device auto
```

Use `--recursive` when images are stored in nested folders. Supported image extensions are `jpg`, `jpeg`, and `png` by default.

如果图片在多层目录中，加入 `--recursive`。默认支持 `jpg`、`jpeg` 和 `png`。

Outputs:

输出：

```text
outputs/batch_predictions/*_image.png
outputs/batch_predictions/*_pred_mask.png
outputs/batch_predictions/*_overlay.png
outputs/batch_predictions/*_lesion_ratio.txt
outputs/batch_predictions/batch_predictions.csv
outputs/batch_predictions/batch_summary.json
```

The CSV records image path, status, lesion area ratio, inference time, device, model name, checkpoint epoch, and output paths.

CSV 会记录图片路径、状态、病灶面积比例、推理时间、设备、模型名称、checkpoint epoch 和输出路径。

## 11. Model Export / 模型导出

Export the final model to TorchScript and ONNX:

导出最终模型为 TorchScript 和 ONNX：

```bash
python export.py \
  --config configs/final_model.yaml \
  --checkpoint checkpoints/best_model.pth \
  --output-dir exports/final_model \
  --formats torchscript,onnx \
  --device cpu
```

Outputs:

输出：

```text
exports/final_model/model_torchscript.pt
exports/final_model/model.onnx
exports/final_model/export_manifest.json
```

The exported model returns one-channel logits. In deployment code, apply sigmoid and threshold after inference.

导出的模型返回单通道 logits。部署代码中需要在推理后执行 sigmoid 和 threshold。

## 12. Running Gradio Demo / 运行 Gradio Demo

Start:

启动：

```bash
python app.py
```

Recommended fields:

推荐填写：

```text
Config: configs/final_model.yaml
Checkpoint: checkpoints/best_model.pth
Model type: Auto (checkpoint/config)
Threshold: 0.35
Device: auto
```

The Demo loads the model once and caches it by checkpoint path, modification time, device, and config. It automatically reads the architecture from project checkpoints and rejects incompatible manual model selections.

Demo 会按 checkpoint 路径、修改时间、device 和 config 缓存模型。它会自动读取本项目 checkpoint 中的架构，并拒绝与权重不兼容的手动模型选择。

The Demo shows original image, predicted mask, overlay, lesion area ratio, inference time, active device, model name, and checkpoint epoch.

Demo 会显示原图、预测 mask、叠加图、病灶面积比例、推理时间和当前设备。

## 13. Docker Demo / Docker 演示

Build the image:

构建镜像：

```bash
docker build -t medical-image-segmentation .
```

Run the CPU Gradio demo:

运行 CPU Gradio Demo：

```bash
docker run --rm -p 7860:7860 \
  -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/outputs:/app/outputs" \
  medical-image-segmentation
```

Open:

打开：

```text
http://localhost:7860
```

Place `best_model.pth` in `checkpoints/` before starting the container. The default container is intended for CPU inference and demo use, not Kaggle training.

启动容器前请先将 `best_model.pth` 放入 `checkpoints/`。默认容器面向 CPU 推理和 Demo，不用于 Kaggle 训练。

## 14. Evaluation / 评估

Evaluate the final model:

评估最终模型：

```bash
python evaluate.py \
  --config configs/final_model.yaml \
  --checkpoint checkpoints/best_model.pth \
  --split val \
  --threshold 0.35
```

Metrics:

指标：

- Dice
- IoU
- Precision
- Recall
- Sensitivity, represented by Recall for the lesion class
- Specificity
- Boundary F1
- Mean validation loss

To evaluate an independent test split, add `test_images_dir` and `test_masks_dir` under `data` in a YAML config, then run with `--split test`. The completed repeated workflow already reports ISIC 2017 independent test metrics and ISIC 2018 external validation metrics.

如需评估独立测试集，请先在 YAML 的 `data` 中增加 `test_images_dir` 和 `test_masks_dir`，再使用 `--split test`。已完成的重复实验流程已经报告 ISIC 2017 独立测试集和 ISIC 2018 外部验证集指标。

## 15. Completed Results / 已完成结果

| Experiment | Model | Val Loss at Best Dice Epoch | Dice | IoU | Precision | Recall | Training Time | Inference Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| U-Net baseline | U-Net | 0.186221 | 0.839209 | 0.749852 | 0.904178 | 0.836919 | 11m 54s | Not available |
| High accuracy | U-Net++ EfficientNet-B3 | 0.153719 | 0.872120 | 0.792033 | 0.905242 | 0.881161 | 18m 26s | Not available |

The default model is the high-accuracy model because it has higher Dice, IoU, and Recall.

默认模型为高精度模型，因为其 Dice、IoU 和 Recall 更高。

Repeated high-accuracy evaluation:

高精度模型重复实验：

| Split | Dice mean ± std | IoU mean ± std | Precision mean ± std | Recall mean ± std |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.870568 ± 0.004248 | 0.791262 ± 0.004706 | 0.918614 ± 0.023204 | 0.866186 ± 0.026165 |
| Independent test | 0.852301 ± 0.009611 | 0.769329 ± 0.012870 | 0.947166 ± 0.010456 | 0.815953 ± 0.022209 |
| External ISIC 2018 | 0.915828 ± 0.006676 | 0.857054 ± 0.011829 | 0.956375 ± 0.014224 | 0.895332 ± 0.025478 |

Research workflow v1.2:

研究增强流程 v1.2：

| Experiment | Dice | IoU | Precision | Recall | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 3-fold CV mean | 0.907006 | 0.841579 | 0.927466 | 0.909939 | Mean over 3 folds |
| EfficientNet-B3 encoder comparison | 0.870200 | 0.790512 | 0.885267 | 0.896205 | Best encoder by Dice |
| ResNet34 encoder comparison | 0.857985 | 0.775294 | 0.913277 | 0.856506 | Higher precision, lower recall |

v1.2 did not change the default inference model.

v1.2 未改变默认推理模型。

Inference benchmark for the best repeated checkpoint:

最佳重复实验 checkpoint 推理基准：

| Device | Mean latency | P95 latency | Throughput | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| CPU x86_64 | 497.374 ms | 524.090 ms | 2.011 img/s | 1313.23 MB RSS |
| CUDA Tesla P100-PCIE-16GB | 23.867 ms | 25.603 ms | 41.898 img/s | 151.85 MB allocated |

Model parameters: `13,624,793`; model state size: `52.32 MB`; checkpoint size: `152.32 MB`.

模型参数量：`13,624,793`；模型 state 大小：`52.32 MB`；checkpoint 大小：`152.32 MB`。

Threshold search selected `0.35` as the recommended default threshold:

阈值搜索选择 `0.35` 作为推荐默认阈值：

| Threshold | Validation Dice | Validation IoU | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: |
| 0.35 | 0.876188 | 0.779657 | 0.895335 | 0.857843 |
| 0.50 | 0.872413 | 0.773700 | 0.917849 | 0.831264 |

Failure case analysis at threshold `0.35`:

阈值 `0.35` 下的失败案例分析：

| Split | Samples | Mean Dice | Mean IoU | Over-segmentation | Under-segmentation | Empty prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ISIC 2017 test | 600 | 0.858912 | 0.778042 | 111 | 174 | 0 |
| External ISIC 2018 | 1002 | 0.924017 | 0.870954 | 93 | 104 | 0 |

## 16. Output Paths / 输出路径

Training curves:

训练曲线：

```text
docs/assets/results/baseline_unet_training_curves.png
docs/assets/results/high_accuracy_training_curves.png
docs/assets/results/repeated_experiment/seed_42_training_curves.png
docs/assets/analysis/threshold_search/threshold_search.md
docs/assets/results/research_v1_2/cv_fold_1_training_curves.png
docs/assets/results/research_v1_2/encoder_effb3_training_curves.png
```

Prediction samples:

预测样例：

```text
docs/assets/samples/baseline_unet/
docs/assets/samples/high_accuracy/
docs/assets/samples/repeated_experiment/
docs/assets/samples/research_v1_2/
docs/assets/analysis/failure_cases_test/
docs/assets/analysis/failure_cases_external/
```

Sanity check:

数据检查：

```text
docs/assets/sanity_check/dataset_check_report.md
docs/assets/sanity_check/dataset_overlay_00.png
docs/assets/sanity_check/dataset_overlay_01.png
docs/assets/sanity_check/repeated_experiment/dataset_overlay_00_isic_0012940.png
docs/assets/sanity_check/research_v1_2/dataset_overlay_00_isic_0012940.png
```

## 17. Troubleshooting / 常见问题

### CUDA is unavailable / CUDA 不可用

Local prediction can run on CPU. Kaggle formal training should use GPU.

本地预测可使用 CPU。Kaggle 正式训练应使用 GPU。

### Tesla P100 kernel error / Tesla P100 CUDA kernel 错误

Run:

执行：

```bash
python scripts/kaggle_prepare_gpu.py --install-if-needed
```

### Checkpoint cannot be loaded / checkpoint 无法加载

Check that the config matches the checkpoint. The final checkpoint requires:

确认 config 与 checkpoint 对应。最终 checkpoint 需要：

```text
model_name: unet_plus_plus
encoder_name: efficientnet-b3
image_size: 384
```

The loader uses `weights_only=True`. A checkpoint rejected by safe loading should not be forced open; download the verified Release artifact and compare its SHA256 instead.

加载器使用 `weights_only=True`。若 checkpoint 被安全加载机制拒绝，不应强制打开；请重新下载经验证的 Release 权重并比对 SHA256。

### Missing segmentation_models_pytorch / 缺少 segmentation_models_pytorch

Install:

安装：

```bash
pip install segmentation-models-pytorch
```

### Prediction mask is all black / 预测 mask 全黑

Try lowering the threshold to `0.3`, verify checkpoint/config matching, and confirm preprocessing uses the same image size.

可尝试将 threshold 降到 `0.3`，检查 checkpoint/config 是否匹配，并确认预处理尺寸一致。

### ONNX export fails / ONNX 导出失败

Install the pinned dependencies first:

先安装锁定依赖：

```bash
pip install -r requirements.txt
```

If the error mentions `onnx`, install it explicitly:

如果错误提示 `onnx`，可单独安装：

```bash
pip install onnx
```

### Docker cannot find checkpoint / Docker 找不到 checkpoint

Mount the local `checkpoints/` directory into the container:

将本地 `checkpoints/` 目录挂载到容器：

```bash
docker run --rm -p 7860:7860 -v "$PWD/checkpoints:/app/checkpoints" medical-image-segmentation
```

### Validation metrics fluctuate / 验证指标波动

This can occur after the best epoch. Use the saved `best_model.pth`, not `last_model.pth`.

最佳 epoch 后验证指标波动是常见现象。应使用 `best_model.pth`，而不是 `last_model.pth`。

## Medical Disclaimer / 医学免责声明

English:
This project is intended only for medical image segmentation experiments and engineering workflow validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world medical decision-making.

中文：
本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。
