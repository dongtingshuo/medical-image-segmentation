# Final User Guide / 最终使用手册

This guide describes how to reproduce the training workflow, download Kaggle outputs, run local prediction, and start the Gradio Demo.

本手册说明如何复现训练流程、下载 Kaggle 输出、本地预测以及启动 Gradio Demo。

## 1. Environment Setup / 环境配置

Create an environment:

创建环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
pytest tests
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
pytest tests
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

## 6. Downloading Kaggle Outputs / 下载 Kaggle 输出

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

Final model config:

最终模型配置：

```text
configs/final_model.yaml
```

## 7. Local Prediction / 本地预测

Use the final model:

使用最终模型：

```bash
python predict.py \
  --config configs/final_model.yaml \
  --checkpoint checkpoints/best_model.pth \
  --image path/to/image.jpg \
  --output outputs/samples \
  --threshold 0.5 \
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

## 8. Running Gradio Demo / 运行 Gradio Demo

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
Model type: U-Net++
Threshold: 0.5
Device: auto
```

The Demo shows original image, predicted mask, overlay, lesion area ratio, inference time, and active device.

Demo 会显示原图、预测 mask、叠加图、病灶面积比例、推理时间和当前设备。

## 9. Evaluation / 评估

Evaluate the final model:

评估最终模型：

```bash
python evaluate.py --config configs/final_model.yaml --checkpoint checkpoints/best_model.pth
```

Metrics:

指标：

- Dice
- IoU
- Precision
- Recall
- Mean validation loss

## 10. Completed Results / 已完成结果

| Experiment | Model | Best Val Loss | Dice | IoU | Precision | Recall | Training Time | Inference Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| U-Net baseline | U-Net | 0.186221 | 0.839209 | 0.749852 | 0.904178 | 0.836919 | 11m 54s | Not available |
| High accuracy | U-Net++ EfficientNet-B3 | 0.153719 | 0.872120 | 0.792033 | 0.905242 | 0.881161 | 18m 26s | Not available |

The default model is the high-accuracy model because it has higher Dice, IoU, and Recall.

默认模型为高精度模型，因为其 Dice、IoU 和 Recall 更高。

## 11. Output Paths / 输出路径

Training curves:

训练曲线：

```text
docs/assets/results/baseline_unet_training_curves.png
docs/assets/results/high_accuracy_training_curves.png
```

Prediction samples:

预测样例：

```text
docs/assets/samples/baseline_unet/
docs/assets/samples/high_accuracy/
```

Sanity check:

数据检查：

```text
docs/assets/sanity_check/dataset_check_report.md
docs/assets/sanity_check/dataset_overlay_00.png
docs/assets/sanity_check/dataset_overlay_01.png
```

## 12. Troubleshooting / 常见问题

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

### Missing segmentation_models_pytorch / 缺少 segmentation_models_pytorch

Install:

安装：

```bash
pip install segmentation-models-pytorch
```

### Prediction mask is all black / 预测 mask 全黑

Try lowering the threshold to `0.3`, verify checkpoint/config matching, and confirm preprocessing uses the same image size.

可尝试将 threshold 降到 `0.3`，检查 checkpoint/config 是否匹配，并确认预处理尺寸一致。

### Validation metrics fluctuate / 验证指标波动

This can occur after the best epoch. Use the saved `best_model.pth`, not `last_model.pth`.

最佳 epoch 后验证指标波动是常见现象。应使用 `best_model.pth`，而不是 `last_model.pth`。
