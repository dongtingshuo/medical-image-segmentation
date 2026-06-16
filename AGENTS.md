# AGENTS.md

## 项目目标

本项目 `medical-image-segmentation` 用于实现“基于 U-Net 改进模型的皮肤病灶图像分割系统”。目标是完成一个可展示的本科深度学习项目，支持皮肤病灶图像像素级二分类分割、训练评估、预测可视化、Gradio Web Demo 和 Kaggle 云训练。

## 技术栈

- Python 3.10+
- PyTorch
- OpenCV
- Albumentations
- NumPy
- Matplotlib
- scikit-learn
- Gradio
- PyYAML
- pytest
- segmentation-models-pytorch，可选高性能模型依赖

## 开发规则

- 所有数据路径必须来自 YAML 配置或命令行参数。
- 不允许在源码中硬编码本地绝对路径。
- 不允许在源码中硬编码 `/kaggle/input` 或 `/kaggle/working`，Kaggle 路径只能出现在 Kaggle 配置模板中。
- 模型输出统一为 1 通道 logits，模型内部不做 sigmoid。
- 推理、评估和指标计算阶段再做 sigmoid + threshold。
- 本地代码必须支持 CPU 和 CUDA 自动选择。
- CUDA 可用时可启用 mixed precision；CPU 下自动关闭。
- 不允许提交真实数据集、大模型 checkpoint、训练中间大文件。
- 每次修改 `model_*`、`losses.py`、`metrics.py`、`trainer.py` 后必须运行 `pytest tests`。

## 常用命令

```bash
pip install -r requirements.txt
pytest tests
python scripts/check_dataset.py --config configs/debug_local.yaml
python scripts/overfit_small_batch.py --config configs/debug_local.yaml
python scripts/quick_train.py --config configs/debug_local.yaml
python train.py --config configs/unet.yaml
python evaluate.py --config configs/unet.yaml --checkpoint checkpoints/best_model.pth
python predict.py --config configs/unet.yaml --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
python app.py
```

## 目录说明

- `configs/`：本地、Kaggle、对比实验配置。
- `src/`：数据集、模型、loss、metrics、训练、可视化和工具代码。
- `scripts/`：正式长时间训练前的检查脚本。
- `notebooks/`：Kaggle 云训练 Notebook。
- `outputs/`：指标、曲线、样例预测和实验记录。
- `checkpoints/`：训练保存的模型权重。
- `docs/`：实验报告、实验计划和评估模板。
- `tests/`：不依赖真实数据集的单元测试。

## Kaggle 训练、本地推理原则

- 正式训练主要在 Kaggle GPU 上完成。
- 本地工程必须可运行，主要用于预测、评估、Gradio Demo 和小规模训练测试。
- Kaggle 配置使用 `/kaggle/input/...` 和 `/kaggle/working/...` 作为可编辑模板路径。
- 本地配置使用相对路径示例，用户按数据集位置修改。

## 禁止事项

- 禁止提交 `data/`、`datasets/`、大 checkpoint、Kaggle 输入数据。
- 禁止把个人机器绝对路径写入源码或默认配置。
- 禁止跳过训练前检查直接长时间跑 Kaggle。
