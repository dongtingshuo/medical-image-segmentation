# AGENTS.md

## Project Goal / 项目目标

English:
`medical-image-segmentation` is a PyTorch-based binary medical image segmentation project for skin lesion segmentation experiments. It supports training, evaluation, visualization, Gradio demo, Kaggle GPU workflows, toy demos, model comparison, and error analysis.

中文：
`medical-image-segmentation` 是一个基于 PyTorch 的医学图像二分类分割项目，当前面向皮肤病灶分割实验。项目支持训练、评估、可视化、Gradio demo、Kaggle GPU 流程、toy demo、模型对比和错误分析。

## Technology Stack / 技术栈

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
- segmentation-models-pytorch as an optional high-capacity model dependency / `segmentation-models-pytorch` 作为可选高容量模型依赖

## Development Rules / 开发规则

English:
All data paths must come from YAML configs or command-line arguments. Source code must not hard-code local absolute paths or Kaggle paths. Models must output one-channel logits, and sigmoid/thresholding should happen only during metrics or inference.

中文：
所有数据路径必须来自 YAML 配置或命令行参数。源码不得硬编码本地绝对路径或 Kaggle 路径。模型统一输出 1 通道 logits，sigmoid 和 threshold 只在指标计算或推理阶段执行。

- Local code must support CPU/CUDA automatic selection. / 本地代码必须支持 CPU/CUDA 自动选择。
- Mixed precision is enabled only when CUDA is usable. / mixed precision 仅在 CUDA 可用时启用。
- Deterministic algorithms and seeded DataLoader workers are the default; performance-oriented exceptions must be explicit in YAML. / 默认使用确定性算法和 DataLoader worker 随机种子；吞吐量优先的例外必须在 YAML 中明确设置。
- Load checkpoints with `weights_only=True` and validate embedded architecture metadata. / checkpoint 必须使用 `weights_only=True` 加载，并验证其内嵌架构元数据。
- Do not commit real datasets, large checkpoints, or training artifacts. / 不提交真实数据集、大型 checkpoint 或训练产物。
- After changing models, losses, metrics, trainer, or analysis logic, run tests. / 修改模型、loss、metrics、trainer 或分析逻辑后需要运行测试。

## Common Commands / 常用命令

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/create_toy_segmentation_data.py
python scripts/run_segmentation_comparison.py --config configs/demo_comparison.yaml
python scripts/run_visualization_demo.py
python scripts/run_error_analysis.py
python scripts/create_cv_folds.py --images-dir data/images/train --masks-dir data/masks/train --output-root outputs/cross_validation --k 3
python scripts/run_cross_validation.py --config configs/kaggle_research_v1_2.yaml --images-dir /path/to/images --masks-dir /path/to/masks
python scripts/run_encoder_comparison.py --config configs/kaggle_research_v1_2.yaml --encoders efficientnet-b3 resnet34
python scripts/analyze_subgroups.py --config configs/final_model.yaml --checkpoint checkpoints/best_model.pth --split test --threshold 0.35
python scripts/analyze_statistics.py --inputs /path/to/metrics.csv --output-dir outputs/statistical_analysis
python scripts/check_dataset.py --config configs/debug_local.yaml
python train.py --config configs/unet.yaml
python evaluate.py --config configs/final_model.yaml --checkpoint checkpoints/best_model.pth --split val
python predict.py --config configs/final_model.yaml --checkpoint checkpoints/best_model.pth --image path/to/image.jpg
python app.py
```

## Directory Guide / 目录说明

- `configs/`: local, Kaggle, and comparison configs / 本地、Kaggle 和对比实验配置。
- `src/`: dataset, models, losses, metrics, training, visualization, and analysis code / 数据集、模型、loss、metrics、训练、可视化和分析代码。
- `scripts/`: sanity checks, toy demo, comparison, visualization, and error analysis scripts / 训练检查、toy demo、对比、可视化和错误分析脚本。
- `examples/`: lightweight synthetic demo data / 轻量合成 demo 数据。
- `notebooks/`: Kaggle training notebook / Kaggle 训练 Notebook。
- `outputs/`: generated local outputs ignored by Git / 本地生成输出，默认被 Git 忽略。
- `checkpoints/`: local model weights ignored by Git / 本地模型权重，默认被 Git 忽略。
- `docs/`: reports, plans, and templates / 报告、计划和模板。
- `models/`: release manifest and verified artifact metadata / Release manifest 和已验证权重元数据。
- `.github/workflows/`: CPU continuous-integration checks / CPU 持续集成检查。
- `tests/`: unit and workflow tests independent of real medical datasets / 不依赖真实医学数据集的单元测试和流程测试。

## Kaggle Training and Local Inference / Kaggle 训练与本地推理

English:
Formal training is designed for Kaggle GPU. Local usage focuses on prediction, evaluation, Gradio demo, toy demos, and small-scale smoke testing.

中文：
正式训练主要面向 Kaggle GPU。本地使用主要用于预测、评估、Gradio demo、toy demo 和小规模流程测试。

## Medical Disclaimer / 医学免责声明

English:
This project is intended only for medical image segmentation experiments and engineering workflow validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world medical decision-making.

中文：
本项目仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。
