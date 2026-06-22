# Contributing / 贡献指南

## Development Setup / 开发环境

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
pre-commit install
```

## Change Rules / 修改规则

- Keep dataset and output paths in YAML or command-line arguments.
- Models must return one-channel logits; apply sigmoid only in metrics or inference.
- Do not commit datasets, credentials, checkpoints, or generated training outputs.
- Preserve CPU execution for tests, evaluation, prediction, and the Gradio demo.
- Update bilingual documentation when behavior or commands change.

- 数据和输出路径必须通过 YAML 或命令行参数传入。
- 模型必须输出单通道 logits，sigmoid 只在指标或推理阶段使用。
- 不得提交数据集、凭据、checkpoint 或训练输出。
- 测试、评估、预测和 Gradio Demo 必须保留 CPU 运行能力。
- 行为或命令变更时同步更新中英文文档。

## Validation / 验证

```bash
python -m pytest -q
ruff check .
python train.py --help
python evaluate.py --help
python predict.py --help
```

Changes to models, losses, metrics, training, checkpoint loading, or inference must include focused tests.

修改模型、损失函数、指标、训练、checkpoint 加载或推理逻辑时，必须增加对应测试。
