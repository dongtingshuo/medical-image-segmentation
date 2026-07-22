# Training Curve Archive / 训练曲线归档

This index covers every unique training curve recovered from the completed local and Kaggle workflows. Curves are stored by version and run instead of as a small representative subset.

本索引收录从已完成的本地与 Kaggle 流程中恢复出的全部唯一训练曲线。曲线按版本和具体 run 归档，不再只保留少量代表图。

| Stage | Curves | Gallery |
| --- | ---: | --- |
| Sanity checks / 流程检查 | 2 | [Open](training_curves/sanity.md) |
| Baseline | 1 | [Open](training_curves/baseline.md) |
| v1.0 high accuracy | 1 | [Open](training_curves/v1.0.md) |
| v1.1 repeated seeds | 3 | [Open](training_curves/v1.1.md) |
| v1.2 CV and encoder comparison | 5 | [Open](training_curves/v1.2.md) |
| v1.3 low-contrast variants | 4 | [Open](training_curves/v1.3.md) |
| v1.4 aggressive candidates | 4 | [Open](training_curves/v1.4.md) |
| v1.5 screening, teachers, students | 15 | [Open](training_curves/v1.5.md) |
| v1.6 pretraining, teachers, students | 14 | [Open](training_curves/v1.6.md) |
| **Total / 合计** | **49** | |

## Archive Policy / 归档规则

- PNG files are the original trainer outputs; they are not redrawn from rounded report values.
- Each normal training run keeps its raw `metrics.csv` or `training_history.csv` next to the curve.
- v1.5 and v1.6 also keep the locked selection and one-time final evaluation JSON files.
- Duplicate copies from release packages are represented once by their canonical run path.
- Checkpoints, source images, masks, prediction caches, and other large training artifacts are not committed.
- Sanity-check curves validate the workflow and must not be compared with formal validation or test results.

- PNG 为 trainer 直接生成的原始输出，不根据报告中的舍入数值重新绘制。
- 常规训练 run 的原始 `metrics.csv` 或 `training_history.csv` 与曲线一并保存。
- v1.5、v1.6 同时保存锁定选择和一次性最终评估 JSON。
- Release 包中的重复副本只按规范路径保留一份。
- checkpoint、原始图像、mask、概率缓存和其他大型训练产物不提交。
- sanity-check 曲线只用于验证流程，不能与正式 validation/test 结果横向比较。

Assets are rooted at [`docs/assets/experiments/`](assets/experiments/).

