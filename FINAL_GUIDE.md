# Final Guide

## 推荐完成路线

1. 用 `configs/debug_local.yaml` 检查代码和数据读取。
2. 在 Kaggle 上运行 `configs/kaggle_debug.yaml`，确认数据路径、显存、loss 和指标正常。
3. 运行手写 U-Net 和 Attention U-Net，完成原理展示对比。
4. 运行 `configs/kaggle_high_accuracy.yaml`，追求更高 Dice 和 IoU。
5. 将 `outputs/experiment_results.csv`、训练曲线和预测样例整理进实验报告。

## 正式训练前检查

必须先运行：

```bash
python scripts/check_dataset.py --config configs/kaggle_debug.yaml
python scripts/overfit_small_batch.py --config configs/kaggle_debug.yaml
python scripts/quick_train.py --config configs/kaggle_debug.yaml
```

确认无误后再启动长时间训练。

## 展示重点

- 手写 U-Net 展示基本结构。
- Attention U-Net 展示注意力机制。
- U-Net++ / DeepLabV3+ 展示预训练 encoder 对精度的提升。
- 用 Dice、IoU、Precision、Recall 和可视化样例说明效果。
