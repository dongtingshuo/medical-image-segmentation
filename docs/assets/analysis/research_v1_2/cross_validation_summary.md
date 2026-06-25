# Cross-Validation Summary

This report summarizes validation metrics across materialized folds. Standard deviation uses `ddof=1` when at least two folds are available.

Best fold by validation Dice: `1`

| Split | Metric | Folds | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| val | loss | 3 | 0.108300 | 0.006976 | 0.103654 | 0.116322 |
| val | dice | 3 | 0.907006 | 0.003104 | 0.903474 | 0.909298 |
| val | iou | 3 | 0.841579 | 0.003732 | 0.837271 | 0.843789 |
| val | precision | 3 | 0.927466 | 0.009870 | 0.917520 | 0.937258 |
| val | recall | 3 | 0.909939 | 0.006971 | 0.902043 | 0.915241 |
| val | specificity | 3 | 0.982133 | 0.001544 | 0.980465 | 0.983513 |
| val | boundary_f1 | 3 | 0.538169 | 0.011199 | 0.531467 | 0.551098 |

## Per-Fold Validation Results

| Fold | Dice | IoU | Precision | Recall | Loss | Checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.903474 | 0.837271 | 0.917520 | 0.915241 | 0.104925 | `/kaggle/working/research_v1_2/cross_validation/fold_0/checkpoints/best_model.pth` |
| 1 | 0.909298 | 0.843677 | 0.937258 | 0.902043 | 0.116322 | `/kaggle/working/research_v1_2/cross_validation/fold_1/checkpoints/best_model.pth` |
| 2 | 0.908246 | 0.843789 | 0.927618 | 0.912533 | 0.103654 | `/kaggle/working/research_v1_2/cross_validation/fold_2/checkpoints/best_model.pth` |
