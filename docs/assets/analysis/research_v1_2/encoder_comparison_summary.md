# Encoder Comparison Summary

This report compares high-capacity segmentation backbones under the same dataset split and training recipe.

Best encoder by validation Dice: `efficientnet-b3`

| Encoder | Dice | IoU | Precision | Recall | Loss | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| efficientnet-b3 | 0.870200 | 0.790512 | 0.885267 | 0.896205 | 0.139043 | `/kaggle/working/research_v1_2/encoder_comparison/efficientnet-b3/checkpoints/best_model.pth` |
| resnet34 | 0.857985 | 0.775294 | 0.913277 | 0.856506 | 0.163083 | `/kaggle/working/research_v1_2/encoder_comparison/resnet34/checkpoints/best_model.pth` |

## Metric Summary

| Split | Metric | Runs | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| val | loss | 2 | 0.151063 | 0.016999 | 0.139043 | 0.163083 |
| val | dice | 2 | 0.864093 | 0.008638 | 0.857985 | 0.870200 |
| val | iou | 2 | 0.782903 | 0.010761 | 0.775294 | 0.790512 |
| val | precision | 2 | 0.899272 | 0.019807 | 0.885267 | 0.913277 |
| val | recall | 2 | 0.876356 | 0.028071 | 0.856506 | 0.896205 |
| val | specificity | 2 | 0.972952 | 0.007022 | 0.967987 | 0.977918 |
| val | boundary_f1 | 2 | 0.489616 | 0.015602 | 0.478584 | 0.500648 |
