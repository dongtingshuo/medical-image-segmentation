# Failure Case Analysis

- Split: `test`
- Threshold: `0.350`
- Samples: `600`

## Mean Metrics

| Metric | Value |
| --- | ---: |
| Dice | 0.858912 |
| IoU | 0.778042 |
| Precision | 0.907465 |
| Recall | 0.863243 |
| Specificity | 0.972649 |

## Error Counts

| Error Type | Count |
| --- | ---: |
| over_segmentation | 111 |
| under_segmentation | 174 |
| small_object_miss | 0 |
| boundary_error | 600 |
| empty_prediction | 0 |
| empty_ground_truth | 0 |

## Worst Cases

| Rank | Index | Image | Dice | IoU | Precision | Recall | Flags |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | 57 | isic_0012837.jpg | 0.041577 | 0.021230 | 0.986481 | 0.021236 | under_segmentation, boundary_error |
| 1 | 400 | isic_0015251.jpg | 0.050887 | 0.026108 | 0.026108 | 1.000000 | over_segmentation, boundary_error |
| 2 | 183 | isic_0014457.jpg | 0.053224 | 0.027340 | 1.000000 | 0.027340 | under_segmentation, boundary_error |
| 3 | 463 | isic_0015607.jpg | 0.119572 | 0.063588 | 1.000000 | 0.063588 | under_segmentation, boundary_error |
| 4 | 187 | isic_0014489.jpg | 0.119679 | 0.063648 | 1.000000 | 0.063648 | under_segmentation, boundary_error |
| 5 | 94 | isic_0013321.jpg | 0.172462 | 0.094369 | 1.000000 | 0.094369 | under_segmentation, boundary_error |
| 6 | 111 | isic_0013565.jpg | 0.178519 | 0.098008 | 0.098008 | 1.000000 | over_segmentation, boundary_error |
| 7 | 421 | isic_0015353.jpg | 0.192915 | 0.106755 | 0.117455 | 0.539551 | over_segmentation, boundary_error |
| 8 | 564 | isic_0016034.jpg | 0.194519 | 0.107738 | 0.107738 | 1.000000 | over_segmentation, boundary_error |
| 9 | 144 | isic_0013966.jpg | 0.198206 | 0.110005 | 0.110005 | 1.000000 | over_segmentation, boundary_error |
| 10 | 38 | isic_0012447.jpg | 0.249546 | 0.142561 | 1.000000 | 0.142561 | under_segmentation, boundary_error |
| 11 | 141 | isic_0013925.jpg | 0.288460 | 0.168538 | 1.000000 | 0.168538 | under_segmentation, boundary_error |
| 12 | 586 | isic_0016058.jpg | 0.288832 | 0.168792 | 0.168792 | 1.000000 | over_segmentation, boundary_error |
| 13 | 39 | isic_0012448.jpg | 0.301720 | 0.177662 | 0.177662 | 1.000000 | over_segmentation, boundary_error |
| 14 | 119 | isic_0013673.jpg | 0.302216 | 0.178006 | 0.178006 | 1.000000 | over_segmentation, boundary_error |
| 15 | 91 | isic_0013281.jpg | 0.304786 | 0.179792 | 1.000000 | 0.179792 | under_segmentation, boundary_error |
