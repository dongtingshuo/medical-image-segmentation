# Failure Case Analysis

- Split: `external`
- Threshold: `0.350`
- Samples: `1002`

## Mean Metrics

| Metric | Value |
| --- | ---: |
| Dice | 0.924017 |
| IoU | 0.870954 |
| Precision | 0.938930 |
| Recall | 0.924878 |
| Specificity | 0.975499 |

## Error Counts

| Error Type | Count |
| --- | ---: |
| over_segmentation | 93 |
| under_segmentation | 104 |
| small_object_miss | 0 |
| boundary_error | 997 |
| empty_prediction | 0 |
| empty_ground_truth | 0 |

## Worst Cases

| Rank | Index | Image | Dice | IoU | Precision | Recall | Flags |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | 532 | isic_0029519.jpg | 0.000043 | 0.000022 | 0.000184 | 0.000024 | under_segmentation, boundary_error |
| 1 | 200 | isic_0026295.jpg | 0.198821 | 0.110384 | 1.000000 | 0.110384 | under_segmentation, boundary_error |
| 2 | 249 | isic_0026870.jpg | 0.210779 | 0.117805 | 1.000000 | 0.117805 | under_segmentation, boundary_error |
| 3 | 372 | isic_0028154.jpg | 0.223165 | 0.125597 | 0.125597 | 1.000000 | over_segmentation, boundary_error |
| 4 | 747 | isic_0031789.jpg | 0.235960 | 0.133761 | 0.140424 | 0.738148 | over_segmentation, boundary_error |
| 5 | 664 | isic_0031041.jpg | 0.268470 | 0.155048 | 0.934302 | 0.156757 | under_segmentation, boundary_error |
| 6 | 709 | isic_0031450.jpg | 0.305647 | 0.180392 | 0.376628 | 0.257178 | under_segmentation, boundary_error |
| 7 | 163 | isic_0025953.jpg | 0.410111 | 0.257949 | 0.980103 | 0.259307 | under_segmentation, boundary_error |
| 8 | 226 | isic_0026646.jpg | 0.416816 | 0.263277 | 0.271735 | 0.894276 | over_segmentation, boundary_error |
| 9 | 297 | isic_0027425.jpg | 0.458480 | 0.297420 | 0.341914 | 0.695636 | over_segmentation, boundary_error |
| 10 | 213 | isic_0026452.jpg | 0.470236 | 0.307391 | 1.000000 | 0.307391 | under_segmentation, boundary_error |
| 11 | 260 | isic_0026952.jpg | 0.497586 | 0.331191 | 0.729895 | 0.377452 | under_segmentation, boundary_error |
| 12 | 458 | isic_0028858.jpg | 0.540598 | 0.370424 | 1.000000 | 0.370424 | under_segmentation, boundary_error |
| 13 | 457 | isic_0028849.jpg | 0.546856 | 0.376326 | 1.000000 | 0.376326 | under_segmentation, boundary_error |
| 14 | 599 | isic_0030280.jpg | 0.560385 | 0.389260 | 0.991164 | 0.390616 | under_segmentation, boundary_error |
| 15 | 528 | isic_0029486.jpg | 0.564784 | 0.393518 | 0.393518 | 1.000000 | over_segmentation, boundary_error |
