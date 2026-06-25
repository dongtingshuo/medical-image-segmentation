# Dataset Check Report

## Summary

- Train images: 2000
- Train masks: 2000
- Train matched pairs: 2000
- Val images: 150
- Val masks: 150
- Val matched pairs: 150
- Mean foreground ratio: 0.192484
- Min foreground ratio: 0.002977
- Max foreground ratio: 0.958930

## Binary Mask Check

- Invalid binary masks: 0
- Image/mask size mismatches: 0

## Warnings

- No all-black or all-white masks found.

## Saved Overlay Samples

- /kaggle/working/research_v1_2/preflight/sanity_check/dataset_overlay_00_isic_0012940.png
- /kaggle/working/research_v1_2/preflight/sanity_check/dataset_overlay_01_isic_0000262.png
- /kaggle/working/research_v1_2/preflight/sanity_check/dataset_overlay_02_isic_0000053.png
- /kaggle/working/research_v1_2/preflight/sanity_check/dataset_overlay_03_isic_0013525.png
- /kaggle/working/research_v1_2/preflight/sanity_check/dataset_overlay_04_isic_0008626.png
- /kaggle/working/research_v1_2/preflight/sanity_check/dataset_overlay_05_isic_0002948.png
- /kaggle/working/research_v1_2/preflight/sanity_check/dataset_overlay_06_isic_0000900.png
- /kaggle/working/research_v1_2/preflight/sanity_check/dataset_overlay_07_isic_0000343.png

## Pass Criteria

- Image and mask counts are equal for each split.
- Image and mask filename stems match.
- Masks are binary or can be safely thresholded.
- Overlay samples align with visible lesion regions.
