# v1.6 Target-Domain Generalization Workflow

v1.6 is a five-session Kaggle workflow targeting the independent ISIC 2017 test macro-Dice gate without training on, selecting on, or repeatedly evaluating that split. ISIC 2018 remains external-only.

## Inputs

Mount the existing ISIC 2017, ISIC 2018, ISIC 2016, and PH2 datasets plus:

- `kmader/skin-cancer-mnist-ham10000` for images and `HAM10000_metadata.csv`.
- `tschandl/ham10000-lesion-segmentations` for reviewed lesion masks.

HAM10000 is pretraining-only. The data manifest records its `CC BY-NC-SA 4.0` restriction, reviewed-label status, lesion group, and all duplicate removals. Any overlap with ISIC 2017 validation/test or ISIC 2018 external is removed before training.

## Sessions

1. Data audit and HAM10000 pretraining for U-Net++ EfficientNet-B3 and SegFormer MiT-B3.
2. Five target-domain U-Net++ teachers.
3. Five target-domain SegFormer teachers.
4. Cross-fit OOF generation and two confidence-gated students.
5. OOF-locked family weighting, ISIC17 validation threshold calibration, and one-time test/external evaluation.

Every session publishes `v1_6_state.zip` with a SHA256 sidecar. The next private Kernel version must mount the latest completed version as a kernel source. State excludes raw/prepared images and rematerializes them after validating the data and fold hashes.

## Release Rule

`best_accuracy` contains all ten teachers, family-weighted by target-domain OOF. It is published only when its test Dice is at least `0.884766` and all existing test/external acceptance metrics pass. A student becomes the default only when its own test and external evaluations also pass. W&B is disabled: no Secret is read and no W&B run is created.
