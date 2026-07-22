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

Retries and time-safe resumes expanded these five logical phases to eight physical Kaggle kernel sessions. Session 8 completed the locked final evaluation without restarting training from scratch.

## Release Rule

`best_accuracy` contains all ten teachers, family-weighted by target-domain OOF. It is published only when its test Dice is at least `0.884766` and all existing test/external acceptance metrics pass. A student becomes the default only when its own test and external evaluations also pass. W&B is disabled: no Secret is read and no W&B run is created.

## Completed Results

OOF selection locked the teacher-family weights at `0.70` U-Net++ and `0.30` SegFormer. The final ensemble used multi-scale flip TTA and threshold `0.70`; `fast` used `student-unetpp`, no TTA, and threshold `0.50`.

| Variant | Split | Macro Dice | Boundary F1 | Low-Contrast Dice | Accepted |
| --- | --- | ---: | ---: | ---: | --- |
| `fast` | ISIC 2017 test | 0.858215 | 0.489497 | 0.825290 | No |
| `best_accuracy` | ISIC 2017 test | 0.857879 | 0.490239 | 0.821152 | No |
| `fast` | ISIC 2018 external | 0.932705 | 0.644461 | - | Yes |
| `best_accuracy` | ISIC 2018 external | 0.947575 | 0.732097 | - | Yes |

The ten-teacher ensemble materially improved the external metrics but did not reach the independent-test Dice gate of `0.884766`. Both publication flags are `false`, so v1.6 is retained as an experiment record rather than a released replacement model.

## Training Curves

All 14 completed runs are directly embedded in the [v1.6 curve gallery](training_curves/v1.6.md): two HAM10000 pretraining runs, ten target-domain teachers, and two confidence-gated students. Every curve is paired with its raw `metrics.csv`; the teacher histories are the persisted target-adaptation stage from the final state and are not reconstructed.
