# v1.5 Multi-Stage Kaggle Training

## Scope

v1.5 is a resumable Kaggle workflow for multi-source skin-lesion segmentation. It screens three architectures, trains two five-fold teacher families, generates leak-free OOF soft masks, distills two full-data students, and locks a maximum-five-member ensemble before test/external evaluation.

The workflow is experimental and is not intended for clinical use.

## Inputs

Mount these datasets as Kaggle inputs and pass their locations through CLI arguments or the `V15_*` environment variables used by `notebooks/kaggle_v1_5.py`:

- ISIC 2017 train/validation/test images and masks.
- ISIC 2018 external images and masks.
- ISIC 2016 train and test images/masks, combined under the supplied image and mask roots.
- PH2 images and expert masks.

No raw image is uploaded to external tracking services by default. `data_manifest.csv` records accepted/removed samples, hashes, source, contrast, lesion ratio, and duplicate decisions.

## W&B Tracking

W&B support remains in the codebase, but v1.5 Kaggle configs now default to `tracking.enabled: false`. No W&B Secret is required for the default formal/debug kernels, and the pipeline uses local histories, metrics, checkpoints, and the resumable state package as the source of truth.

To re-enable W&B for a future experiment, set `tracking.enabled: true`, choose `tracking.mode: online` or `offline`, attach Kaggle Secrets named `WANDB_API_KEY` and optionally `WANDB_ENTITY`, and pass `--require-wandb-online` only when the run must hard-fail before training if online W&B is unavailable.

## Sessions

Run the same entrypoint for every Kaggle session:

```bash
python notebooks/kaggle_v1_5.py \
  --isic16-dataset OWNER/ISIC16_SLUG \
  --ph2-dataset OWNER/PH2_SLUG
```

The default recursively classifies image and mask files below each dataset root, including separate ISIC 2016 train/test folders and nested PH2 lesion masks. Use the `--*-images-rel` and matching `--*-masks-rel` arguments only when a mirror has an ambiguous layout.

At the end of a session, publish `v1_5_state.zip` and `v1_5_state.zip.sha256` as the next Kernel version's input dataset. The notebook auto-detects a mounted `v1_5_state.zip`; `--state-input` can select one explicitly.

The controller validates:

- source Git commit;
- base config SHA256;
- data manifest SHA256;
- folds SHA256;
- stable W&B task-to-run mapping.

It rematerializes raw/prepared data from the mounted inputs and reconstructs fold hardlinks after restore. Raw medical images, fold copies, and validation probability caches are excluded from the state archive; regenerated data and folds must reproduce their saved SHA256 values before training resumes.

## Direct Pipeline Command

```bash
python scripts/run_v1_5_pipeline.py \
  --config configs/kaggle_v1_5.yaml \
  --output-root /path/to/working/research_v1_5 \
  --isic17-root /path/to/isic17 \
  --isic18-root /path/to/isic18 \
  --isic16-images /path/to/isic16/images \
  --isic16-masks /path/to/isic16/masks \
  --ph2-images /path/to/ph2/images \
  --ph2-masks /path/to/ph2/masks
```

The default time budget is 570 minutes with a 30-minute packaging reserve. `last_model.pth` is retained only in the state package; W&B model artifacts contain `best_model.pth` plus whitelisted configs, histories, metrics, overlays, and manifests.

## Release Decision

`selection/locked_decision.json` freezes ensemble members, threshold, TTA, and post-processing on the original ISIC 2017 validation set. `scripts/evaluate_locked_v1_5.py` then evaluates test and external splits once and writes `final/evaluation_complete.json`.

- Replace the default model only when the distilled `fast` model passes every acceptance threshold.
- Publish only `best_accuracy` when the ensemble passes but `fast` does not.
- Keep the current default model when neither variant passes.

## Completed Results

The completed resumable run selected `student-segformer` as `fast`. Greedy validation selection produced a three-member `best_accuracy` ensemble: `student-segformer`, `teacher-segformer-fold2`, and `teacher-unetpp-fold1`. The locked thresholds were `0.45` for `fast` and `0.425` for the ensemble; neither used TTA or post-processing.

| Variant | Split | Macro Dice | Boundary F1 | Low-Contrast Dice | Accepted |
| --- | --- | ---: | ---: | ---: | --- |
| `fast` | ISIC 2017 test | 0.855476 | 0.480229 | 0.826353 | No |
| `best_accuracy` | ISIC 2017 test | 0.858502 | 0.490880 | 0.827653 | No |
| `fast` | ISIC 2018 external | 0.928365 | 0.619147 | - | Yes |
| `best_accuracy` | ISIC 2018 external | 0.929573 | 0.642300 | - | Yes |

The external gates passed, but both variants missed the independent-test Dice and low-contrast gates. Therefore `publish_default` and `publish_best_accuracy` are both `false`; the existing default model remains unchanged. Raw locked decisions and final metrics are linked from the curve gallery.

## Training Curves

All 15 completed runs are directly embedded in the [v1.5 curve gallery](training_curves/v1.5.md): three architecture-screening runs, ten five-fold teachers, and two distilled students. Every curve is paired with its raw `metrics.csv`.

## Validation

```bash
python -m pytest -q
ruff check train.py src/ scripts/run_v1_5_pipeline.py scripts/generate_oof_targets.py \
  scripts/select_ensemble_v1_5.py scripts/evaluate_locked_v1_5.py notebooks/kaggle_v1_5.py
python notebooks/kaggle_v1_5.py --debug
```

Formal submission should only follow a successful Kaggle GPU debug run with all required datasets and secrets mounted.
