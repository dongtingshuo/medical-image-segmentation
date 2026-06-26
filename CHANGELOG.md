# Changelog / 变更记录

## Unreleased / 未发布

- Added `batch_predict.py` for directory-level inference with masks, overlays, lesion ratios, CSV, and JSON summaries.
- Added `export.py` for TorchScript and ONNX export with manifest and SHA256 checksums.
- Added Docker CPU demo runtime for the Gradio application.
- Added deployment-oriented tests for export and batch prediction workflows.
- Added v1.3 low-contrast specialist Kaggle workflow with contrast-aware augmentation variants.
- Added Tversky loss for false-negative-sensitive segmentation experiments.
- Added low-contrast variant comparison reports and replacement-decision criteria.
- Added training resume support from `last_model.pth`, including optimizer, scheduler, AMP scaler, metric history, and early-stopping state.
- Made the v1.3 Kaggle workflow restart-safe with completed-variant skipping and automatic resume.
- Reported the completed v1.3 Kaggle results; `contrast_aug_bce_dice` was best but did not meet the configured default-model replacement threshold.

## v1.2.0 - 2026-06-25

This release adds and reports the Kaggle research workflow for robustness analysis without changing the default inference checkpoint.

本版本新增并报告 Kaggle 研究增强流程，用于稳健性分析，不更改默认推理 checkpoint。

- Added 3-fold cross-validation with leakage checks.
- Added U-Net++ encoder comparison between EfficientNet-B3 and ResNet34.
- Added threshold search, subgroup analysis, and statistical summaries.
- Added a sanitized v1.2 artifact package that excludes materialized fold data and medical image files.
- Added representative v1.2 curves, overlays, and analysis reports under `docs/assets/`.
- Updated README, technical report, release notes, and final guide with real v1.2 Kaggle results.

## v1.1.0 - 2026-06-22

This release strengthens reproducibility, checkpoint safety, continuous integration, experiment traceability, and local inference behavior without changing the published v1.0.0 model weights.

本版本增强可复现性、checkpoint 安全、持续集成、实验可追溯性和本地推理行为，不修改 v1.0.0 已发布的模型权重。

- Added deterministic training controls and seeded DataLoader workers.
- Added versioned, safe checkpoint loading and architecture validation.
- Made prediction samples use the best checkpoint rather than the final epoch.
- Separated minimum validation loss from loss at the best monitored epoch.
- Added Specificity and Boundary F1 metrics.
- Added checkpoint-aware cached inference for the Gradio demo.
- Added pinned dependencies, GitHub Actions CI, pre-commit configuration, and governance documents.
- Added dataset provenance and a model manifest.

## v1.0.0 - 2026-06-18

- Published the Kaggle-trained U-Net++ EfficientNet-B3 checkpoint.
- Added baseline and high-accuracy experiment results, local inference, evaluation, and Gradio workflows.
- Added MIT License.
