# Changelog / 变更记录

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
