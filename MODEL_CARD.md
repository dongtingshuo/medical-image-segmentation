# Model Card: U-Net++ EfficientNet-B3 / 模型卡：U-Net++ EfficientNet-B3

## Model Summary / 模型概要

The default inference model is U-Net++ with an EfficientNet-B3 encoder initialized from ImageNet weights and trained for binary skin-lesion segmentation on Kaggle GPU.

默认推理模型为 U-Net++ + EfficientNet-B3 encoder，encoder 以 ImageNet 权重初始化，并在 Kaggle GPU 上进行二分类皮肤病灶分割训练。

## Verified Artifact / 已验证权重

- Release: https://github.com/dongtingshuo/medical-image-segmentation/releases/tag/v1.0.0
- Asset: `best_model.pth`
- Size: `159715596` bytes
- SHA256: `4b04ccd5f4fbdad492a91ea9866d31b9329a886e74464ddf42fffa1854f76577`
- Embedded best epoch: `4`
- Runtime config: `configs/final_model.yaml`
- Recommended threshold: `0.35`
- Machine-readable manifest: `models/model_manifest.yaml`

## Validation Results / 验证结果

| Metric | Value |
| --- | ---: |
| Validation loss at selected epoch | 0.153719 |
| Dice | 0.872120 |
| IoU | 0.792033 |
| Precision | 0.905242 |
| Recall / Sensitivity | 0.881161 |
| Inference time | Not recorded |

These values are from the 150-image validation split described in `DATASET.md`. They are not external-test or clinical-performance claims.

上述数值来自 `DATASET.md` 记录的 150 张验证集，不代表外部测试或临床性能。

Post-hoc threshold search selected `0.35` by validation Dice. Repeated engineering evaluation was also run on validation, ISIC 2017 independent test, and ISIC 2018 external splits; see `README.md`, `docs/report.md`, and the release experiment artifacts for those reports.

后处理阈值搜索按验证集 Dice 选择 `0.35`。项目还完成了验证集、ISIC 2017 独立测试集和 ISIC 2018 外部集的重复工程评估；相关报告见 `README.md`、`docs/report.md` 和 Release 实验产物。

## Intended Use / 预期用途

- Reproduce the repository's segmentation inference workflow.
- Compare engineering behavior across CPU and CUDA environments.
- Demonstrate mask, overlay, and error-analysis visualization.

本模型不得用于临床诊断、治疗建议或真实医疗决策。

## Current Limitations / 当前限制

- Evaluation is dataset-level engineering validation, not clinical validation.
- Package versions and source commit were not embedded in the legacy v1.0.0 checkpoint.
- No cross-validation, subgroup, calibration, reader study, or clinical validation was performed.
- Performance may degrade under acquisition, device, skin-tone, lesion-type, or annotation shifts.
- The selected model showed mild overfitting after its best epoch.

Future checkpoints produced by the current code use checkpoint format version 2 and embed Python/PyTorch metadata.

当前代码新生成的 checkpoint 使用格式版本 2，并写入 Python/PyTorch 环境元数据。

## Medical Disclaimer / 医学免责声明

This model is intended only for medical image segmentation experiments and engineering workflow validation. It is not intended for clinical diagnosis, treatment recommendation, or real-world medical decision-making.

本模型仅用于医学图像分割算法实验和工程流程验证，不用于临床诊断、治疗建议或真实医疗决策。
