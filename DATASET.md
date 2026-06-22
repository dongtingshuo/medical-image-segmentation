# Dataset Card / 数据集说明

## Experiment Dataset / 实验数据集

The reported Kaggle experiments used the **ISIC 2017 Skin Lesion Analysis Towards Melanoma Detection** image and segmentation-mask split through the Kaggle mirror below:

已报告的 Kaggle 实验使用 **ISIC 2017 Skin Lesion Analysis Towards Melanoma Detection** 图像与分割 mask，数据来自下列 Kaggle 镜像：

- Official challenge / 官方挑战：https://challenge.isic-archive.com/landing/2017/
- Official data index / 官方数据索引：https://challenge.isic-archive.com/data/#2017
- Kaggle mirror / Kaggle 镜像：https://www.kaggle.com/datasets/moon1570/isic-2017-train-val-test-images-and-masks

The project does not redistribute the medical dataset.

本项目不再分发医疗数据集。

## Split Used / 已使用划分

| Split | Images | Masks | Used for reported metrics |
| --- | ---: | ---: | --- |
| Train | 2000 | 2000 | Training |
| Validation | 150 | 150 | Model selection and reported metrics |
| Test | Not evaluated | Not evaluated | No |

The repository currently reports validation results only. The test split was not evaluated, and no test metric is claimed.

仓库当前仅报告验证集结果；未评估 test split，也不声明任何测试集指标。

## Integrity Checks / 完整性检查

- Exact image/mask filename-stem matching.
- 2000/2000 training pairs and 150/150 validation pairs.
- No invalid binary masks or image/mask size mismatches in the saved sanity report.
- Mean foreground ratio: `0.192484`.

Saved evidence is available under `docs/assets/sanity_check/`.

检查证据保存在 `docs/assets/sanity_check/`。

## Licensing and Responsible Use / 授权与责任使用

The MIT License in this repository covers project code only. It does not grant rights to ISIC images or annotations. Users must review the current terms on the official source and Kaggle mirror before downloading or redistributing data. Medical images must remain de-identified and must not be used for clinical decisions through this project.

本仓库的 MIT License 仅适用于项目代码，不授予 ISIC 图像或标注的使用权。下载或再分发数据前，使用者必须查阅官方来源和 Kaggle 镜像的当前条款。医疗图像必须保持去标识化，不得通过本项目用于临床决策。
