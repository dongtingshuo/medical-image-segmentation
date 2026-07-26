# Security Policy / 安全政策

## Supported Version / 支持版本

Security fixes target the latest commit on `main` and the latest published release.

安全修复面向 `main` 分支最新提交和最新发布版本。

## Reporting / 报告方式

Do not publish credentials, private dataset paths, patient information, or exploit details in a public issue. Use [GitHub private vulnerability reporting](https://github.com/dongtingshuo/medical-image-segmentation/security/advisories/new).

请勿在公开 Issue 中发布凭据、私有数据集路径、患者信息或漏洞利用细节。请使用 [GitHub 私有漏洞报告](https://github.com/dongtingshuo/medical-image-segmentation/security/advisories/new)。

## Checkpoint Safety / 模型权重安全

PyTorch checkpoints can be unsafe when loaded from untrusted sources. This project loads checkpoints with `weights_only=True` and expects dictionary payloads produced by this repository. Use the verified GitHub Release asset and compare its SHA256 digest with `models/model_manifest.yaml`.

PyTorch checkpoint 在来源不可信时可能存在安全风险。本项目使用 `weights_only=True` 加载，并要求输入为本仓库生成的字典格式。请使用经验证的 GitHub Release 权重，并与 `models/model_manifest.yaml` 中的 SHA256 进行比对。

## Dependency Compatibility Boundary / 依赖兼容边界

The reproducibility lock still contains legacy major versions used by the published checkpoint and archived Tesla P100 workflow. An automated advisory scan found upgrade candidates, but PyTorch, Transformers, Gradio, and ONNX major/minor migrations must be tested together against the verified release checkpoint, export paths, Docker image, and Kaggle hardware before they can be adopted. Until that migration is completed, use the pinned environment only with trusted inputs and keep the Gradio/Docker demo on a private or loopback interface; do not expose it as a public service.

可复现依赖锁仍包含发布权重与已归档 Tesla P100 流程使用的历史大版本。自动化漏洞扫描已经发现升级候选，但 PyTorch、Transformers、Gradio 与 ONNX 的大版本/次版本迁移必须共同通过发布权重、导出流程、Docker 镜像和 Kaggle 硬件验证后才能采用。在完成迁移前，仅使用可信输入运行固定环境，并将 Gradio/Docker 演示限制在私有网络或回环地址；不要将其作为公开服务暴露。

## Medical Data / 医疗数据

Do not commit identifiable medical data. Dataset licenses, consent requirements, de-identification, and access controls remain the responsibility of the dataset user.

不得提交可识别的医疗数据。数据授权、知情同意、去标识化和访问控制由数据使用者负责。
