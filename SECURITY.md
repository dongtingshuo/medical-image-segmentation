# Security Policy / 安全政策

## Supported Version / 支持版本

Security fixes target the latest commit on `main` and the latest published release.

安全修复面向 `main` 分支最新提交和最新发布版本。

## Reporting / 报告方式

Do not publish credentials, private dataset paths, patient information, or exploit details in a public issue. Use GitHub private vulnerability reporting when it is enabled for the repository.

请勿在公开 Issue 中发布凭据、私有数据集路径、患者信息或漏洞利用细节。仓库启用 GitHub 私有漏洞报告后，请优先使用该渠道。

## Checkpoint Safety / 模型权重安全

PyTorch checkpoints can be unsafe when loaded from untrusted sources. This project loads checkpoints with `weights_only=True` and expects dictionary payloads produced by this repository. Use the verified GitHub Release asset and compare its SHA256 digest with `models/model_manifest.yaml`.

PyTorch checkpoint 在来源不可信时可能存在安全风险。本项目使用 `weights_only=True` 加载，并要求输入为本仓库生成的字典格式。请使用经验证的 GitHub Release 权重，并与 `models/model_manifest.yaml` 中的 SHA256 进行比对。

## Medical Data / 医疗数据

Do not commit identifiable medical data. Dataset licenses, consent requirements, de-identification, and access controls remain the responsibility of the dataset user.

不得提交可识别的医疗数据。数据授权、知情同意、去标识化和访问控制由数据使用者负责。
