## Summary / 变更摘要

Describe the maintenance issue and the smallest corrective change.

说明维护问题与最小必要修复。

## Scope / 范围

- [ ] Security fix / 安全修复
- [ ] Dependency compatibility / 依赖兼容
- [ ] Reproducibility correction / 可复现性修正
- [ ] Critical bug / 严重 bug
- [ ] Documentation correction / 文档修正

## Validation / 验证

- [ ] `python -m pytest -q`
- [ ] `ruff check .`
- [ ] Relevant CLI smoke tests / 相关 CLI 检查
- [ ] Documentation links checked / 文档链接已检查

List exact commands and results:

列出实际命令与结果：

## Evidence and safety / 证据与安全

- [ ] No dataset, patient data, credentials, checkpoints, or generated training outputs were added.
- [ ] Reported metrics and default-model decisions were not changed without traceable source artifacts.
- [ ] 未加入数据集、患者数据、凭据、checkpoint 或训练产物。
- [ ] 未在缺少可追溯原始产物时改写指标或默认模型结论。
