# 基于 U-Net 改进模型的皮肤病灶图像分割系统

## 1. 项目背景

皮肤病灶图像分割是医学图像分析中的重要任务。其目标是在皮肤镜图像中自动定位病灶区域，并生成像素级二值 mask。准确的病灶分割结果可以为病灶面积统计、边界形态分析、后续良恶性分类和辅助诊断提供基础。

U-Net 及其改进模型在医学图像分割中应用广泛，原因在于其编码器-解码器结构能够同时利用深层语义信息和浅层空间细节。本项目以 U-Net 为基础，进一步实现 Attention U-Net，并通过 `segmentation-models-pytorch` 支持 U-Net++、DeepLabV3+ 等高性能结构。

## 2. 项目目标

本项目目标是实现一个完整的皮肤病灶图像分割系统，包括数据集读取、数据预处理、数据增强、模型训练、模型评估、结果可视化、Gradio Web Demo 和 Kaggle 云训练流程。项目训练主要在 Kaggle GPU 上完成，本地支持 CPU/CUDA 自动选择，用于预测、评估和 Demo 展示。

## 3. 数据集说明

数据集由皮肤病灶图像和对应二值 mask 组成。图像通常为 RGB 皮肤镜图像，mask 为单通道二值图，其中前景表示病灶区域，背景表示非病灶区域。项目要求 image 和 mask 以相同文件名 stem 匹配，例如 `ISIC_001.jpg` 对应 `ISIC_001.png`。

实际实验数据集路径通过 YAML 配置文件传入，不在源代码中写死本地路径或 Kaggle 路径。

## 4. 方法原理

### 4.1 二分类图像分割任务定义

二分类图像分割的输入为一张 RGB 图像，输出为与输入空间尺寸对应的单通道预测图。模型输出 logits，训练时使用适合二分类任务的 loss，推理时对 logits 执行 sigmoid 得到概率图，并通过阈值生成二值 mask。

### 4.2 U-Net 网络结构

U-Net 由编码器、解码器和跳跃连接组成。编码器逐步下采样以提取高层语义特征，解码器逐步上采样恢复空间分辨率。跳跃连接将编码器中的浅层特征与解码器特征拼接，从而保留边界和纹理等细节信息。

### 4.3 Encoder-Decoder

Encoder-Decoder 结构适合分割任务。Encoder 通过卷积和池化扩大感受野，Decoder 通过上采样恢复像素级预测。该结构能够在全局语义理解和局部定位之间取得平衡。

### 4.4 Skip Connection

Skip Connection 是 U-Net 的关键设计。它将编码器同尺度特征直接传递给解码器，有助于恢复病灶边界细节，减少上采样过程中空间信息丢失。

### 4.5 Attention U-Net

Attention U-Net 在跳跃连接处引入注意力门控机制。该机制利用解码器中的语义信息对编码器特征进行筛选，使模型更关注与病灶相关的区域，减少背景干扰。

### 4.6 U-Net++ / DeepLabV3+

U-Net++ 通过嵌套跳跃连接和更密集的特征融合改善编码器与解码器之间的语义差距。DeepLabV3+ 利用空洞卷积和多尺度上下文建模能力，适合处理尺度变化明显的分割任务。本项目通过预训练 encoder 提升模型特征提取能力，用于追求更高 Dice 和 IoU。

### 4.7 Loss 函数

BCEWithLogitsLoss 适合像素级二分类任务。Dice Loss 直接优化分割重叠程度，能够缓解前景和背景类别不平衡问题。BCE + Dice Loss 综合像素级分类误差和区域重叠误差，是本项目默认推荐 loss。

## 5. 系统设计

### 5.1 数据加载模块

数据加载模块负责读取图像和 mask，自动按文件名匹配，转换 mask 为单通道二值形式，并使用 Albumentations 对图像和 mask 进行同步增强。

### 5.2 模型模块

模型模块包含手写 U-Net、Attention U-Net 和模型工厂。模型工厂统一创建 U-Net、Attention U-Net、U-Net++、DeepLabV3+ 和 FPN，并支持 ImageNet 预训练 encoder。

### 5.3 训练模块

训练模块负责训练循环、验证循环、mixed precision、scheduler、early stopping、best/last checkpoint 保存、训练曲线保存和预测样例保存。

### 5.4 评估模块

评估模块加载指定 checkpoint，在验证集或测试集上计算 Dice、IoU、Precision、Recall 和平均 loss，并将结果保存为 CSV 文件。

### 5.5 可视化模块

可视化模块支持保存原图、真实 mask、预测 mask 和叠加图，并计算预测病灶面积比例。训练结束后会保存曲线图和样例预测图。

### 5.6 Kaggle 训练模块

项目提供 Kaggle 配置文件和 Kaggle Notebook。Kaggle 训练使用 GPU，输出保存到 `/kaggle/working/outputs` 和 `/kaggle/working/checkpoints`。Notebook 会动态生成 runtime config，避免将 Kaggle 数据路径写入源代码。

### 5.7 本地 Gradio Demo 模块

本地 Demo 支持上传单张皮肤病灶图像、选择模型和 checkpoint，并输出预测 mask、叠加图、病灶面积比例和推理时间。本地无 GPU 时自动使用 CPU。

## 6. 训练前检查流程

正式长时间训练前，项目要求执行 dataset check、mask 可视化、small batch overfit 和 quick train。

### 6.1 数据检查

`scripts/check_dataset.py` 检查训练集和验证集路径是否存在、image/mask 数量是否一致、文件名是否能正确匹配、mask 是否为有效二值 mask，并统计前景像素比例。

### 6.2 Mask 可视化

数据检查脚本会随机保存 8 张 image + mask 叠加图到 `outputs/sanity_check/`。如果叠加图中 mask 与病灶区域明显不对齐，则不能开始正式训练。

### 6.3 Small Batch Overfit

`scripts/overfit_small_batch.py` 从训练集中取少量样本，训练 50 到 100 epochs。若模型、loss、数据读取和 mask 对齐正常，loss 应明显下降，Dice 和 IoU 应明显上升。

### 6.4 Quick Train

`scripts/quick_train.py` 使用完整训练集进行 1 到 3 epochs 的短训练，确认完整训练流程、验证流程、checkpoint 保存、曲线保存和样例预测都能正常运行。

## 7. 实验设计

### 7.1 Kaggle GPU 训练环境

正式训练主要在 Kaggle GPU 环境中完成。Kaggle 配置使用 `/kaggle/input/` 读取数据，训练结果保存到 `/kaggle/working/`。

### 7.2 本地 CPU/CUDA 推理环境

本地环境用于预测、评估和 Gradio Demo。设备选择为 `auto` 时，若 CUDA 可用则使用 CUDA，否则使用 CPU。

### 7.3 模型对比实验

计划比较手写 U-Net、Attention U-Net、U-Net++ 和 DeepLabV3+ 在 Dice、IoU、Precision 和 Recall 上的表现。

### 7.4 Loss 对比实验

计划比较 BCE、Dice Loss 和 BCE + Dice Loss 对训练稳定性和分割指标的影响。

### 7.5 数据增强对比实验

计划比较开启数据增强与关闭数据增强时模型泛化能力的差异。

### 7.6 输入尺寸对比实验

计划比较 256 和 384 输入尺寸对细节恢复、显存占用和训练速度的影响。

## 8. 实验结果

当前不编造具体实验数值。真实训练完成后，应将 `outputs/experiment_results.csv` 中结果填入下表。

| 实验名称 | 模型 | Loss | 输入尺寸 | Dice | IoU | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 待填入实验结果 | U-Net | BCE + Dice | 256 | 待填入实验结果 | 待填入实验结果 | 待填入实验结果 | 待填入实验结果 |
| 待填入实验结果 | Attention U-Net | BCE + Dice | 256 | 待填入实验结果 | 待填入实验结果 | 待填入实验结果 | 待填入实验结果 |
| 待填入实验结果 | U-Net++ | BCE + Dice | 384 | 待填入实验结果 | 待填入实验结果 | 待填入实验结果 | 待填入实验结果 |
| 待填入实验结果 | DeepLabV3+ | BCE + Dice | 384 | 待填入实验结果 | 待填入实验结果 | 待填入实验结果 | 待填入实验结果 |

## 9. 结果分析

待真实训练完成后，应从以下角度分析结果：不同模型的 Dice 和 IoU 差异；Attention 机制是否改善边界定位；预训练 encoder 是否提升收敛速度和最终指标；不同 loss 对前景较小样本的影响；失败案例中是否存在边界模糊、低对比度或标注噪声。

## 10. Gradio 系统展示

Gradio Demo 支持用户上传单张图像并选择 checkpoint 进行预测。系统输出原图、预测 mask、叠加图、病灶面积比例、推理时间和当前设备。训练完成后，可将 Kaggle 下载的 `best_model.pth` 放入本地 `checkpoints/` 目录，并运行 `python app.py` 展示系统。

## 11. 总结与展望

本项目实现了从数据检查、模型训练、指标评估到可视化 Demo 的完整医学图像分割工程。项目既包含可解释的手写 U-Net，也支持基于预训练 encoder 的高性能模型，适合作为本科深度学习项目展示。

后续可进一步加入交叉验证、独立测试集评估、更多 encoder 对比、ONNX 导出、批量预测和不确定性可视化，以提升系统完整性和工程实用性。
