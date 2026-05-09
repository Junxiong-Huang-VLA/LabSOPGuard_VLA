# 训练报告

## 1. 训练目标

本次训练目标是得到适用于实验室视频分析链路的 YOLO 检测模型，并作为运行时默认权重接入正式系统。

## 2. 训练数据

数据配置文件：

- `data/dataset/dataset.yaml`

类别数：

- `13`

主要类别：

- `balance`
- `beaker`
- `gloved_hand`
- `lab_coat`
- `paper`
- `reagent_bottle`
- `sample_bottle`
- `sample_bottle_blue`
- `spatula`
- `tube`
- `tube-cap`
- `spearhead`
- `pipette`

## 3. 训练环境

- 平台：AutoDL
- 系统：Ubuntu 22.04
- GPU：NVIDIA GeForce RTX 5090
- 显存：约 32GB
- Python：3.12.3
- PyTorch：2.8.0 + cu128
- Ultralytics：8.4.37

## 4. 训练参数

最终稳定运行参数：

- 模型：`yolo26s.pt`
- Epochs：`100`
- Image size：`640`
- Batch size：`16`
- Device：`0`
- Workers：`8`
- Run name：`yolo26s_autodl_8_1_1`

为避免显存风险，调试阶段也验证过：

- `imgsz=512`
- `batch=8`

## 5. 训练结果

### 5.1 验证集

- P：`0.958`
- R：`0.945`
- mAP50：`0.978`
- mAP50-95：`0.921`

### 5.2 测试集

- P：`0.962`
- R：`0.949`
- mAP50：`0.975`
- mAP50-95：`0.916`

## 6. 训练产物

核心产物目录：

- `outputs/training/yolo26s_autodl_8_1_1/`

关键文件：

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`

## 7. 已接入正式链路

当前默认运行权重已切换为：

- `outputs/training/yolo26s_autodl_8_1_1/weights/best.pt`

配置文件：

- `configs/model/detection_runtime.yaml`

## 8. 训练过程中的问题与处理

### 8.1 数据路径兼容

原始 `dataset.yaml` 存在 Windows 绝对路径问题，已在训练链路中做兼容处理。

### 8.2 旧依赖与新 GPU 兼容

未采用仓库旧版 `torch==2.1.1` 作为 AutoDL 训练环境基础，而是利用远端现成较新的 Torch / CUDA 栈完成训练。

### 8.3 detect / segment 混合警告

训练日志提示数据集中存在 detect / segment 混合信息，最终按检测任务训练，分割信息被忽略。

## 9. 结论

本次 `best.pt` 可作为当前正式版基线检测模型。  
测试集指标与验证集接近，没有明显过拟合，可用于当前视频分析、报警辅助和标注视频导出链路。
