# 模型卡

## 1. 模型概述

当前默认检测模型为实验室场景目标检测模型：

- 权重路径：`outputs/training/yolo26s_autodl_8_1_1/weights/best.pt`
- 模型用途：实验室台面与操作场景中的目标检测、PPE 辅助判断、视频标注可视化

语义分析模型为：

- DashScope / Qwen-VL

整体链路采用：

- YOLO 负责逐帧检测框
- Qwen 负责抽帧语义分析、活动描述、步骤提示、PPE 语义补充

## 2. 检测类别

当前数据集类别共 `13` 类：

1. `balance`
2. `beaker`
3. `gloved_hand`
4. `lab_coat`
5. `paper`
6. `reagent_bottle`
7. `sample_bottle`
8. `sample_bottle_blue`
9. `spatula`
10. `tube`
11. `tube-cap`
12. `spearhead`
13. `pipette`

## 3. 训练环境

- 平台：AutoDL
- GPU：RTX 5090 32GB
- 系统：Ubuntu 22.04
- Python：3.12.3
- Torch：2.8.0 + cu128
- Ultralytics：8.4.37

## 4. 指标结果

### 4.1 验证集结果

- Precision：`0.958`
- Recall：`0.945`
- mAP50：`0.978`
- mAP50-95：`0.921`

### 4.2 测试集结果

- Precision：`0.962`
- Recall：`0.949`
- mAP50：`0.975`
- mAP50-95：`0.916`

## 5. 类别表现摘要

相对较强的类别：

- `reagent_bottle`
- `sample_bottle`
- `sample_bottle_blue`
- `spatula`

相对较弱的类别：

- `beaker`
- `lab_coat`
- `paper`

## 6. 模型适用范围

适合：

- 实验台面正视或近正视视频
- 目标尺寸中等、遮挡有限的室内场景
- 需要输出检测框、物体列表、报警辅助的视频分析任务

不适合：

- 强运动模糊视频
- 极端背光、过曝或严重低照度环境
- 需要精细实例分割的任务
- 需要高精度护目镜检测的纯视觉任务

## 7. 已知限制

### 7.1 护目镜不是显式检测类

当前数据集没有稳定的 `goggles` 检测类，因此：

- `missing_goggles` 主要由 Qwen 语义判断补充
- 不应把护目镜告警等同于稳定的纯检测结果

### 7.2 标注集中包含 detect/segment 混合信息

训练日志出现过：

- `detect-segment mixed dataset`

当前训练按检测任务执行，分割信息被忽略。

## 8. 安全与使用声明

本模型用于实验室视频分析辅助，不应用作唯一安全判定依据。  
对于 PPE 缺失、危险操作、违规步骤等结论，仍需人工复核。

## 9. 推荐用法

推荐在以下场景使用：

1. 视频回放分析
2. 教学演示视频自动标注
3. 实验规范复盘
4. 违规操作预警辅助
