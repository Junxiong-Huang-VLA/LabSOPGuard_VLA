# val-per-class — 生成各类最佳检测样图

对当前最佳权重运行验证，生成每个类别置信度最高的带框样图。

## 执行步骤

### 1. 读取当前权重配置
读取 `configs/model/detection_runtime.yaml` 获取当前 `model` 字段路径。

### 2. 运行测试集验证（生成统计图）
```bash
cd D:/LabEmbodiedVLA/LabSOPGuard
yolo val \
  model=<model_path> \
  data=data/dataset/dataset.yaml \
  imgsz=640 \
  split=test \
  save=True \
  plots=True \
  project=outputs/val \
  name=<model_name>_test
```

### 3. 生成各类最佳检测样图
运行：
```bash
python tools/export_best_per_class.py
```
输出到 `outputs/val/best_per_class/`。

### 4. 展示结果
依次展示：
1. `confusion_matrix_normalized.png` — 混淆矩阵
2. `BoxPR_curve.png` — PR 曲线（各类 mAP@0.5）
3. `BoxF1_curve.png` — F1-Confidence 曲线
4. 各类最佳检测样图（`best_per_class/` 目录下所有图片）

### 5. 汇总报告
以表格形式输出各类指标：P / R / mAP50 / mAP50-95，并标注强/弱类别。
