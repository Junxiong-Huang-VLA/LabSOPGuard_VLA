# train-yolo — YOLO 模型训练助手

帮助用户准备数据集、生成训练命令、训练完成后切换权重。

## 执行步骤

### 1. 检查数据集
读取 `data/dataset/dataset.yaml`，统计各类标注数量：
```bash
cat data/dataset/labels/train/*.txt | awk '{print $1}' | sort -n | uniq -c | sort -rn
```
报告哪些类别样本不足（< 100 条）。

### 2. 生成训练命令
基于用户需求（epoch 数、模型大小 s/n）生成完整训练命令：
```bash
yolo detect train \
  model=yolo26s.pt \
  data=data/dataset/dataset.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  name=yolo26s_<run_name> \
  project=outputs/training
```

### 3. AutoDL 上传提示
如需在 AutoDL 训练，提示用户：
- 打包数据：`tar -czf dataset.tar.gz data/dataset/`
- 上传权重和数据
- 参考 `docs/autodl_quickstart.md`

### 4. 训练完成后切换权重
训练完成后，自动更新 `configs/model/detection_runtime.yaml` 的 `model` 字段：
```yaml
model: outputs/training/<run_name>/weights/best.pt
```
并运行验证：
```bash
yolo val model=outputs/training/<run_name>/weights/best.pt data=data/dataset/dataset.yaml imgsz=640 split=test plots=True project=outputs/val name=<run_name>_test
```

### 5. 展示结果
读取 `outputs/val/<run_name>_test/` 下的 confusion_matrix_normalized.png、BoxPR_curve.png 并展示。
