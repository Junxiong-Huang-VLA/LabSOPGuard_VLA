# 角色一：检测工程师（Detection Engineer）

## 职责定位

负责 YOLO 目标检测模型的训练、评估、权重管理与迭代升级。是模型质量的第一责任人。

## 核心工作内容

### 数据管理
- 维护 `data/dataset/`，确保 train/val/test 三分集比例（8:1:1）
- 监控各类标注数量，`balance/beaker/gloved_hand/lab_coat/paper/reagent_bottle/sample_bottle/sample_bottle_blue/spatula` 各类 train ≥ 500 条
- `tube/tube-cap/spearhead/pipette` 尚无标注，需从 Roboflow 云端导出并合并
- 发现类别分布不均时，使用 `tools/build_ppe_hardcase_pack.py` 补充困难样本

### 训练
- 训练命令：`yolo detect train model=yolo26s.pt data=data/dataset/dataset.yaml epochs=100 imgsz=640 batch=16 device=0`
- 远端训练优先走 AutoDL（RTX 5090），参考 `docs/autodl_quickstart.md`
- 训练完成后只改 `configs/model/detection_runtime.yaml` 的 `model` 字段，不改代码

### 评估
- 验证命令：`yolo val model=<best.pt> data=data/dataset/dataset.yaml split=test plots=True`
- 目标指标：mAP50 ≥ 0.97，mAP50-95 ≥ 0.90
- 当前最佳：`yolo26s_autodl_8_1_1`（mAP50=0.977，mAP50-95=0.925，测试集）
- 生成各类最佳检测样图：`python tools/export_best_per_class.py`

### 权重归档
- 每次训练结果存 `outputs/training/<run_name>/`，至少保留 `best.pt` 和 `results.csv`
- 在 `docs/training_report.md` 更新最新训练记录

## 关注指标

| 类别 | 当前 mAP50 | 目标 |
|------|-----------|------|
| lab_coat | 0.940 | ≥ 0.96 |
| beaker | 0.959 | ≥ 0.97 |
| gloved_hand | 0.959 | ≥ 0.97 |
| sample_bottle / spatula | 0.995 | 维持 |

## 常用命令
```bash
# 快速启动训练评估
/train-yolo
/val-per-class
```

## 禁止事项
- 禁止在 Python 代码中硬编码权重路径
- 禁止直接修改 `data/dataset/` 中已有标注，修改需走 PR 审查
