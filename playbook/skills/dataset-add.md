# dataset-add — 合并新标注数据到主数据集

将 Roboflow 导出的 zip 或本地新标注数据合并进主数据集，并重新划分 train/val/test。

## 用法
```
/dataset-add <zip路径或目录路径> [--class-remap 旧类名:新类名,...]
```

## 执行步骤

### 1. 验证输入数据
- 检查输入目录/zip 中是否包含 `images/` 和 `labels/`
- 读取其 `dataset.yaml` 确认类别名与主数据集一致
- 若不一致，提示用户是否需要重映射（`--class-remap`）

### 2. 统计现有数据集
```bash
for cls in {0..12}; do
  count=$(grep -rl "^$cls " data/dataset/labels/train/ | wc -l)
  echo "class $cls: $count"
done
```

### 3. 合并数据
- 将新数据图片复制到 `data/dataset/images/` 下
- 将新标注复制到 `data/dataset/labels/` 下
- 若存在类别重映射，批量替换标注文件中的类别 ID

### 4. 重新划分 train/val/test（8:1:1）
```bash
python tools/data_split.py --dataset data/dataset --ratio 0.8 0.1 0.1 --seed 42
```

### 5. 统计合并后各类数量
再次运行统计，对比合并前后的变化，特别关注新增类别（tube/tube-cap/spearhead/pipette）。

### 6. 提示后续步骤
合并完成后提示：
- 运行 `/train-yolo` 触发重训练
- 训练完成后运行 `/val-per-class` 验证效果
