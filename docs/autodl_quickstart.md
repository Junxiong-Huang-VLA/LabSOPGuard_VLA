# AutoDL Quickstart

这份文档只覆盖一个目标：把当前仓库和当前标准数据集放到 AutoDL 上，然后重新跑一遍 YOLO 训练。

## 当前前提

本地已经整理好的标准数据集位置：

- `data/dataset/images/train`
- `data/dataset/images/val`
- `data/dataset/images/test`
- `data/dataset/labels/train`
- `data/dataset/labels/val`
- `data/dataset/labels/test`
- `data/dataset/dataset.yaml`

当前 split：

- train: 2383
- val: 297
- test: 299

## 推荐的 AutoDL 目录

假设 AutoDL 工作目录为：

```bash
/root/autodl-tmp/LabSOPGuard
```

本文命令统一按这个路径写。你也可以换成自己的路径，但后续命令要一起改。

## 1. 把仓库传到 AutoDL

### 方式 A：直接上传整个项目目录

在本地 Windows PowerShell：

```powershell
scp -r D:\LabEmbodiedVLA\LabSOPGuard root@<AUTODL_HOST>:/root/autodl-tmp/
```

### 方式 B：AutoDL 上 git clone，再单独传 dataset

如果仓库代码已经在远端：

```bash
git clone <your_repo_url> /root/autodl-tmp/LabSOPGuard
```

然后从本地传 dataset：

```powershell
scp -r D:\LabEmbodiedVLA\LabSOPGuard\data\dataset root@<AUTODL_HOST>:/root/autodl-tmp/LabSOPGuard/data/
```

## 2. 登录 AutoDL

```bash
ssh root@<AUTODL_HOST>
```

## 3. 创建 Python 环境

如果实例里已经有 conda，推荐这样：

```bash
conda create -n LabSOPGuard python=3.10 -y
conda activate LabSOPGuard
```

如果没有 conda，就用系统 python 或 venv。

## 4. 进入项目目录

```bash
cd /root/autodl-tmp/LabSOPGuard
```

## 5. 安装依赖

```bash
pip install -U pip
pip install -r requirements.txt
```

## 6. 检查数据集是否就位

```bash
python - <<'PY'
from pathlib import Path
root = Path('/root/autodl-tmp/LabSOPGuard/data/dataset')
for rel in [
    'images/train', 'images/val', 'images/test',
    'labels/train', 'labels/val', 'labels/test',
    'dataset.yaml',
]:
    p = root / rel if rel != 'dataset.yaml' else root / 'dataset.yaml'
    print(rel, p.exists())
PY
```

## 7. 最短训练命令

```bash
python scripts/train_yolo_lab.py \
  --dataset-yaml data/dataset/dataset.yaml \
  --model yolo26s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --project outputs/training \
  --name yolo26s_autodl_8_1_1
```

## 8. 用一键脚本跑

仓库里已提供：

- `scripts/autodl_train_yolo.sh`

执行：

```bash
chmod +x scripts/autodl_train_yolo.sh
PROJECT_ROOT=/root/autodl-tmp/LabSOPGuard \
PYTHON_BIN=python \
DATASET_YAML=data/dataset/dataset.yaml \
MODEL=yolo26s.pt \
EPOCHS=100 \
IMGSZ=640 \
BATCH=16 \
DEVICE=0 \
WORKERS=8 \
RUN_NAME=yolo26s_autodl_8_1_1 \
./scripts/autodl_train_yolo.sh
```

## 9. 训练完成后看哪里

输出目录：

```bash
/root/autodl-tmp/LabSOPGuard/outputs/training/yolo26s_autodl_8_1_1
```

重点文件：

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `train_run_meta.json`

## 10. 拉回本地

在本地 Windows PowerShell：

```powershell
scp -r root@<AUTODL_HOST>:/root/autodl-tmp/LabSOPGuard/outputs/training/yolo26s_autodl_8_1_1 D:\LabEmbodiedVLA\LabSOPGuard\outputs\training\
```

## 11. 当前建议参数

如果你只是先重跑一遍，先用下面这组：

- model: `yolo26s.pt`
- epochs: `100`
- imgsz: `640`
- batch: `16`
- device: `0`
- workers: `8`

如果显存吃紧：

- 把 `batch` 改成 `8`
- 或把 `imgsz` 改成 `512`

## 12. 最小排障

### 数据集找不到

检查：

```bash
ls data/dataset
cat data/dataset/dataset.yaml
```

### CUDA 不可用

检查：

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

### 爆显存

把训练参数降到：

```bash
--imgsz 512 --batch 8
```

## 13. 这次不要再做的事

- 不要再用旧的 `val=test` 配置
- 不要再用旧的 `data/dataset/build_report.json` 和 `source_to_dataset_mapping.csv`
- 不要先跑旧 demo，再怀疑主训练链路

当前只认：

- `data/dataset/dataset.yaml`
- `data/dataset/images/{train,val,test}`
- `data/dataset/labels/{train,val,test}`
