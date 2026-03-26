# Quickstart

## 1) Setup environment

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -ProjectName LabEmbodiedVLA -PythonVersion 3.10
```

## 2) Validate environment

```bash
conda run -n LabEmbodiedVLA python 14_check_environment.py --project-name LabEmbodiedVLA
```

## 3) Prepare and validate data

```bash
conda run -n LabEmbodiedVLA python scripts/data_check.py --config configs/data/dataset.yaml
conda run -n LabEmbodiedVLA python scripts/data_split.py --config configs/data/dataset.yaml
```

## 4) Training and inference

```bash
conda run -n LabEmbodiedVLA python scripts/train.py --config configs/model/vla_model.yaml
conda run -n LabEmbodiedVLA python scripts/infer.py --video data/raw/videos/session_001.mp4
```

## 5) Batch export and evaluation

```bash
conda run -n LabEmbodiedVLA python scripts/export_results.py --split data/splits/test.jsonl
conda run -n LabEmbodiedVLA python scripts/evaluate.py --pred outputs/predictions/test_predictions.jsonl
```

## 6) Runtime monitoring with report

```bash
conda run -n LabEmbodiedVLA python scripts/run_monitor.py --video data/raw/videos/session_001.mp4 --session-id session_001
```
