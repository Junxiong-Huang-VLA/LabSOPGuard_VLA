param(
    [string]$PythonExe = "C:\Users\Win10\miniconda3\envs\LabSOPGuard\python.exe",
    [string]$DatasetYaml = "data\dataset\dataset.yaml",
    [string]$Model = "yolo26s.pt",
    [int]$Epochs = 200,
    [int]$ImgSize = 640,
    [int]$Batch = 16,
    [string]$Device = "0",
    [string]$RunName = "yolo26n_allphotos_8_2_e20",
    [switch]$RebuildSplit
)

$scriptDir = Split-Path $MyInvocation.MyCommand.Path -Parent
$projectRoot = Split-Path $scriptDir -Parent
Set-Location $projectRoot
$ErrorActionPreference = "Stop"

Write-Host "[INFO] Project root: $PWD"

if ($RebuildSplit) {
    Write-Host "[INFO] Rebuilding standardized dataset split from data\interim\frames ..."
    & $PythonExe .\scripts\build_dataset_standardized.py `
        --src-root data\interim\frames `
        --class-yaml configs\data\class_schema.yaml `
        --out-root data\dataset `
        --train-ratio 0.8 `
        --val-ratio 0.0 `
        --test-ratio 0.2 `
        --clean
}

Write-Host "[INFO] Training ..."
& $PythonExe .\scripts\train_yolo_lab.py `
    --dataset-yaml $DatasetYaml `
    --model $Model `
    --epochs $Epochs `
    --imgsz $ImgSize `
    --batch $Batch `
    --device $Device `
    --workers 0 `
    --project outputs\training `
    --name $RunName


    $weights = "outputs/training/$RunName/weights/best.pt"
if (!(Test-Path $weights)) {
    throw "best.pt not found: $weights"
}

Write-Host "[INFO] Running inference on test split ..."
$env:YOLO_CONFIG_DIR = (Resolve-Path ".\.ultralytics").Path
& $PythonExe -c "import yaml;from pathlib import Path;from ultralytics import YOLO; ds=yaml.safe_load(open('$DatasetYaml',encoding='utf-8')); root=Path(ds['path']); test_rel=ds.get('test') or ds.get('val') or 'images/test'; source=str((root/test_rel).resolve()); m=YOLO('$weights'); m.predict(source=source, imgsz=$ImgSize, conf=0.25, device='$Device', save=True, save_txt=True, project='outputs/inference', name='${RunName}_test', exist_ok=True, verbose=False)"

Write-Host "[INFO] Reading final metrics ..."
& $PythonExe -c "import csv;rows=list(csv.DictReader(open('outputs/training/$RunName/results.csv',encoding='utf-8')));r=rows[-1];print('precision',r['metrics/precision(B)']);print('recall',r['metrics/recall(B)']);print('mAP50',r['metrics/mAP50(B)']);print('mAP50-95',r['metrics/mAP50-95(B)'])"

Write-Host "[DONE] Weights: outputs/training/$RunName/weights/best.pt"
Write-Host "[DONE] Inference: outputs/inference/${RunName}_test"
