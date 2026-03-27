# Environment Guide (LabSOPGuard)

## 环境管理强约束

- 项目仅使用一个虚拟环境
- 环境名称固定为 `LabSOPGuard`
- 如已存在同名环境，直接复用
- 默认采用 conda

## 1) 创建/复用环境

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1 -ProjectName LabSOPGuard -PythonVersion 3.10
```

Windows CMD:

```bat
scripts\setup_env.bat LabSOPGuard 3.10
```

Linux/macOS:

```bash
bash scripts/setup_env.sh LabSOPGuard 3.10
```

## 2) 激活环境

```bash
conda activate LabSOPGuard
```

## 3) 环境检查

```bash
conda run -n LabSOPGuard python 14_check_environment.py --project-name LabSOPGuard
```

检查项：

- Python 版本
- torch/cuda
- transformers
- opencv
- numpy
- reportlab
- 解释器与环境一致性

## 4) 重建环境

```bash
conda env remove -n LabSOPGuard
conda run -n base python 00_setup_environment.py --project-name LabSOPGuard --python-version 3.10
```

## 5) 常见问题建议

- 安装失败：先升级 pip，再 `--no-cache-dir`
- CUDA 不可用：检查驱动与 torch 构建匹配
- 导入失败：确认命令执行在 `LabSOPGuard` 环境内

## 6) MediaPipe 兼容修复（`module 'mediapipe' has no attribute 'solutions'`）

如果集成页面报错：

`module 'mediapipe' has no attribute 'solutions'`

执行一键修复脚本（会在 `LabSOPGuard` 环境内重装兼容版本）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fix_mediapipe_compat.ps1 -EnvName LabSOPGuard
```

修复后重启本地预览：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop_preview.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start_preview.ps1
```
