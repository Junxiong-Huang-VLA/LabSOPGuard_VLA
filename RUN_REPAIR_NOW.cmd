@echo off
setlocal
cd /d "%~dp0"
set "EXP_ID=6c357850-cdd8-4be4-89c2-a776ad30915f"
set "LABSOPGUARD_FORCE_YOLO_REPAIR=1"
set "LABSOPGUARD_MATERIAL_YOLO_CONF=0.08"
echo [INFO] Project root: %CD%
echo [INFO] Experiment: %EXP_ID%
D:\anaconda\python.exe .\scripts\repair_material_workspace.py --experiment-id %EXP_ID%
if errorlevel 1 (
  echo [FAIL] repair script failed
  pause
  exit /b 1
)
D:\anaconda\python.exe -c "import json,pathlib; e=pathlib.Path('outputs/experiments/6c357850-cdd8-4be4-89c2-a776ad30915f'); p=json.loads((e/'published_materials.json').read_text(encoding='utf-8')); clips=[i.get('published_paths',{}).get('clip') for i in p.get('items',[])]; print('[CHECK] clip_total=',len(clips)); print('[CHECK] webm_total=',sum(1 for c in clips if str(c).lower().endswith('.webm'))); print('[CHECK] first_clip=',clips[0] if clips else None)"
echo [OK] Done. Close launcher, start it again, then press Ctrl+F5 in browser.
pause
