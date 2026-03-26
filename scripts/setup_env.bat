@echo off
set PROJECT_NAME=%~1
if "%PROJECT_NAME%"=="" set PROJECT_NAME=LabSOPGuard
set PY_VER=%~2
if "%PY_VER%"=="" set PY_VER=3.10

pushd %~dp0..
conda run -n base python 00_setup_environment.py --project-name %PROJECT_NAME% --python-version %PY_VER%
if errorlevel 1 (
  echo [ERROR] Environment setup failed.
  popd
  exit /b 1
)
echo [DONE] Environment ready: %PROJECT_NAME%
popd
