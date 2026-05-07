@echo off
cd /d D:\LabEmbodiedVLA\LabSOPGuard
set PYTHONPATH=src
D:\anaconda\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 >> outputs\run_logs\backend_8000.out.log 2>> outputs\run_logs\backend_8000.err.log
