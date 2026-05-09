@echo off
cd /d D:\LabEmbodiedVLA\LabSOPGuard
D:\anaconda\python.exe tools\emergency_camera_frontend.py --host 127.0.0.1 --port 5173 --backend http://127.0.0.1:8000 >> outputs\run_logs\frontend_5173.out.log 2>> outputs\run_logs\frontend_5173.err.log
