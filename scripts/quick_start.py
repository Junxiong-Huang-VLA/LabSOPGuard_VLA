"""
快速启动脚本 - 实验室SOP合规监控系统
跳过模型训练，直接使用预训练模型启动系统
"""
import os
import sys
import time
import subprocess
from pathlib import Path

def main():
    """快速启动系统"""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print("实验室SOP合规监控系统 - 快速启动")
    print("=" * 50)

    # 检查必要文件
    required_files = [
        "backend/main.py",
        "frontend/dashboard.py",
        "requirements.txt"
    ]

    for file_path in required_files:
        if not (project_root / file_path).exists():
            print(f"错误: 缺少必要文件 {file_path}")
            return False

    print("检查依赖模块...")
    try:
        import fastapi
        import uvicorn
        import nicegui
        import cv2
        import numpy
        print("所有依赖模块检查通过")
    except ImportError as e:
        print(f"依赖模块缺失: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

    print("\n启动服务...")

    # 启动Redis（如果可用）
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("Redis服务已连接")
    except:
        print("Redis服务不可用，部分功能可能受限")

    # 启动后端服务
    print("启动FastAPI后端服务...")
    backend_process = subprocess.Popen(
        [sys.executable, "backend/main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # 等待后端启动
    time.sleep(3)

    # 启动前端服务
    print("启动NiceGUI前端服务...")
    frontend_process = subprocess.Popen(
        [sys.executable, "frontend/dashboard.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print("\n服务启动完成!")
    print("后端API: http://localhost:8000")
    print("前端Dashboard: http://localhost:8080")
    print("API文档: http://localhost:8000/docs")
    print("\n按Ctrl+C停止所有服务")

    try:
        # 等待用户中断
        while True:
            time.sleep(1)
            # 检查进程状态
            if backend_process.poll() is not None:
                print("后端服务异常退出")
                break
            if frontend_process.poll() is not None:
                print("前端服务异常退出")
                break
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        backend_process.terminate()
        frontend_process.terminate()
        print("服务已停止")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)