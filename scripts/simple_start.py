"""
简化版启动脚本 - 直接启动服务
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("实验室SOP合规监控系统 - 简化启动")
print("=" * 50)

# 检查依赖
try:
    import fastapi
    import uvicorn
    import nicegui
    import cv2
    import numpy
    print("所有依赖模块检查通过")
except ImportError as e:
    print(f"依赖模块缺失: {e}")
    sys.exit(1)

print("\n启动FastAPI后端服务...")
print("后端API地址: http://localhost:8000")
print("API文档地址: http://localhost:8000/docs")

# 直接启动uvicorn服务器
if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(
            "backend.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except Exception as e:
        print(f"启动失败: {e}")
        print("尝试直接导入模块...")

        # 备用方案：直接导入并运行
        try:
            from backend.main import app
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except Exception as e2:
            print(f"备用方案也失败: {e2}")
            sys.exit(1)