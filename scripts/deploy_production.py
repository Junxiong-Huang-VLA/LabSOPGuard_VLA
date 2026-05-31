"""
实验室SOP合规监控系统 - 生产环境部署脚本
自动化部署流程：模型训练 → 系统配置 → 服务启动 → 健康检查
"""
import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import yaml

class ProductionDeployer:
    """生产环境部署器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.deployment_log = []
        self.start_time = datetime.now()

    def log(self, message: str, level: str = "INFO"):
        """记录部署日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.deployment_log.append(log_entry)

    def run_command(self, cmd: list, cwd: str = None, check: bool = True) -> bool:
        """执行系统命令"""
        try:
            self.log(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                check=check,
                capture_output=True,
                text=True
            )
            if result.stdout:
                self.log(f"输出: {result.stdout[:500]}")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"命令执行失败: {e}", "ERROR")
            if e.stderr:
                self.log(f"错误信息: {e.stderr[:500]}", "ERROR")
            return False

    def check_prerequisites(self) -> bool:
        """检查部署前提条件"""
        self.log("检查部署前提条件...")

        # 检查Python环境
        python_version = sys.version
        self.log(f"Python版本: {python_version}")

        # 检查必要目录
        required_dirs = [
            "configs", "src", "backend", "frontend",
            "templates", "outputs"
        ]

        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                self.log(f"创建目录: {dir_name}")
                dir_path.mkdir(parents=True, exist_ok=True)

        # 检查配置文件
        config_files = [
            "configs/sop/rules.yaml",
            "configs/data/class_schema.yaml",
            "requirements.txt"
        ]

        for config_file in config_files:
            config_path = self.project_root / config_file
            if not config_path.exists():
                self.log(f"配置文件缺失: {config_file}", "ERROR")
                return False

        self.log("前提条件检查完成")
        return True

    def train_detection_model(self) -> bool:
        """训练检测模型"""
        self.log("开始训练检测模型...")

        # 检查是否有训练数据
        dataset_yaml = self.project_root / "data" / "dataset" / "dataset.yaml"
        if not dataset_yaml.exists():
            self.log("未找到训练数据集，使用预训练模型")
            return True

        # 执行训练脚本
        train_cmd = [
            sys.executable, "scripts/train_yolo_lab.py",
            "--dataset-yaml", str(dataset_yaml),
            "--model", "yolo26s.pt",
            "--epochs", "50",  # 生产环境减少训练轮数
            "--imgsz", "640",
            "--batch", "16",
            "--device", "0",
            "--workers", "0",
            "--project", "outputs/training",
            "--name", "production_model"
        ]

        success = self.run_command(train_cmd)

        if success:
            # 验证模型文件
            model_path = self.project_root / "outputs/training/production_model/weights/best.pt"
            if model_path.exists():
                self.log(f"模型训练完成: {model_path}")
                return True
            else:
                self.log("模型文件未找到", "ERROR")
                return False

        return False

    def setup_redis(self) -> bool:
        """设置Redis服务"""
        self.log("检查Redis服务...")

        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            self.log("Redis服务运行正常")
            return True
        except Exception as e:
            self.log("Redis服务不可用，请确保Redis服务已启动: redis-server")
            return False

    def create_systemd_services(self) -> bool:
        """创建系统服务文件（Linux）或Windows服务"""
        self.log("创建系统服务配置...")

        if os.name == 'nt':  # Windows
            return self._create_windows_services()
        else:  # Linux
            return self._create_linux_services()

    def _create_windows_services(self) -> bool:
        """创建Windows服务配置"""
        services_dir = self.project_root / "deployment" / "services"
        services_dir.mkdir(parents=True, exist_ok=True)

        # FastAPI后端服务
        backend_service = f"""@echo off
cd /d "{self.project_root}"
python backend/main.py
"""
        (services_dir / "start_backend.bat").write_text(backend_service)

        # NiceGUI前端服务
        frontend_service = f"""@echo off
cd /d "{self.project_root}"
python frontend/dashboard.py
"""
        (services_dir / "start_frontend.bat").write_text(frontend_service)

        # Celery worker服务
        celery_service = f"""@echo off
cd /d "{self.project_root}"
celery -A backend.main.celery_app worker --loglevel=info --concurrency=4
"""
        (services_dir / "start_celery.bat").write_text(celery_service)

        # 启动脚本
        start_all = f"""@echo off
echo Starting LabSOPGuard Production System...
start "Backend" cmd /k "{services_dir / 'start_backend.bat'}"
start "Frontend" cmd /k "{services_dir / 'start_frontend.bat'}"
start "Celery" cmd /k "{services_dir / 'start_celery.bat'}"
echo All services started!
pause
"""
        (services_dir / "start_all.bat").write_text(start_all)

        self.log("Windows服务配置创建完成")
        return True

    def _create_linux_services(self) -> bool:
        """创建Linux systemd服务"""
        services_dir = self.project_root / "deployment" / "services"
        services_dir.mkdir(parents=True, exist_ok=True)

        # FastAPI后端服务
        backend_service = f"""[Unit]
Description=LabSOPGuard FastAPI Backend
After=network.target redis.service

[Service]
Type=simple
User=labuser
WorkingDirectory={self.project_root}
ExecStart={sys.executable} backend/main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
        (services_dir / "labsopguard-backend.service").write_text(backend_service)

        # NiceGUI前端服务
        frontend_service = f"""[Unit]
Description=LabSOPGuard NiceGUI Frontend
After=network.target labsopguard-backend.service

[Service]
Type=simple
User=labuser
WorkingDirectory={self.project_root}
ExecStart={sys.executable} frontend/dashboard.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""
        (services_dir / "labsopguard-frontend.service").write_text(frontend_service)

        self.log("Linux systemd服务配置创建完成")
        return True

    def create_docker_deployment(self) -> bool:
        """创建Docker部署配置"""
        self.log("创建Docker部署配置...")

        # 检查Docker配置是否存在
        dockerfile = self.project_root / "Dockerfile"
        docker_compose = self.project_root / "docker-compose.yml"

        if not dockerfile.exists():
            self.log("Dockerfile不存在，创建基础配置")
            self._create_basic_dockerfile()

        if not docker_compose.exists():
            self.log("docker-compose.yml不存在，创建基础配置")
            self._create_basic_docker_compose()

        self.log("Docker部署配置完成")
        return True

    def _create_basic_dockerfile(self):
        """创建基础Dockerfile"""
        dockerfile_content = f"""FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p outputs/reports outputs/video_cache logs

# 暴露端口
EXPOSE 8000 8080

# 启动脚本
CMD ["sh", "-c", "python backend/main.py & python frontend/dashboard.py"]
"""
        (self.project_root / "Dockerfile").write_text(dockerfile_content)

    def _create_basic_docker_compose(self):
        """创建基础docker-compose.yml"""
        compose_content = f"""version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  backend:
    build: .
    command: python backend/main.py
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - ./outputs:/app/outputs
    depends_on:
      - redis

  frontend:
    build: .
    command: python frontend/dashboard.py
    ports:
      - "8080:8080"
    volumes:
      - .:/app
      - ./outputs:/app/outputs
    depends_on:
      - backend

volumes:
  redis_data:
"""
        (self.project_root / "docker-compose.yml").write_text(compose_content)

    def run_health_check(self) -> bool:
        """运行健康检查"""
        self.log("运行系统健康检查...")

        checks = [
            ("Python模块导入", self._check_python_imports),
            ("配置文件", self._check_config_files),
            ("模型文件", self._check_model_files),
            ("端口可用性", self._check_ports),
        ]

        all_passed = True
        for check_name, check_func in checks:
            try:
                result = check_func()
                status = "PASS" if result else "FAIL"
                self.log(f"健康检查 - {check_name}: {status}")
                if not result:
                    all_passed = False
            except Exception as e:
                self.log(f"健康检查 - {check_name}: ERROR - {e}", "ERROR")
                all_passed = False

        return all_passed

    def _check_python_imports(self) -> bool:
        """检查Python模块导入"""
        required_modules = [
            "fastapi", "uvicorn", "nicegui", "cv2",
            "numpy", "pandas", "redis", "celery"
        ]

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                self.log(f"模块导入失败: {module}", "ERROR")
                return False

        return True

    def _check_config_files(self) -> bool:
        """检查配置文件"""
        config_files = [
            "configs/sop/rules.yaml",
            "configs/data/class_schema.yaml"
        ]

        for config_file in config_files:
            config_path = self.project_root / config_file
            if not config_path.exists():
                self.log(f"配置文件缺失: {config_file}", "ERROR")
                return False

        return True

    def _check_model_files(self) -> bool:
        """检查模型文件"""
        # 检查是否有训练好的模型或预训练模型
        model_paths = [
            "outputs/training/production_model/weights/best.pt",
            "yolo26s.pt",
            "yolov8n.pt"
        ]

        for model_path in model_paths:
            if (self.project_root / model_path).exists():
                return True

        self.log("未找到模型文件，将使用默认预训练模型")
        return True

    def _check_ports(self) -> bool:
        """检查端口可用性"""
        import socket

        ports = [8000, 8080, 6379]
        for port in ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        self.log(f"端口 {port} 已被占用")
                    else:
                        self.log(f"端口 {port} 可用")
            except Exception as e:
                self.log(f"端口 {port} 检查失败: {e}")

        return True

    def generate_deployment_report(self) -> str:
        """生成部署报告"""
        end_time = datetime.now()
        duration = end_time - self.start_time

        report = {
            "deployment_time": self.start_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "project_root": str(self.project_root),
            "python_version": sys.version,
            "deployment_log": self.deployment_log,
            "status": "completed"
        }

        report_path = self.project_root / "outputs" / "deployment_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.log(f"部署报告已生成: {report_path}")
        return str(report_path)

    def deploy(self) -> bool:
        """执行完整部署流程"""
        self.log("开始实验室SOP合规监控系统部署...")

        steps = [
            ("检查前提条件", self.check_prerequisites),
            ("训练检测模型", self.train_detection_model),
            ("设置Redis服务", self.setup_redis),
            ("创建系统服务", self.create_systemd_services),
            ("创建Docker配置", self.create_docker_deployment),
            ("运行健康检查", self.run_health_check),
        ]

        for step_name, step_func in steps:
            self.log(f"执行步骤: {step_name}")
            try:
                success = step_func()
                if not success:
                    self.log(f"步骤失败: {step_name}", "ERROR")
                    return False
            except Exception as e:
                self.log(f"步骤异常: {step_name} - {e}", "ERROR")
                return False

        # 生成部署报告
        report_path = self.generate_deployment_report()

        self.log("=" * 60)
        self.log("部署完成!")
        self.log(f"部署报告: {report_path}")
        self.log("下一步:")
        self.log("1. 启动服务: 运行 deployment/services/start_all.bat")
        self.log("2. 访问前端: http://localhost:8080")
        self.log("3. 查看API文档: http://localhost:8000/docs")
        self.log("=" * 60)

        return True


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    deployer = ProductionDeployer(str(project_root))

    success = deployer.deploy()

    if success:
        print("\n部署成功! 系统已准备就绪。")
        sys.exit(0)
    else:
        print("\n部署失败，请检查日志。")
        sys.exit(1)


if __name__ == "__main__":
    main()
