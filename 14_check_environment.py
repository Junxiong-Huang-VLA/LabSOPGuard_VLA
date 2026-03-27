from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REQUIRED_IMPORTS: Dict[str, str] = {
    "numpy": "numpy",
    "opencv-python": "cv2",
    "transformers": "transformers",
    "torch": "torch",
    "reportlab": "reportlab",
    "flask": "flask",
    "openai": "openai",
    "mediapipe": "mediapipe",
}


def safe_run(cmd: List[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=False)
        return p.returncode, p.stdout
    except Exception as exc:
        return 1, str(exc)


def check_python_version() -> dict:
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = major == 3 and (10 <= minor <= 12)
    return {
        "ok": ok,
        "current": sys.version.split()[0],
        "expected": "3.10-3.12",
        "message": "ok" if ok else "python version mismatch",
    }


def check_imports() -> dict:
    result = {"ok": True, "missing": [], "details": {}}
    for pkg, mod in REQUIRED_IMPORTS.items():
        try:
            m = importlib.import_module(mod)
            result["details"][pkg] = {"ok": True, "version": getattr(m, "__version__", "unknown")}
        except Exception as exc:
            result["ok"] = False
            result["missing"].append(pkg)
            result["details"][pkg] = {"ok": False, "error": str(exc)}
    return result


def check_torch_cuda() -> dict:
    info = {
        "ok": True,
        "torch_cuda": "N/A",
        "cuda_available": False,
        "nvidia_smi": "unavailable",
        "message": "",
    }
    try:
        import torch  # type: ignore

        info["torch_cuda"] = str(getattr(torch.version, "cuda", None))
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        return {"ok": False, "message": f"torch import failed: {exc}"}

    if shutil.which("nvidia-smi"):
        code, out = safe_run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if code == 0:
            info["nvidia_smi"] = out.strip()

    if info["nvidia_smi"] != "unavailable" and not info["cuda_available"]:
        info["ok"] = False
        info["message"] = "GPU detected but torch cuda unavailable"
    elif info["torch_cuda"] in ("None", "N/A") and info["nvidia_smi"] != "unavailable":
        info["ok"] = False
        info["message"] = "torch build may be CPU-only while GPU exists"
    else:
        info["message"] = "ok"
    return info


def check_mediapipe_api() -> dict:
    """Check whether mediapipe legacy solutions API is available."""
    info = {
        "ok": True,
        "mediapipe_version": "unknown",
        "has_solutions": False,
        "has_tasks": False,
        "message": "ok",
    }
    try:
        import mediapipe as mp  # type: ignore

        info["mediapipe_version"] = str(getattr(mp, "__version__", "unknown"))
        info["has_solutions"] = bool(hasattr(mp, "solutions"))
        info["has_tasks"] = bool(hasattr(mp, "tasks"))
        if not info["has_solutions"]:
            info["ok"] = False
            info["message"] = (
                "mediapipe.solutions missing; integrated hand_detection will degrade to skip mode. "
                "Recommend: pip install \"mediapipe>=0.10.14,<0.10.20\""
            )
    except Exception as exc:
        info["ok"] = False
        info["message"] = f"mediapipe import failed: {exc}"
    return info


def check_interpreter_alignment(project_name: str) -> dict:
    active = os.environ.get("CONDA_DEFAULT_ENV", "")
    exe = sys.executable
    ok = (active == project_name) or (project_name.lower() in exe.lower())
    return {
        "ok": ok,
        "active_env": active,
        "python_executable": exe,
        "message": "ok" if ok else "interpreter mismatch",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep environment check for LabSOPGuard")
    parser.add_argument("--project-name", default="LabSOPGuard")
    args = parser.parse_args()

    report = {
        "project_name": args.project_name,
        "platform": platform.platform(),
        "python": check_python_version(),
        "imports": check_imports(),
        "torch_cuda": check_torch_cuda(),
        "mediapipe_api": check_mediapipe_api(),
        "interpreter_alignment": check_interpreter_alignment(args.project_name),
    }

    errors = []
    if not report["python"]["ok"]:
        errors.append("python_version")
    if not report["imports"]["ok"]:
        errors.append("imports")
    if not report["torch_cuda"]["ok"]:
        errors.append("torch_cuda")
    if not report["mediapipe_api"]["ok"]:
        errors.append("mediapipe_api")
    if not report["interpreter_alignment"]["ok"]:
        errors.append("interpreter_alignment")

    report["status"] = "pass" if not errors else "fail"
    report["errors"] = errors

    out = Path("outputs/reports/environment_check_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
