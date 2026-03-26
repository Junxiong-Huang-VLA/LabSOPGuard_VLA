from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys

REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "yaml",
    "matplotlib",
    "PIL",
    "cv2",
    "torch",
    "transformers",
    "accelerate",
    "reportlab",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick environment check for LabEmbodiedVLA")
    parser.add_argument("--project-name", default="LabEmbodiedVLA")
    args = parser.parse_args()

    print(f"[INFO] Project: {args.project_name}")
    print(f"[INFO] Python: {sys.version.split()[0]}")
    print(f"[INFO] Platform: {platform.platform()}")

    missing = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(module)

    cuda = {"cuda_available": False, "torch_cuda": "N/A", "nvidia_smi": "N/A"}
    try:
        import torch  # type: ignore

        cuda["cuda_available"] = bool(torch.cuda.is_available())
        cuda["torch_cuda"] = str(getattr(torch.version, "cuda", None))
    except Exception:
        pass

    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True)
        cuda["nvidia_smi"] = out.strip()
    except Exception:
        cuda["nvidia_smi"] = "unavailable"

    print(json.dumps({"missing": missing, "cuda": cuda}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
