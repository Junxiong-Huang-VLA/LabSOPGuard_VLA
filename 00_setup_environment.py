from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=False)
    return p.returncode, p.stdout


def ensure_conda() -> None:
    if shutil.which("conda") is None:
        raise RuntimeError("conda not found. Install Miniconda/Anaconda first.")


def env_exists(name: str) -> bool:
    code, out = run(["conda", "env", "list", "--json"])
    if code != 0:
        return False
    try:
        envs = json.loads(out).get("envs", [])
        names = [Path(p).name for p in envs]
        return name in names
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Create/reuse conda env for LabSOPGuard")
    parser.add_argument("--project-name", default="LabSOPGuard")
    parser.add_argument("--python-version", default="3.10")
    args = parser.parse_args()

    try:
        ensure_conda()

        if env_exists(args.project_name):
            print(f"[INFO] Reusing existing env: {args.project_name}")
        else:
            print(f"[INFO] Creating env: {args.project_name} (python={args.python_version})")
            code, out = run(["conda", "create", "-y", "-n", args.project_name, f"python={args.python_version}"])
            print(out)
            if code != 0:
                return code

        print("[INFO] Installing via environment.yml ...")
        code, out = run(["conda", "env", "update", "-n", args.project_name, "-f", "environment.yml", "--prune"])
        print(out)
        if code != 0:
            return code

        print("[INFO] Installing via requirements.txt ...")
        code, out = run(["conda", "run", "-n", args.project_name, "python", "-m", "pip", "install", "--upgrade", "pip"])
        print(out)
        if code != 0:
            return code

        code, out = run(["conda", "run", "-n", args.project_name, "python", "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"])
        print(out)
        if code != 0:
            return code

        print("[INFO] Running deep environment check ...")
        code, out = run(["conda", "run", "-n", args.project_name, "python", "14_check_environment.py", "--project-name", args.project_name])
        print(out)
        return code

    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
