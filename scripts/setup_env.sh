#!/usr/bin/env bash
set -euo pipefail
PROJECT_NAME="${1:-LabSOPGuard}"
PY_VER="${2:-3.10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
conda run -n base python 00_setup_environment.py --project-name "${PROJECT_NAME}" --python-version "${PY_VER}"
echo "[DONE] Environment ready: ${PROJECT_NAME}"
