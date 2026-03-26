#!/usr/bin/env bash
set -euo pipefail
PROJECT_NAME="${1:-LabSOPGuard}"
PYTHON_VERSION="${2:-3.10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/setup_env.sh" "${PROJECT_NAME}" "${PYTHON_VERSION}"
