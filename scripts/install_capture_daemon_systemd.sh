#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${1:-labsopguard-capture.service}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python3}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/runtime/multicam_soak.yaml}"
UNIT_PATH="/etc/systemd/system/$SERVICE_NAME"

sudo tee "$UNIT_PATH" >/dev/null <<EOF
[Unit]
Description=LabSOPGuard supervised multi-camera capture daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO_ROOT
ExecStart=$PYTHON_EXE $REPO_ROOT/scripts/run_capture_daemon.py --config $CONFIG_PATH
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
echo "Installed systemd service: $SERVICE_NAME"
