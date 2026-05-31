from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_runtime_configs import validate_runtime_configs


pytestmark = pytest.mark.unit


def test_runtime_compose_and_monitoring_configs_are_valid():
    assert validate_runtime_configs(PROJECT_ROOT) == []


def test_ci_runs_docker_compose_and_promtool_runtime_checks():
    ci_text = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "docker compose -f docker-compose.yml --profile monitoring config --quiet" in ci_text
    moved_profile = "--profile " + "-".join(["wireless", "video"])
    assert moved_profile not in ci_text
    assert "python scripts/check_project_scope.py" in ci_text
    assert "--entrypoint promtool" in ci_text
    assert "check config /etc/prometheus/prometheus.yml" in ci_text
    assert "prom/prometheus:latest" in ci_text
