from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _service_env(service: dict[str, Any]) -> set[str]:
    raw = service.get("environment") or []
    if isinstance(raw, dict):
        return {f"{key}={value}" for key, value in raw.items()}
    return {str(item) for item in raw}


def _healthcheck_text(service: dict[str, Any]) -> str:
    test = service.get("healthcheck", {}).get("test", [])
    if isinstance(test, str):
        return test
    return " ".join(str(item) for item in test)


def validate_compose(path: Path) -> list[str]:
    errors: list[str] = []
    compose = _load_yaml(path)
    services = compose.get("services") if isinstance(compose, dict) else None
    if not isinstance(services, dict):
        return [f"{path}: missing services map"]

    required = {"backend", "prometheus"}
    missing = sorted(required - set(services))
    if missing:
        errors.append(f"{path}: missing services: {', '.join(missing)}")

    forbidden_services = {"wireless-video-sender", "wireless-video-receiver", "camera-proxy", "ptz-tracker"}
    leaked = sorted(forbidden_services & set(services))
    if leaked:
        errors.append(f"{path}: out-of-scope services belong in sibling projects: {', '.join(leaked)}")

    prometheus = services.get("prometheus", {})
    volumes = [str(item) for item in prometheus.get("volumes", [])]
    if not any("prometheus-alerts.yml" in volume for volume in volumes):
        errors.append(f"{path}: prometheus must mount monitoring/prometheus-alerts.yml")
    if "host.docker.internal:host-gateway" not in [str(item) for item in prometheus.get("extra_hosts", [])]:
        errors.append(f"{path}: prometheus must map host.docker.internal")
    return errors


def validate_prometheus(prometheus_path: Path, alerts_path: Path) -> list[str]:
    errors: list[str] = []
    prometheus = _load_yaml(prometheus_path)
    rule_files = [str(item) for item in prometheus.get("rule_files", [])]
    if not any("prometheus-alerts.yml" in item for item in rule_files):
        errors.append(f"{prometheus_path}: missing prometheus-alerts.yml rule file")

    scrape_configs = prometheus.get("scrape_configs") or []
    jobs = {item.get("job_name"): item for item in scrape_configs if isinstance(item, dict)}
    forbidden_jobs = sorted(name for name in jobs if "wireless_video" in str(name) or "ptz" in str(name) or "camera_proxy" in str(name))
    if forbidden_jobs:
        errors.append(f"{prometheus_path}: out-of-scope scrape jobs present: {', '.join(forbidden_jobs)}")

    alerts = _load_yaml(alerts_path)
    groups = alerts.get("groups") if isinstance(alerts, dict) else None
    if not isinstance(groups, list) or not groups:
        return errors + [f"{alerts_path}: missing alert groups"]

    expected_alerts = {
        "LabSOPGuardExternalCircuitOpen",
        "LabSOPGuardCaptureDiskLow",
        "LabSOPGuardMaterialBrokenClips",
        "LabSOPGuardVLMCircuitOpen",
    }
    seen: dict[str, dict[str, Any]] = {}
    for group in groups:
        for rule in group.get("rules") or []:
            name = rule.get("alert")
            if not name:
                errors.append(f"{alerts_path}: alert rule missing alert name")
                continue
            seen[str(name)] = rule
            for key in ("expr", "for", "labels", "annotations"):
                if key not in rule:
                    errors.append(f"{alerts_path}: {name} missing {key}")
            labels = rule.get("labels") or {}
            if labels.get("severity") not in {"warning", "critical"}:
                errors.append(f"{alerts_path}: {name} missing warning/critical severity")
            annotations = rule.get("annotations") or {}
            if not annotations.get("summary") or not annotations.get("description"):
                errors.append(f"{alerts_path}: {name} missing summary/description")

    missing_alerts = sorted(expected_alerts - set(seen))
    if missing_alerts:
        errors.append(f"{alerts_path}: missing expected alerts: {', '.join(missing_alerts)}")
    forbidden_alerts = sorted(name for name in seen if "WirelessVideo" in name or "PTZ" in name or "CameraProxy" in name)
    if forbidden_alerts:
        errors.append(f"{alerts_path}: out-of-scope alerts present: {', '.join(forbidden_alerts)}")
    return errors


def validate_runtime_configs(project_root: Path = PROJECT_ROOT) -> list[str]:
    return [
        *validate_compose(project_root / "docker-compose.yml"),
        *validate_prometheus(
            project_root / "monitoring" / "prometheus.yml",
            project_root / "monitoring" / "prometheus-alerts.yml",
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LabEmbodied runtime compose and monitoring config.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    errors = validate_runtime_configs(args.project_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("runtime config ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
