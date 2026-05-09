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

    required = {"backend", "prometheus", "wireless-video-sender", "wireless-video-receiver"}
    missing = sorted(required - set(services))
    if missing:
        errors.append(f"{path}: missing services: {', '.join(missing)}")

    sender = services.get("wireless-video-sender", {})
    receiver = services.get("wireless-video-receiver", {})
    for name, service, port, env_name in (
        ("wireless-video-sender", sender, "9301", "WIRELESS_VIDEO_SENDER_METRICS_PORT=9301"),
        ("wireless-video-receiver", receiver, "9302", "WIRELESS_VIDEO_RECEIVER_METRICS_PORT=9302"),
    ):
        if service.get("profiles") != ["wireless-video"]:
            errors.append(f"{path}: {name} must use wireless-video profile")
        if env_name not in _service_env(service):
            errors.append(f"{path}: {name} missing {env_name}")
        if f"{port}:{port}" not in [str(item) for item in service.get("ports", [])]:
            errors.append(f"{path}: {name} missing {port}:{port} port mapping")
        if f"{port}/metrics" not in _healthcheck_text(service):
            errors.append(f"{path}: {name} healthcheck must probe :{port}/metrics")
        if service.get("restart") != "unless-stopped":
            errors.append(f"{path}: {name} should restart unless-stopped")
        logging_opts = service.get("logging", {}).get("options", {})
        if logging_opts.get("max-size") != "20m" or logging_opts.get("max-file") != "3":
            errors.append(f"{path}: {name} should cap json-file logs at 20m x3")

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
    for job_name, target in (
        ("lab_sop_guard_wireless_video_sender", "host.docker.internal:9301"),
        ("lab_sop_guard_wireless_video_receiver", "host.docker.internal:9302"),
    ):
        job = jobs.get(job_name)
        targets = []
        if isinstance(job, dict):
            for config in job.get("static_configs") or []:
                targets.extend(str(item) for item in config.get("targets") or [])
        if target not in targets:
            errors.append(f"{prometheus_path}: {job_name} must scrape {target}")

    alerts = _load_yaml(alerts_path)
    groups = alerts.get("groups") if isinstance(alerts, dict) else None
    if not isinstance(groups, list) or not groups:
        return errors + [f"{alerts_path}: missing alert groups"]

    expected_alerts = {
        "LabSOPGuardExternalCircuitOpen",
        "LabSOPGuardWirelessVideoPacketLoss",
        "LabSOPGuardWirelessVideoNotRunning",
        "LabSOPGuardCaptureDiskLow",
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
    state_rule = seen.get("LabSOPGuardWirelessVideoNotRunning", {})
    if "== 1" not in str(state_rule.get("expr", "")):
        errors.append(f"{alerts_path}: wireless-video state alert must compare one-hot state to 1")
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
    parser = argparse.ArgumentParser(description="Validate LabSOPGuard runtime compose and monitoring config.")
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
