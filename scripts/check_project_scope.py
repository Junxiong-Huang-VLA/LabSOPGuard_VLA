from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCAN_TARGETS = [
    ".github",
    "backend",
    "config",
    "configs",
    "docker-compose.yml",
    "frontend/src",
    "monitoring",
    "scripts",
    "src/experiment",
    "src/labsopguard",
    "tests",
]

FORBIDDEN = {
    "ptz": "PTZ tracking/control lives in D:\\PtzTracker",
    "ptz_tracker": "PTZ tracking/control lives in D:\\PtzTracker",
    "pan_tilt": "PTZ tracking/control lives in D:\\PtzTracker",
    "/api/v1/cameras": "camera-control APIs live outside LabSOPGuard",
    "wireless-video": "wireless video transport lives in D:\\MultiCameraMonitor",
    "wireless_video": "wireless video transport lives in D:\\MultiCameraMonitor",
    "backend.wireless_video": "wireless video transport lives in D:\\MultiCameraMonitor",
    "wvd_sdk": "wireless video SDKs live in D:\\MultiCameraMonitor",
    "camera_proxy": "camera proxy services live in D:\\MultiCameraMonitor",
    "camera_streaming": "camera streaming services live in D:\\MultiCameraMonitor",
    "usb_camera_worker": "USB capture workers live in D:\\MultiCameraMonitor",
    "multi_monitor": "multi-monitor orchestration lives in D:\\MultiCameraMonitor",
    "multi-monitor": "multi-monitor orchestration lives in D:\\MultiCameraMonitor",
    "multicam_soak": "multi-camera soak testing lives in D:\\MultiCameraMonitor",
    "capture_daemon": "multi-camera capture daemons live in D:\\MultiCameraMonitor",
}

ALLOWED_SUBSTRINGS = {
    "camera_id",
    "camera_view",
    "camera_intrinsics",
    "camera_xyz",
    "camera_to_robot",
    "split_yolo_by_group",
}

ALLOWLIST_FILES = {
    Path("scripts/check_project_scope.py"),
    Path("scripts/check_runtime_configs.py"),
}


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for target in SCAN_TARGETS:
        path = root / target
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        for child in path.rglob("*"):
            if child.is_file() and child.suffix.lower() not in {
                ".dll",
                ".exe",
                ".gif",
                ".jpeg",
                ".jpg",
                ".log",
                ".mp4",
                ".png",
                ".pyc",
                ".sqlite",
                ".zip",
            }:
                files.append(child)
    return files


def check_project_scope(root: Path = PROJECT_ROOT) -> list[str]:
    errors: list[str] = []
    for path in _iter_files(root):
        rel = path.relative_to(root)
        if rel in ALLOWLIST_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lowered = text.lower()
        for pattern, reason in FORBIDDEN.items():
            if pattern in ALLOWED_SUBSTRINGS:
                continue
            if pattern in lowered:
                errors.append(f"{rel}: forbidden '{pattern}' found; {reason}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep LabEmbodied focused on dual-view key-action material evidence.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()

    errors = check_project_scope(args.project_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("project scope ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
