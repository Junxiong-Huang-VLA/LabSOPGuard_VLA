from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ConfigError, load_config
from .frames import FrameDataError, load_camera_frames
from .manifest import build_manifest, resolve_streams
from .sync import run_sync


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="realityloop-sync")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run metadata-only multi-camera frame synchronization")
    run_parser.add_argument("--config", required=True)

    inspect_parser = sub.add_parser("inspect", help="Inspect configured camera frames without syncing")
    inspect_parser.add_argument("--config", required=True)

    manifest_parser = sub.add_parser("build-manifest", help="Build a manifest CSV from a video database root")
    manifest_parser.add_argument("--root", required=True)
    manifest_parser.add_argument("--date", required=True)
    manifest_parser.add_argument("--start-time", required=True)
    manifest_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config = load_config(args.config)
            result = run_sync(config)
            print(json.dumps({"status": "completed", "output_dir": str(result.output_dir), **result.report.payload}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "inspect":
            config = load_config(args.config)
            effective, streams, warnings = resolve_streams(config)
            cameras = [load_camera_frames(stream, effective.timestamp) for stream in streams]
            payload = {
                "status": "ok",
                "run_id": effective.run_id,
                "reference_camera": effective.reference_camera,
                "camera_count": len(cameras),
                "timestamp_col_used": {camera.camera_id: camera.timestamp_col for camera in cameras},
                "frame_count": {camera.camera_id: camera.frame_count for camera in cameras},
                "detected_median_frame_interval_us": {
                    camera.camera_id: camera.median_frame_interval_us for camera in cameras
                },
                "warnings": [*warnings, *(warning for camera in cameras for warning in camera.warnings)],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "build-manifest":
            output = build_manifest(
                video_database_root=Path(args.root),
                date=args.date,
                start_time=args.start_time,
                output=Path(args.output),
            )
            print(json.dumps({"status": "completed", "manifest_csv": str(output)}, ensure_ascii=False, indent=2))
            return 0
    except (ConfigError, FrameDataError, FileNotFoundError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

