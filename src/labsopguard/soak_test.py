from __future__ import annotations

import csv
import json
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

from labsopguard.runtime_paths import RUNTIME_ROOT, ensure_runtime_dirs
from labsopguard.stream_buffer import RingSegmentRecorder
from labsopguard.ops_metrics import set_capture_disk_free, set_capture_snapshot_metrics


DEFAULT_SOAK_DURATION_SEC = 24 * 60 * 60


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class CameraSoakConfig:
    camera_id: str
    source: str
    source_type: str = "rtsp"
    enabled: bool = True
    expected_fps: float = 30.0
    video_index: int = 0
    sync_group: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    loop_file: bool = False
    segment_duration_sec: Optional[float] = None
    retention_sec: Optional[float] = None
    read_timeout_sec: Optional[float] = None
    reconnect_backoff_sec: Optional[float] = None
    max_reconnect_attempts: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SoakAcceptanceCriteria:
    min_actual_fps_ratio: float = 0.85
    max_drop_rate: float = 0.02
    max_decode_error_rate: float = 0.01
    max_reconnect_count: int = 10
    require_segment_files: bool = True


@dataclass
class DiskProtectionConfig:
    min_free_disk_gb: float = 5.0
    max_output_dir_gb: Optional[float] = None


@dataclass
class SoakTestConfig:
    experiment_id: str
    duration_sec: float = DEFAULT_SOAK_DURATION_SEC
    output_root: str = ".runtime/soak_tests"
    segment_duration_sec: float = 300.0
    retention_sec: float = DEFAULT_SOAK_DURATION_SEC + 3600.0
    heartbeat_interval_sec: float = 10.0
    read_timeout_sec: float = 5.0
    reconnect_backoff_sec: float = 2.0
    cameras: List[CameraSoakConfig] = field(default_factory=list)
    acceptance: SoakAcceptanceCriteria = field(default_factory=SoakAcceptanceCriteria)
    disk_protection: DiskProtectionConfig = field(default_factory=DiskProtectionConfig)


@dataclass
class CameraHealthSnapshot:
    experiment_id: str
    camera_id: str
    source_type: str
    timestamp_utc: str
    elapsed_sec: float
    expected_fps: float
    actual_fps: float
    frame_count: int
    dropped_frame_count: int
    decode_error_count: int
    reconnect_count: int
    segment_count: int
    last_frame_at: Optional[str]
    last_error: Optional[str]
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SyntheticCapture:
    def __init__(self, camera_id: str, fps: float = 10.0, width: int = 96, height: int = 72) -> None:
        self.camera_id = camera_id
        self.fps = max(1.0, float(fps))
        self.width = int(width)
        self.height = int(height)
        self.frame_index = 0
        self.opened = True
        self._next_frame_at = time.monotonic()

    def isOpened(self) -> bool:  # noqa: N802 - keep cv2-like API
        return self.opened

    def read(self) -> tuple[bool, Any]:
        now = time.monotonic()
        if now < self._next_frame_at:
            time.sleep(self._next_frame_at - now)
        self._next_frame_at = time.monotonic() + (1.0 / self.fps)
        value = (self.frame_index * 17) % 255
        frame = np.full((self.height, self.width, 3), value, dtype=np.uint8)
        cv2.putText(
            frame,
            self.camera_id[:12],
            (4, min(self.height - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255 - value, 255, 255),
            1,
            cv2.LINE_AA,
        )
        self.frame_index += 1
        return True, frame

    def release(self) -> None:
        self.opened = False

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0


class CameraSoakWorker:
    def __init__(
        self,
        experiment_id: str,
        camera: CameraSoakConfig,
        *,
        output_dir: Path,
        duration_sec: float,
        segment_duration_sec: float,
        retention_sec: float,
        heartbeat_interval_sec: float,
        read_timeout_sec: float,
        reconnect_backoff_sec: float,
    ) -> None:
        self.experiment_id = experiment_id
        self.camera = camera
        self.output_dir = output_dir
        self.duration_sec = max(0.1, float(duration_sec))
        self.segment_duration_sec = max(1.0, float(camera.segment_duration_sec or segment_duration_sec))
        self.retention_sec = max(self.segment_duration_sec, float(camera.retention_sec or retention_sec))
        self.heartbeat_interval_sec = max(0.2, float(heartbeat_interval_sec))
        self.read_timeout_sec = max(0.2, float(camera.read_timeout_sec or read_timeout_sec))
        self.reconnect_backoff_sec = max(0.1, float(camera.reconnect_backoff_sec or reconnect_backoff_sec))
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.heartbeats: List[CameraHealthSnapshot] = []
        self.frame_count = 0
        self.dropped_frame_count = 0
        self.decode_error_count = 0
        self.reconnect_count = 0
        self.last_frame_at: Optional[str] = None
        self.last_frame_monotonic: Optional[float] = None
        self.last_error: Optional[str] = None
        self.status = "pending"
        self.started_monotonic: Optional[float] = None
        self.finished_monotonic: Optional[float] = None
        self.recorder = RingSegmentRecorder(
            camera_id=camera.camera_id,
            source_id=experiment_id,
            output_dir=output_dir / camera.camera_id,
            segment_duration_sec=self.segment_duration_sec,
            retention_sec=self.retention_sec,
            fps=max(1.0, camera.expected_fps),
        )

    def request_stop(self) -> None:
        self.stop_event.set()

    def snapshot(self) -> CameraHealthSnapshot:
        now = time.monotonic()
        started = self.started_monotonic or now
        elapsed = max(0.0, now - started)
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        return CameraHealthSnapshot(
            experiment_id=self.experiment_id,
            camera_id=self.camera.camera_id,
            source_type=self.camera.source_type,
            timestamp_utc=_utc_now_iso(),
            elapsed_sec=round(elapsed, 3),
            expected_fps=round(float(self.camera.expected_fps), 3),
            actual_fps=round(actual_fps, 3),
            frame_count=self.frame_count,
            dropped_frame_count=self.dropped_frame_count,
            decode_error_count=self.decode_error_count,
            reconnect_count=self.reconnect_count,
            segment_count=len(self.recorder.segments),
            last_frame_at=self.last_frame_at,
            last_error=self.last_error,
            status=self.status,
        )

    def _append_heartbeat(self) -> None:
        snapshot = self.snapshot()
        self.heartbeats.append(snapshot)
        set_capture_snapshot_metrics(snapshot)

    def heartbeat_rows(self) -> List[CameraHealthSnapshot]:
        with self.lock:
            return list(self.heartbeats)

    def _open_capture(self) -> Any:
        source_type = self.camera.source_type.lower()
        if source_type == "synthetic":
            return SyntheticCapture(
                self.camera.camera_id,
                fps=self.camera.expected_fps,
                width=self.camera.width or 96,
                height=self.camera.height or 72,
            )
        source: Any = self.camera.source
        if source_type == "usb":
            try:
                source = int(source)
            except (TypeError, ValueError):
                pass
        cap = cv2.VideoCapture(source)
        if self.camera.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.camera.width))
        if self.camera.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.camera.height))
        if self.camera.expected_fps:
            cap.set(cv2.CAP_PROP_FPS, float(self.camera.expected_fps))
        return cap

    def _close_capture(self, cap: Any) -> None:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _mark_frame(self, frame: Any) -> None:
        now = time.monotonic()
        if self.last_frame_monotonic is not None and self.camera.expected_fps > 0:
            delta = max(0.0, now - self.last_frame_monotonic)
            expected_gap = 1.0 / max(1.0, self.camera.expected_fps)
            if delta > expected_gap * 1.75:
                self.dropped_frame_count += max(0, int(round(delta * self.camera.expected_fps)) - 1)
        self.last_frame_monotonic = now
        self.last_frame_at = _utc_now_iso()
        self.frame_count += 1
        self.recorder.append_frame(frame, now - (self.started_monotonic or now))

    def _should_reconnect(self) -> bool:
        if self.camera.max_reconnect_attempts is None:
            return True
        return self.reconnect_count < int(self.camera.max_reconnect_attempts)

    def run(self) -> None:
        self.started_monotonic = time.monotonic()
        deadline = self.started_monotonic + self.duration_sec
        next_heartbeat = self.started_monotonic
        cap = None
        self.status = "running"
        try:
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                if cap is None or not cap.isOpened():
                    if not self._should_reconnect():
                        self.status = "failed"
                        self.last_error = "max reconnect attempts exceeded"
                        break
                    if self.reconnect_count > 0:
                        time.sleep(self.reconnect_backoff_sec)
                    try:
                        cap = self._open_capture()
                    except Exception as exc:
                        self.last_error = f"open failed: {exc}"
                        self.decode_error_count += 1
                        self.reconnect_count += 1
                        continue
                    if not cap.isOpened():
                        self.last_error = "capture open failed"
                        self.decode_error_count += 1
                        self.reconnect_count += 1
                        self._close_capture(cap)
                        cap = None
                        continue

                ok, frame = cap.read()
                if ok and frame is not None:
                    with self.lock:
                        self._mark_frame(frame)
                        self.status = "running"
                        self.last_error = None
                else:
                    with self.lock:
                        self.decode_error_count += 1
                        self.last_error = "frame read failed"
                    if self.camera.source_type.lower() == "file" and self.camera.loop_file:
                        self._close_capture(cap)
                        cap = self._open_capture()
                    elif self.last_frame_monotonic is None or time.monotonic() - self.last_frame_monotonic >= self.read_timeout_sec:
                        self.reconnect_count += 1
                        self._close_capture(cap)
                        cap = None

                now = time.monotonic()
                if now >= next_heartbeat:
                    self._append_heartbeat()
                    next_heartbeat = now + self.heartbeat_interval_sec
        except Exception as exc:
            with self.lock:
                self.status = "failed"
                self.last_error = f"worker crashed: {exc}"
        finally:
            self.finished_monotonic = time.monotonic()
            self._close_capture(cap)
            self.recorder.close()
            if self.status == "running":
                self.status = "completed"
            with self.lock:
                self._append_heartbeat()


def load_soak_test_config(path: str | Path) -> SoakTestConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    run = payload.get("run") or {}
    defaults = payload.get("defaults") or {}
    acceptance_payload = payload.get("acceptance") or {}
    disk_payload = payload.get("disk_protection") or {}
    cameras: List[CameraSoakConfig] = []
    for index, item in enumerate(payload.get("cameras") or []):
        if not isinstance(item, dict):
            continue
        metadata = {k: v for k, v in item.items() if k not in {
            "camera_id",
            "source",
            "source_type",
            "enabled",
            "expected_fps",
            "video_index",
            "sync_group",
            "width",
            "height",
            "loop_file",
            "segment_duration_sec",
            "retention_sec",
            "read_timeout_sec",
            "reconnect_backoff_sec",
            "max_reconnect_attempts",
        }}
        cameras.append(
            CameraSoakConfig(
                camera_id=str(item.get("camera_id") or f"camera_{index:02d}"),
                source=str(item.get("source") or ""),
                source_type=str(item.get("source_type") or defaults.get("source_type") or "rtsp"),
                enabled=bool(item.get("enabled", True)),
                expected_fps=_safe_float(item.get("expected_fps"), _safe_float(defaults.get("expected_fps"), 30.0)),
                video_index=int(item.get("video_index", index)),
                sync_group=item.get("sync_group") or defaults.get("sync_group"),
                width=item.get("width"),
                height=item.get("height"),
                loop_file=bool(item.get("loop_file", False)),
                segment_duration_sec=item.get("segment_duration_sec"),
                retention_sec=item.get("retention_sec"),
                read_timeout_sec=item.get("read_timeout_sec"),
                reconnect_backoff_sec=item.get("reconnect_backoff_sec"),
                max_reconnect_attempts=item.get("max_reconnect_attempts"),
                metadata=metadata,
            )
        )

    return SoakTestConfig(
        experiment_id=str(run.get("experiment_id") or f"multicam_soak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        duration_sec=_safe_float(run.get("duration_sec"), DEFAULT_SOAK_DURATION_SEC),
        output_root=str(run.get("output_root") or ".runtime/soak_tests"),
        segment_duration_sec=_safe_float(defaults.get("segment_duration_sec"), 300.0),
        retention_sec=_safe_float(defaults.get("retention_sec"), DEFAULT_SOAK_DURATION_SEC + 3600.0),
        heartbeat_interval_sec=_safe_float(defaults.get("heartbeat_interval_sec"), 10.0),
        read_timeout_sec=_safe_float(defaults.get("read_timeout_sec"), 5.0),
        reconnect_backoff_sec=_safe_float(defaults.get("reconnect_backoff_sec"), 2.0),
        cameras=[camera for camera in cameras if camera.enabled],
        acceptance=SoakAcceptanceCriteria(
            min_actual_fps_ratio=_safe_float(acceptance_payload.get("min_actual_fps_ratio"), 0.85),
            max_drop_rate=_safe_float(acceptance_payload.get("max_drop_rate"), 0.02),
            max_decode_error_rate=_safe_float(acceptance_payload.get("max_decode_error_rate"), 0.01),
            max_reconnect_count=int(acceptance_payload.get("max_reconnect_count", 10)),
            require_segment_files=bool(acceptance_payload.get("require_segment_files", True)),
        ),
        disk_protection=DiskProtectionConfig(
            min_free_disk_gb=_safe_float(disk_payload.get("min_free_disk_gb"), 5.0),
            max_output_dir_gb=(
                _safe_float(disk_payload.get("max_output_dir_gb"), 0.0)
                if disk_payload.get("max_output_dir_gb") is not None else None
            ),
        ),
    )


def make_dry_run_config(duration_sec: float = 3.0, output_root: str = ".runtime/soak_tests") -> SoakTestConfig:
    return SoakTestConfig(
        experiment_id=f"multicam_soak_dry_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        duration_sec=duration_sec,
        output_root=output_root,
        segment_duration_sec=1.0,
        retention_sec=60.0,
        heartbeat_interval_sec=0.5,
        read_timeout_sec=1.0,
        reconnect_backoff_sec=0.2,
        cameras=[
            CameraSoakConfig(camera_id="dry_front", source="synthetic://front", source_type="synthetic", expected_fps=5.0),
            CameraSoakConfig(camera_id="dry_side", source="synthetic://side", source_type="synthetic", expected_fps=5.0),
        ],
        acceptance=SoakAcceptanceCriteria(
            min_actual_fps_ratio=0.5,
            max_drop_rate=0.2,
            max_decode_error_rate=0.2,
            max_reconnect_count=2,
            require_segment_files=True,
        ),
        disk_protection=DiskProtectionConfig(min_free_disk_gb=0.01, max_output_dir_gb=None),
    )


def _video_inputs(config: SoakTestConfig) -> List[Dict[str, Any]]:
    return [
        {
            "schema_version": "video_input.v1",
            "video_index": camera.video_index,
            "camera_id": camera.camera_id,
            "source_type": camera.source_type,
            "source": camera.source,
            "video_path": camera.source,
            "sync_group": camera.sync_group,
            "expected_fps": camera.expected_fps,
            "capture_duration_sec": config.duration_sec,
            "segment_duration_sec": camera.segment_duration_sec or config.segment_duration_sec,
            "metadata": camera.metadata,
        }
        for camera in config.cameras
    ]


def _build_material_stream(config: SoakTestConfig, workers: List[CameraSoakWorker]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for worker in workers:
        camera = worker.camera
        for segment in worker.recorder.segments:
            path = Path(segment.file_path)
            item_id = f"{config.experiment_id}:{segment.segment_id}"
            items.append(
                {
                    "schema_version": "material_stream.v1",
                    "item_id": item_id,
                    "experiment_id": config.experiment_id,
                    "material_type": "video_segment",
                    "stream_id": camera.camera_id,
                    "camera_id": camera.camera_id,
                    "video_index": camera.video_index,
                    "source_type": camera.source_type,
                    "source_path": camera.source,
                    "recorded_file_path": str(path),
                    "file_exists": path.exists(),
                    "timestamp_sec": segment.start_time_sec,
                    "local_timestamp_sec": segment.start_time_sec,
                    "end_time_sec": segment.end_time_sec,
                    "duration_sec": round(segment.end_time_sec - segment.start_time_sec, 3),
                    "clip_id": segment.segment_id,
                    "clip_file_path": str(path),
                    "segment_id": segment.segment_id,
                    "frame_count": segment.frame_count,
                    "fps": segment.fps,
                    "sync_group": camera.sync_group,
                }
            )
    return items


def _evaluate_camera(
    config: SoakTestConfig,
    worker: CameraSoakWorker,
) -> Dict[str, Any]:
    snapshot = worker.snapshot()
    total_frame_slots = snapshot.frame_count + snapshot.dropped_frame_count
    drop_rate = snapshot.dropped_frame_count / total_frame_slots if total_frame_slots else 1.0
    decode_error_rate = snapshot.decode_error_count / max(1, snapshot.frame_count + snapshot.decode_error_count)
    fps_ratio = snapshot.actual_fps / snapshot.expected_fps if snapshot.expected_fps > 0 else 1.0
    segment_files = [Path(segment.file_path) for segment in worker.recorder.segments]
    missing_segments = [str(path) for path in segment_files if not path.exists()]
    reasons: List[str] = []
    if snapshot.frame_count <= 0:
        reasons.append("no frames captured")
    if fps_ratio < config.acceptance.min_actual_fps_ratio:
        reasons.append(f"actual fps ratio {fps_ratio:.3f} below {config.acceptance.min_actual_fps_ratio:.3f}")
    if drop_rate > config.acceptance.max_drop_rate:
        reasons.append(f"drop rate {drop_rate:.3f} above {config.acceptance.max_drop_rate:.3f}")
    if decode_error_rate > config.acceptance.max_decode_error_rate:
        reasons.append(f"decode error rate {decode_error_rate:.3f} above {config.acceptance.max_decode_error_rate:.3f}")
    if snapshot.reconnect_count > config.acceptance.max_reconnect_count:
        reasons.append(f"reconnect count {snapshot.reconnect_count} above {config.acceptance.max_reconnect_count}")
    if config.acceptance.require_segment_files and missing_segments:
        reasons.append(f"{len(missing_segments)} segment files missing")
    if config.acceptance.require_segment_files and not segment_files:
        reasons.append("no segment files produced")

    return {
        "camera_id": worker.camera.camera_id,
        "status": "passed" if not reasons else "failed",
        "failure_reasons": reasons,
        "actual_fps": snapshot.actual_fps,
        "expected_fps": snapshot.expected_fps,
        "fps_ratio": round(fps_ratio, 6),
        "drop_rate": round(drop_rate, 6),
        "decode_error_rate": round(decode_error_rate, 6),
        "frame_count": snapshot.frame_count,
        "dropped_frame_count": snapshot.dropped_frame_count,
        "decode_error_count": snapshot.decode_error_count,
        "reconnect_count": snapshot.reconnect_count,
        "segment_count": len(segment_files),
        "missing_segment_files": missing_segments,
        "last_error": snapshot.last_error,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_health_csv(path: Path, rows: List[CameraHealthSnapshot]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CameraHealthSnapshot.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        f"# Multi-Camera Soak Test Report",
        "",
        f"- experiment_id: `{report['experiment_id']}`",
        f"- status: `{report['status']}`",
        f"- started_at: `{report['started_at']}`",
        f"- finished_at: `{report['finished_at']}`",
        f"- duration_sec: `{report['duration_sec']}`",
        "",
        "## Camera Summary",
        "",
        "| camera_id | status | frames | actual_fps | dropped | reconnects | segments |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["camera_results"]:
        lines.append(
            "| {camera_id} | {status} | {frame_count} | {actual_fps} | {dropped_frame_count} | {reconnect_count} | {segment_count} |".format(
                **item
            )
        )
    lines.extend(["", "## Failed Checks", ""])
    any_failure = False
    for item in report["camera_results"]:
        for reason in item["failure_reasons"]:
            any_failure = True
            lines.append(f"- `{item['camera_id']}`: {reason}")
    if not any_failure:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def _disk_status(path: Path) -> Dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gb": round(usage.free / (1024 ** 3), 3),
    }


def _check_disk_protection(config: SoakTestConfig, run_dir: Path) -> Optional[str]:
    disk = _disk_status(run_dir)
    set_capture_disk_free(config.experiment_id, int(disk["free_bytes"]))
    min_free = max(0.0, float(config.disk_protection.min_free_disk_gb))
    if disk["free_gb"] < min_free:
        return f"free disk {disk['free_gb']}GB below {min_free}GB"
    if config.disk_protection.max_output_dir_gb is not None:
        size_gb = _dir_size_bytes(run_dir) / (1024 ** 3)
        if size_gb > float(config.disk_protection.max_output_dir_gb):
            return f"output dir {size_gb:.3f}GB above {config.disk_protection.max_output_dir_gb}GB"
    return None


def _collect_health_rows(workers: List[CameraSoakWorker], *, include_current: bool = False) -> List[CameraHealthSnapshot]:
    rows: List[CameraHealthSnapshot] = []
    for worker in workers:
        rows.extend(worker.heartbeat_rows())
        if include_current:
            rows.append(worker.snapshot())
    return rows


def _write_live_artifacts(run_dir: Path, config: SoakTestConfig, workers: List[CameraSoakWorker]) -> None:
    health_rows = _collect_health_rows(workers, include_current=True)
    _write_json(run_dir / "stream_health.json", [row.to_dict() for row in health_rows])
    _write_health_csv(run_dir / "stream_health.csv", health_rows)
    _write_json(run_dir / "material_stream.json", _build_material_stream(config, workers))


def preflight_soak_test_sources(config: SoakTestConfig, *, read_attempts: int = 3) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for camera in config.cameras:
        worker = CameraSoakWorker(
            config.experiment_id,
            camera,
            output_dir=Path(config.output_root) / "_preflight",
            duration_sec=1.0,
            segment_duration_sec=1.0,
            retention_sec=2.0,
            heartbeat_interval_sec=1.0,
            read_timeout_sec=config.read_timeout_sec,
            reconnect_backoff_sec=config.reconnect_backoff_sec,
        )
        cap = None
        frame_ok = False
        error = None
        try:
            cap = worker._open_capture()
            opened = bool(cap and cap.isOpened())
            if opened:
                for _ in range(max(1, int(read_attempts))):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        frame_ok = True
                        break
            else:
                error = "capture open failed"
        except Exception as exc:
            opened = False
            error = str(exc)
        finally:
            worker._close_capture(cap)
            worker.recorder.close()
        results.append(
            {
                "camera_id": camera.camera_id,
                "source_type": camera.source_type,
                "source": camera.source,
                "opened": opened,
                "frame_read": frame_ok,
                "status": "passed" if opened and frame_ok else "failed",
                "error": error,
            }
        )
    return {
        "schema_version": "multicam_source_preflight.v1",
        "experiment_id": config.experiment_id,
        "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
        "camera_count": len(results),
        "results": results,
    }


def run_soak_test(config: SoakTestConfig) -> Dict[str, Any]:
    if not config.cameras:
        raise ValueError("No enabled cameras configured. Enable cameras in config or run with --dry-run.")

    ensure_runtime_dirs()
    output_root = Path(config.output_root)
    if not output_root.is_absolute():
        output_root = RUNTIME_ROOT.parent / output_root
    run_dir = output_root / config.experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "video_inputs.json", _video_inputs(config))
    disk_stop_reason: Optional[str] = None

    started_at = _utc_now_iso()
    workers = [
        CameraSoakWorker(
            config.experiment_id,
            camera,
            output_dir=run_dir / "segments",
            duration_sec=config.duration_sec,
            segment_duration_sec=config.segment_duration_sec,
            retention_sec=config.retention_sec,
            heartbeat_interval_sec=config.heartbeat_interval_sec,
            read_timeout_sec=config.read_timeout_sec,
            reconnect_backoff_sec=config.reconnect_backoff_sec,
        )
        for camera in config.cameras
    ]
    threads = [threading.Thread(target=worker.run, name=f"soak-{worker.camera.camera_id}", daemon=True) for worker in workers]
    for thread in threads:
        thread.start()
    try:
        monitor_interval_sec = max(1.0, min(10.0, config.heartbeat_interval_sec))
        while any(thread.is_alive() for thread in threads):
            time.sleep(monitor_interval_sec)
            _write_live_artifacts(run_dir, config, workers)
            disk_stop_reason = _check_disk_protection(config, run_dir)
            if disk_stop_reason:
                for worker in workers:
                    worker.last_error = disk_stop_reason
                    worker.status = "failed"
                    worker.request_stop()
                break
    except KeyboardInterrupt:
        for worker in workers:
            worker.request_stop()
        for thread in threads:
            thread.join(timeout=5.0)
    finally:
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=1.0)

    finished_at = _utc_now_iso()
    health_rows = _collect_health_rows(workers)

    material_stream = _build_material_stream(config, workers)
    camera_results = [_evaluate_camera(config, worker) for worker in workers]
    status = "passed" if not disk_stop_reason and all(item["status"] == "passed" for item in camera_results) else "failed"
    report = {
        "schema_version": "multicam_soak_test.v1",
        "experiment_id": config.experiment_id,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": config.duration_sec,
        "output_dir": str(run_dir),
        "acceptance": asdict(config.acceptance),
        "disk_protection": asdict(config.disk_protection),
        "disk_status": _disk_status(run_dir),
        "disk_stop_reason": disk_stop_reason,
        "video_inputs": _video_inputs(config),
        "camera_results": camera_results,
        "artifact_paths": {
            "report_json": str(run_dir / "soak_test_report.json"),
            "report_md": str(run_dir / "soak_test_report.md"),
            "stream_health_csv": str(run_dir / "stream_health.csv"),
            "stream_health_json": str(run_dir / "stream_health.json"),
            "material_stream": str(run_dir / "material_stream.json"),
            "video_inputs": str(run_dir / "video_inputs.json"),
        },
    }

    _write_json(run_dir / "video_inputs.json", report["video_inputs"])
    _write_json(run_dir / "stream_health.json", [row.to_dict() for row in health_rows])
    _write_health_csv(run_dir / "stream_health.csv", health_rows)
    _write_json(run_dir / "material_stream.json", material_stream)
    _write_json(run_dir / "soak_test_report.json", report)
    _write_markdown_report(run_dir / "soak_test_report.md", report)
    return report
