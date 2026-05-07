from __future__ import annotations

import copy
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from wireless_video.config import ReceiverConfig, load_receiver_config
from wireless_video.models import VideoFrame
from wireless_video.receiver_app import DeviceRegistry, ReceiverService

from .types import (
    CalibrationInfo,
    CameraCapabilities,
    CameraInfo,
    CameraStatus,
    FrameMeta,
    PairMeta,
    PairMode,
    ReadStatus,
    SenderInfo,
    StreamName,
    StreamProfile,
)


def _status_value(status: ReadStatus) -> str:
    return status.value if isinstance(status, ReadStatus) else str(status)


@dataclass
class _StreamSample:
    seq: int
    stream_name: str
    frame_id: int
    timestamp_us: int
    pixel_fmt: str
    data: np.ndarray


@dataclass
class _PairLookup:
    online: bool
    depth_stream_available: bool
    matched: bool
    saw_both: bool
    rgb: Optional[_StreamSample]
    depth: Optional[_StreamSample]
    delta_ms: Optional[float]


@dataclass
class _CameraRuntime:
    sender_id: str
    camera_id: str
    sender_name: str
    camera_name: str
    online: bool = False
    last_status_code: Optional[str] = None
    last_frame_ts_us: Optional[int] = None
    last_seen_monotonic: float = 0.0
    latest: dict[str, _StreamSample] = field(default_factory=dict)
    history: dict[str, deque[_StreamSample]] = field(default_factory=dict)
    stream_fps: dict[str, float] = field(default_factory=dict)
    stream_profiles: dict[str, StreamProfile] = field(default_factory=dict)
    expected_streams: set[str] = field(default_factory=lambda: {StreamName.RGB.value})
    seq_counter: int = 0


class _SdkStore:
    def __init__(self, history_size: int = 32, offline_ttl_s: float = 2.5) -> None:
        self._history_size = max(2, history_size)
        self._offline_ttl_s = max(0.5, offline_ttl_s)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._cameras: dict[tuple[str, str], _CameraRuntime] = {}

    def _get_or_create_runtime_locked(self, sender_id: str, camera_id: str) -> _CameraRuntime:
        key = (sender_id, camera_id)
        runtime = self._cameras.get(key)
        if runtime is None:
            runtime = _CameraRuntime(
                sender_id=sender_id,
                camera_id=camera_id,
                sender_name=sender_id,
                camera_name=camera_id,
            )
            self._cameras[key] = runtime
        return runtime

    def _refresh_online_locked(self) -> None:
        now = time.monotonic()
        for runtime in self._cameras.values():
            if runtime.online and runtime.last_seen_monotonic > 0 and now - runtime.last_seen_monotonic > self._offline_ttl_s:
                runtime.online = False

    def register_expected_camera(
        self,
        sender_id: str,
        camera_id: str,
        *,
        sender_name: Optional[str] = None,
        camera_name: Optional[str] = None,
        expected_streams: Optional[set[str]] = None,
    ) -> None:
        sid = sender_id or "unknown_sender"
        cid = camera_id or "unknown_camera"
        with self._cond:
            runtime = self._get_or_create_runtime_locked(sid, cid)
            if sender_name:
                runtime.sender_name = sender_name
            if camera_name:
                runtime.camera_name = camera_name
            if expected_streams:
                runtime.expected_streams.update({item for item in expected_streams if item})
            self._cond.notify_all()

    def update_from_registry(self, records: list[dict[str, object]]) -> None:
        with self._cond:
            now_ms = int(time.time() * 1000)
            freshness_ms = int(self._offline_ttl_s * 1000)
            for record in records:
                sid = str(record.get("sender_id") or "unknown_sender")
                cid = str(record.get("camera_id") or "unknown_camera")
                runtime = self._get_or_create_runtime_locked(sid, cid)
                runtime.sender_name = sid
                runtime.camera_name = cid
                state = str(record.get("state") or "")
                raw_last_seen = record.get("last_seen_ms")
                try:
                    last_seen_ms = int(raw_last_seen) if raw_last_seen is not None else now_ms
                except Exception:
                    last_seen_ms = now_ms
                fresh = (now_ms - last_seen_ms) <= freshness_ms
                runtime.online = state in ("RUNNING", "DEGRADED", "RECONNECTING") and fresh
                if runtime.online:
                    runtime.last_seen_monotonic = time.monotonic()
                last_error = str(record.get("last_error") or "")
                runtime.last_status_code = last_error or runtime.last_status_code
            self._refresh_online_locked()
            self._cond.notify_all()

    def push_frame(self, frame: VideoFrame) -> None:
        sid = frame.sender_id or "unknown_sender"
        cid = frame.camera_id or frame.channel_id or "unknown_camera"
        stream_name = frame.stream_name or StreamName.RGB.value
        with self._cond:
            runtime = self._get_or_create_runtime_locked(sid, cid)
            runtime.seq_counter += 1
            ts_us = int(frame.capture_ts_ms) * 1000
            sample = _StreamSample(
                seq=runtime.seq_counter,
                stream_name=stream_name,
                frame_id=int(frame.frame_id),
                timestamp_us=ts_us,
                pixel_fmt=frame.pixel_fmt,
                data=frame.data,
            )
            queue = runtime.history.get(stream_name)
            if queue is None:
                queue = deque(maxlen=self._history_size)
                runtime.history[stream_name] = queue
            prev = runtime.latest.get(stream_name)
            queue.append(sample)
            runtime.latest[stream_name] = sample
            runtime.expected_streams.add(stream_name)
            runtime.online = True
            runtime.last_seen_monotonic = time.monotonic()
            runtime.last_frame_ts_us = ts_us
            runtime.last_status_code = ReadStatus.OK.value
            if sid != "unknown_sender":
                unknown_key = ("unknown_sender", cid)
                unknown_runtime = self._cameras.get(unknown_key)
                if unknown_runtime is not None and unknown_runtime is not runtime:
                    unknown_runtime.online = False
            if prev is not None:
                delta_us = sample.timestamp_us - prev.timestamp_us
                if delta_us > 0:
                    inst_fps = 1_000_000.0 / float(delta_us)
                    current = runtime.stream_fps.get(stream_name, inst_fps)
                    runtime.stream_fps[stream_name] = current * 0.8 + inst_fps * 0.2
            fps = runtime.stream_fps.get(stream_name, 0.0)
            runtime.stream_profiles[stream_name] = StreamProfile(
                stream_name=stream_name,
                width=int(frame.width),
                height=int(frame.height),
                fps=round(fps, 2),
                pixel_format=frame.pixel_fmt,
            )
            self._cond.notify_all()

    def list_senders(self) -> list[SenderInfo]:
        with self._lock:
            self._refresh_online_locked()
            grouped: dict[str, dict[str, object]] = defaultdict(lambda: {"online": False, "cameras": set()})
            for runtime in self._cameras.values():
                item = grouped[runtime.sender_id]
                item["online"] = bool(item["online"]) or runtime.online
                cameras: set[str] = item["cameras"]  # type: ignore[assignment]
                cameras.add(runtime.camera_id)
            out = [
                SenderInfo(
                    sender_id=sender_id,
                    sender_name=sender_id,
                    online=bool(item["online"]),
                    camera_count=len(item["cameras"]),  # type: ignore[arg-type]
                )
                for sender_id, item in grouped.items()
            ]
            out.sort(key=lambda x: x.sender_id)
            return out

    def list_cameras(self) -> list[CameraInfo]:
        with self._lock:
            self._refresh_online_locked()
            preferred_by_camera: dict[str, _CameraRuntime] = {}
            for runtime in self._cameras.values():
                existing = preferred_by_camera.get(runtime.camera_id)
                if existing is None:
                    preferred_by_camera[runtime.camera_id] = runtime
                    continue
                existing_score = (
                    1 if existing.online else 0,
                    1 if existing.sender_id != "unknown_sender" else 0,
                    1 if existing.last_frame_ts_us is not None else 0,
                    int(existing.last_frame_ts_us or 0),
                )
                current_score = (
                    1 if runtime.online else 0,
                    1 if runtime.sender_id != "unknown_sender" else 0,
                    1 if runtime.last_frame_ts_us is not None else 0,
                    int(runtime.last_frame_ts_us or 0),
                )
                if current_score > existing_score:
                    preferred_by_camera[runtime.camera_id] = runtime
            out = [
                CameraInfo(
                    sender_id=runtime.sender_id,
                    sender_name=runtime.sender_name,
                    camera_id=runtime.camera_id,
                    camera_name=runtime.camera_name,
                    online=runtime.online,
                )
                for runtime in preferred_by_camera.values()
            ]
            out.sort(key=lambda x: (x.sender_id, x.camera_id))
            return out

    def resolve_camera_key(self, camera_id: str, sender_id: Optional[str] = None) -> tuple[str, str]:
        cid = camera_id.strip()
        if not cid:
            raise ValueError("camera_id is empty")
        with self._lock:
            if sender_id:
                key = (sender_id, cid)
                if key in self._cameras:
                    return key
                self._get_or_create_runtime_locked(sender_id, cid)
                return key
            candidates = [key for key in self._cameras.keys() if key[1] == cid]
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                self._refresh_online_locked()
                return max(
                    candidates,
                    key=lambda key: (
                        1 if self._cameras[key].online else 0,
                        1 if self._cameras[key].sender_id != "unknown_sender" else 0,
                        1 if self._cameras[key].last_frame_ts_us is not None else 0,
                        int(self._cameras[key].last_frame_ts_us or 0),
                        self._cameras[key].last_seen_monotonic,
                    ),
                )
            key = ("unknown_sender", cid)
            self._get_or_create_runtime_locked(*key)
            return key

    def get_capabilities(self, key: tuple[str, str]) -> CameraCapabilities:
        with self._lock:
            runtime = self._cameras.get(key)
            if runtime is None:
                return CameraCapabilities(
                    supports_rgb=False,
                    supports_depth_raw=False,
                    supports_depth_aligned_to_rgb=False,
                    supports_depth_color=False,
                    supports_strict_pair=False,
                    has_calibration=False,
                )
            streams = set(runtime.expected_streams) | set(runtime.latest.keys())
            supports_rgb = StreamName.RGB.value in streams
            supports_depth_raw = StreamName.DEPTH_RAW.value in streams
            supports_depth_aligned = StreamName.DEPTH_ALIGNED_TO_RGB.value in streams
            supports_depth_color = StreamName.DEPTH_COLOR.value in streams
            return CameraCapabilities(
                supports_rgb=supports_rgb,
                supports_depth_raw=supports_depth_raw,
                supports_depth_aligned_to_rgb=supports_depth_aligned,
                supports_depth_color=supports_depth_color,
                supports_strict_pair=supports_rgb and (supports_depth_raw or supports_depth_aligned),
                has_calibration=False,
            )

    def get_profiles(self, key: tuple[str, str]) -> list[StreamProfile]:
        with self._lock:
            runtime = self._cameras.get(key)
            if runtime is None:
                return []
            profiles = list(runtime.stream_profiles.values())
            profiles.sort(key=lambda item: item.stream_name)
            return profiles

    def get_status(self, key: tuple[str, str]) -> CameraStatus:
        with self._lock:
            self._refresh_online_locked()
            runtime = self._cameras.get(key)
            if runtime is None:
                return CameraStatus(
                    sender_id=key[0],
                    camera_id=key[1],
                    online=False,
                    rgb_available=False,
                    depth_available=False,
                    depth_color_available=False,
                    pair_mode_supported=[],
                    calibration_available=False,
                    last_frame_ts_us=None,
                    last_status_code=ReadStatus.CAMERA_OFFLINE.value,
                )
            streams = set(runtime.expected_streams) | set(runtime.latest.keys())
            rgb_available = StreamName.RGB.value in streams and runtime.online
            depth_available = (
                StreamName.DEPTH_RAW.value in streams or StreamName.DEPTH_ALIGNED_TO_RGB.value in streams
            ) and runtime.online
            depth_color_available = StreamName.DEPTH_COLOR.value in streams and runtime.online
            pair_modes = [PairMode.REALTIME.value]
            if rgb_available and depth_available:
                pair_modes.insert(0, PairMode.STRICT.value)
            return CameraStatus(
                sender_id=runtime.sender_id,
                camera_id=runtime.camera_id,
                online=runtime.online,
                rgb_available=rgb_available,
                depth_available=depth_available,
                depth_color_available=depth_color_available,
                pair_mode_supported=pair_modes,
                calibration_available=False,
                last_frame_ts_us=runtime.last_frame_ts_us,
                last_status_code=runtime.last_status_code,
            )

    def wait_for_stream(
        self,
        key: tuple[str, str],
        stream_name: str,
        timeout_ms: int,
        min_seq: Optional[int] = None,
    ) -> tuple[ReadStatus, Optional[_StreamSample]]:
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        with self._cond:
            while True:
                self._refresh_online_locked()
                runtime = self._cameras.get(key)
                if runtime is None:
                    return ReadStatus.CAMERA_OFFLINE, None
                sample = runtime.latest.get(stream_name)
                stream_available = stream_name in runtime.expected_streams or sample is not None
                if not stream_available:
                    return ReadStatus.STREAM_UNAVAILABLE, None
                if sample is not None and (min_seq is None or sample.seq > min_seq):
                    return ReadStatus.OK, sample
                if not runtime.online and sample is None:
                    return ReadStatus.CAMERA_OFFLINE, None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not runtime.online and sample is None:
                        return ReadStatus.CAMERA_OFFLINE, None
                    return ReadStatus.TIMEOUT, None
                self._cond.wait(timeout=remaining)

    def get_latest_stream(self, key: tuple[str, str], stream_name: str) -> tuple[ReadStatus, Optional[_StreamSample]]:
        with self._lock:
            self._refresh_online_locked()
            runtime = self._cameras.get(key)
            if runtime is None:
                return ReadStatus.CAMERA_OFFLINE, None
            sample = runtime.latest.get(stream_name)
            stream_available = stream_name in runtime.expected_streams or sample is not None
            if not stream_available:
                return ReadStatus.STREAM_UNAVAILABLE, None
            if sample is None:
                return (ReadStatus.CAMERA_OFFLINE, None) if not runtime.online else (ReadStatus.TIMEOUT, None)
            return ReadStatus.OK, sample

    def pair_lookup(
        self,
        key: tuple[str, str],
        *,
        depth_stream: str,
        mode: PairMode,
        max_delta_ms: int,
        min_rgb_seq: Optional[int],
        min_depth_seq: Optional[int],
    ) -> _PairLookup:
        with self._lock:
            self._refresh_online_locked()
            runtime = self._cameras.get(key)
            if runtime is None:
                return _PairLookup(
                    online=False,
                    depth_stream_available=False,
                    matched=False,
                    saw_both=False,
                    rgb=None,
                    depth=None,
                    delta_ms=None,
                )
            rgb_latest = runtime.latest.get(StreamName.RGB.value)
            depth_latest = runtime.latest.get(depth_stream)
            depth_stream_available = depth_stream in runtime.expected_streams or depth_latest is not None
            rgb_candidates = list(runtime.history.get(StreamName.RGB.value, ()))
            depth_candidates = list(runtime.history.get(depth_stream, ()))
            if min_rgb_seq is not None:
                rgb_candidates = [item for item in rgb_candidates if item.seq > min_rgb_seq]
            if min_depth_seq is not None:
                depth_candidates = [item for item in depth_candidates if item.seq > min_depth_seq]
            if mode == PairMode.REALTIME:
                if rgb_candidates and depth_candidates:
                    rgb = rgb_candidates[-1]
                    depth = depth_candidates[-1]
                    delta_ms = abs(rgb.timestamp_us - depth.timestamp_us) / 1000.0
                    return _PairLookup(
                        online=runtime.online,
                        depth_stream_available=depth_stream_available,
                        matched=True,
                        saw_both=True,
                        rgb=rgb,
                        depth=depth,
                        delta_ms=delta_ms,
                    )
                return _PairLookup(
                    online=runtime.online,
                    depth_stream_available=depth_stream_available,
                    matched=False,
                    saw_both=bool(rgb_candidates and depth_candidates),
                    rgb=rgb_candidates[-1] if rgb_candidates else rgb_latest,
                    depth=depth_candidates[-1] if depth_candidates else depth_latest,
                    delta_ms=None,
                )
            best_rgb: Optional[_StreamSample] = None
            best_depth: Optional[_StreamSample] = None
            best_delta: Optional[float] = None
            for rgb in rgb_candidates[-8:]:
                for depth in depth_candidates[-8:]:
                    delta_ms = abs(rgb.timestamp_us - depth.timestamp_us) / 1000.0
                    if delta_ms > max_delta_ms:
                        continue
                    if best_delta is None or delta_ms < best_delta:
                        best_rgb = rgb
                        best_depth = depth
                        best_delta = delta_ms
            return _PairLookup(
                online=runtime.online,
                depth_stream_available=depth_stream_available,
                matched=best_rgb is not None and best_depth is not None,
                saw_both=bool(rgb_candidates and depth_candidates),
                rgb=best_rgb or (rgb_candidates[-1] if rgb_candidates else rgb_latest),
                depth=best_depth or (depth_candidates[-1] if depth_candidates else depth_latest),
                delta_ms=best_delta,
            )

    def wait_for_update(self, timeout_ms: int) -> None:
        with self._cond:
            self._cond.wait(timeout=max(0, timeout_ms) / 1000.0)


@dataclass
class _ServiceBinding:
    service: ReceiverService
    sender_id: str
    camera_id: str
    stream_name: str


class _SdkReceiverManager:
    def __init__(self, cfg: ReceiverConfig, store: _SdkStore) -> None:
        self._cfg = cfg
        self._store = store
        self._registry = DeviceRegistry()
        self._stop_event = threading.Event()
        self._bindings = self._build_bindings(cfg)
        self._threads: list[threading.Thread] = []
        self._registry_thread: Optional[threading.Thread] = None

    def _build_bindings(self, cfg: ReceiverConfig) -> list[_ServiceBinding]:
        channels = [item for item in cfg.channels if item.enabled]
        if not channels:
            channel_cfg = copy.deepcopy(cfg)
            channel_cfg.channels = []
            service = ReceiverService(
                channel_cfg,
                name="rx0",
                window_name=cfg.display.window_name,
                show_stats=False,
                registry=self._registry,
                enable_renderer=False,
            )
            self._store.register_expected_camera("unknown_sender", "rx0", expected_streams={StreamName.RGB.value})
            return [
                _ServiceBinding(
                    service=service,
                    sender_id="unknown_sender",
                    camera_id="rx0",
                    stream_name=StreamName.RGB.value,
                )
            ]
        bindings: list[_ServiceBinding] = []
        port_step = max(1, cfg.network.port_step)
        for index, channel in enumerate(channels):
            channel_cfg = copy.deepcopy(cfg)
            channel_cfg.channels = []
            channel_cfg.display.show_stats = False
            channel_cfg.network.listen_port = (
                channel.listen_port if channel.listen_port is not None else cfg.network.listen_port + index * port_step
            )
            channel_name = channel.channel_id or f"rx{index}"
            service = ReceiverService(
                channel_cfg,
                name=channel_name,
                window_name=channel.window_name or f"{cfg.display.window_name}-{channel_name}",
                show_stats=False,
                registry=self._registry,
                enable_renderer=False,
            )
            sender_id = channel.sender_id or "unknown_sender"
            camera_id = channel.camera_id or channel_name
            stream_name = channel.stream_name or StreamName.RGB.value
            self._store.register_expected_camera(sender_id, camera_id, expected_streams={stream_name})
            bindings.append(
                _ServiceBinding(
                    service=service,
                    sender_id=sender_id,
                    camera_id=camera_id,
                    stream_name=stream_name,
                )
            )
        return bindings

    def start(self) -> None:
        started: list[_ServiceBinding] = []
        for binding in self._bindings:
            try:
                binding.service.start()
                started.append(binding)
            except Exception:
                binding.service.stop()
        if not started:
            raise RuntimeError("no receiver channel started for SDK")
        self._bindings = started
        self._stop_event.clear()
        for binding in self._bindings:
            thread = threading.Thread(target=self._pump_loop, args=(binding,), daemon=True)
            thread.start()
            self._threads.append(thread)
        self._registry_thread = threading.Thread(target=self._registry_loop, daemon=True)
        self._registry_thread.start()

    def _pump_loop(self, binding: _ServiceBinding) -> None:
        while not self._stop_event.is_set():
            frame = binding.service.decoded_queue.pop(50)
            if frame is None:
                continue
            if not frame.sender_id:
                frame.sender_id = binding.sender_id
            frame.camera_id = binding.camera_id
            if not frame.stream_name:
                frame.stream_name = binding.stream_name
            self._store.push_frame(frame)

    def _registry_loop(self) -> None:
        while not self._stop_event.is_set():
            records = self._registry.snapshot()
            if records:
                bindings_by_port = {
                    int(binding.service.cfg.network.listen_port): binding
                    for binding in self._bindings
                }
                remapped_records: list[dict[str, object]] = []
                for record in records:
                    remapped = dict(record)
                    try:
                        listen_port = int(remapped.get("listen_port") or 0)
                    except Exception:
                        listen_port = 0
                    binding = bindings_by_port.get(listen_port)
                    if binding is not None:
                        remapped["camera_id"] = binding.camera_id
                        remapped["stream_channel_id"] = binding.service.name
                    remapped_records.append(remapped)
                records = remapped_records
                self._store.update_from_registry(records)
            time.sleep(0.5)

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()
        if self._registry_thread is not None:
            self._registry_thread.join(timeout=1.0)
            self._registry_thread = None
        for binding in self._bindings:
            binding.service.stop()


def _normalize_stream_name(stream_name: str | StreamName) -> str:
    if isinstance(stream_name, StreamName):
        return stream_name.value
    text = str(stream_name).strip().lower()
    for item in StreamName:
        if text == item.value:
            return item.value
    return text


class CameraHandle:
    def __init__(self, store: _SdkStore, key: tuple[str, str]) -> None:
        self._store = store
        self._key = key
        self._closed = False
        self._last_seq: dict[str, int] = {}

    def close(self) -> None:
        self._closed = True

    def _closed_result(self):
        return ReadStatus.STREAM_UNAVAILABLE, None, None

    def _maybe_rebind_key(self) -> None:
        if self._closed:
            return
        if self._key[0] != "unknown_sender":
            return
        try:
            better = self._store.resolve_camera_key(self._key[1])
        except Exception:
            return
        if better != self._key:
            self._key = better

    def _to_frame_meta(
        self,
        sample: Optional[_StreamSample],
        *,
        stream_name: str,
        status_code: Optional[ReadStatus] = None,
    ) -> Optional[FrameMeta]:
        if sample is None:
            return None
        return FrameMeta(
            sender_id=self._key[0],
            camera_id=self._key[1],
            stream_name=stream_name,
            frame_id=sample.frame_id,
            timestamp_us=sample.timestamp_us,
            online=True,
            status_code=_status_value(status_code) if status_code else None,
        )

    def get_status(self) -> CameraStatus:
        self._maybe_rebind_key()
        return self._store.get_status(self._key)

    def get_capabilities(self) -> CameraCapabilities:
        self._maybe_rebind_key()
        return self._store.get_capabilities(self._key)

    def get_stream_profiles(self) -> list[StreamProfile]:
        self._maybe_rebind_key()
        return self._store.get_profiles(self._key)

    def get_active_profile(self) -> list[StreamProfile]:
        self._maybe_rebind_key()
        return self._store.get_profiles(self._key)

    def get_calibration(self) -> tuple[ReadStatus, CalibrationInfo | None]:
        return ReadStatus.CALIBRATION_MISSING, None

    def read_stream(
        self,
        stream_name: str,
        timeout_ms: int = 100,
    ) -> tuple[ReadStatus, np.ndarray | None, FrameMeta | None]:
        if self._closed:
            return self._closed_result()
        self._maybe_rebind_key()
        stream = _normalize_stream_name(stream_name)
        min_seq = self._last_seq.get(stream)
        status, sample = self._store.wait_for_stream(self._key, stream, timeout_ms, min_seq=min_seq)
        if status in (ReadStatus.CAMERA_OFFLINE, ReadStatus.TIMEOUT):
            old_key = self._key
            self._maybe_rebind_key()
            if self._key != old_key:
                min_seq = self._last_seq.get(stream)
                status, sample = self._store.wait_for_stream(self._key, stream, min(80, timeout_ms), min_seq=min_seq)
        if status != ReadStatus.OK or sample is None:
            latest_status, latest = self._store.get_latest_stream(self._key, stream)
            meta = self._to_frame_meta(latest, stream_name=stream, status_code=status if latest_status == ReadStatus.OK else None)
            return status, None, meta
        self._last_seq[stream] = sample.seq
        return ReadStatus.OK, sample.data, self._to_frame_meta(sample, stream_name=stream)

    def get_latest_stream(
        self,
        stream_name: str,
    ) -> tuple[ReadStatus, np.ndarray | None, FrameMeta | None]:
        if self._closed:
            return self._closed_result()
        self._maybe_rebind_key()
        stream = _normalize_stream_name(stream_name)
        status, sample = self._store.get_latest_stream(self._key, stream)
        if status != ReadStatus.OK or sample is None:
            return status, None, None
        return ReadStatus.OK, sample.data, self._to_frame_meta(sample, stream_name=stream)

    def read_rgb(self, timeout_ms: int = 100) -> tuple[ReadStatus, np.ndarray | None, FrameMeta | None]:
        return self.read_stream(StreamName.RGB.value, timeout_ms=timeout_ms)

    def read_depth_raw(self, timeout_ms: int = 100) -> tuple[ReadStatus, np.ndarray | None, FrameMeta | None]:
        return self.read_stream(StreamName.DEPTH_RAW.value, timeout_ms=timeout_ms)

    def read_depth_aligned_to_rgb(self, timeout_ms: int = 100) -> tuple[ReadStatus, np.ndarray | None, FrameMeta | None]:
        return self.read_stream(StreamName.DEPTH_ALIGNED_TO_RGB.value, timeout_ms=timeout_ms)

    def read_depth_color(self, timeout_ms: int = 100) -> tuple[ReadStatus, np.ndarray | None, FrameMeta | None]:
        return self.read_stream(StreamName.DEPTH_COLOR.value, timeout_ms=timeout_ms)

    def get_latest_rgb(self) -> tuple[ReadStatus, np.ndarray | None, FrameMeta | None]:
        return self.get_latest_stream(StreamName.RGB.value)

    def get_latest_depth_raw(self) -> tuple[ReadStatus, np.ndarray | None, FrameMeta | None]:
        return self.get_latest_stream(StreamName.DEPTH_RAW.value)

    def get_latest_depth_aligned_to_rgb(self) -> tuple[ReadStatus, np.ndarray | None, FrameMeta | None]:
        return self.get_latest_stream(StreamName.DEPTH_ALIGNED_TO_RGB.value)

    def get_latest_depth_color(self) -> tuple[ReadStatus, np.ndarray | None, FrameMeta | None]:
        return self.get_latest_stream(StreamName.DEPTH_COLOR.value)

    def read_rgbd(
        self,
        timeout_ms: int = 100,
        mode: PairMode = PairMode.STRICT,
        aligned: bool = True,
        max_delta_ms: int | None = None,
        allow_partial: bool = False,
    ) -> tuple[ReadStatus, np.ndarray | None, np.ndarray | None, PairMeta | None]:
        if self._closed:
            return ReadStatus.STREAM_UNAVAILABLE, None, None, None
        self._maybe_rebind_key()
        pair_mode = mode if isinstance(mode, PairMode) else PairMode(str(mode).lower())
        depth_stream = StreamName.DEPTH_ALIGNED_TO_RGB.value if aligned else StreamName.DEPTH_RAW.value
        delta_limit = 33 if max_delta_ms is None else max(0, int(max_delta_ms))
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        last_rgb_seq = self._last_seq.get(StreamName.RGB.value)
        last_depth_seq = self._last_seq.get(depth_stream)
        partial_rgb: Optional[_StreamSample] = None
        partial_depth: Optional[_StreamSample] = None
        saw_both = False
        lookup = self._store.pair_lookup(
            self._key,
            depth_stream=depth_stream,
            mode=pair_mode,
            max_delta_ms=delta_limit,
            min_rgb_seq=last_rgb_seq,
            min_depth_seq=last_depth_seq,
        )
        if not lookup.depth_stream_available:
            if allow_partial:
                rgb_status, rgb, rgb_meta = self.read_rgb(timeout_ms=timeout_ms)
                if rgb_status == ReadStatus.OK and rgb_meta is not None:
                    meta = PairMeta(
                        sender_id=self._key[0],
                        camera_id=self._key[1],
                        rgb_frame_id=rgb_meta.frame_id,
                        depth_frame_id=0,
                        rgb_timestamp_us=rgb_meta.timestamp_us,
                        depth_timestamp_us=0,
                        pair_mode=pair_mode.value,
                        pair_complete=False,
                        aligned=aligned,
                        online=True,
                        pair_delta_ms=None,
                        drop_count=None,
                        status_code=ReadStatus.STREAM_UNAVAILABLE.value,
                    )
                    return ReadStatus.STREAM_UNAVAILABLE, rgb, None, meta
            return ReadStatus.STREAM_UNAVAILABLE, None, None, None
        while True:
            if lookup.matched and lookup.rgb is not None and lookup.depth is not None:
                self._last_seq[StreamName.RGB.value] = lookup.rgb.seq
                self._last_seq[depth_stream] = lookup.depth.seq
                meta = PairMeta(
                    sender_id=self._key[0],
                    camera_id=self._key[1],
                    rgb_frame_id=lookup.rgb.frame_id,
                    depth_frame_id=lookup.depth.frame_id,
                    rgb_timestamp_us=lookup.rgb.timestamp_us,
                    depth_timestamp_us=lookup.depth.timestamp_us,
                    pair_mode=pair_mode.value,
                    pair_complete=True,
                    aligned=aligned,
                    online=lookup.online,
                    pair_delta_ms=lookup.delta_ms,
                    drop_count=None,
                    status_code=None,
                )
                return ReadStatus.OK, lookup.rgb.data, lookup.depth.data, meta
            if lookup.rgb is not None:
                partial_rgb = lookup.rgb
            if lookup.depth is not None:
                partial_depth = lookup.depth
            saw_both = saw_both or lookup.saw_both
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                break
            self._store.wait_for_update(min(remaining_ms, 50))
            lookup = self._store.pair_lookup(
                self._key,
                depth_stream=depth_stream,
                mode=pair_mode,
                max_delta_ms=delta_limit,
                min_rgb_seq=last_rgb_seq,
                min_depth_seq=last_depth_seq,
            )
        if not allow_partial:
            if saw_both:
                return ReadStatus.PAIR_FAILED, None, None, None
            if not lookup.online and partial_rgb is None and partial_depth is None:
                return ReadStatus.CAMERA_OFFLINE, None, None, None
            return ReadStatus.TIMEOUT, None, None, None
        status = ReadStatus.PAIR_FAILED if saw_both else (ReadStatus.CAMERA_OFFLINE if not lookup.online else ReadStatus.TIMEOUT)
        meta = PairMeta(
            sender_id=self._key[0],
            camera_id=self._key[1],
            rgb_frame_id=partial_rgb.frame_id if partial_rgb is not None else 0,
            depth_frame_id=partial_depth.frame_id if partial_depth is not None else 0,
            rgb_timestamp_us=partial_rgb.timestamp_us if partial_rgb is not None else 0,
            depth_timestamp_us=partial_depth.timestamp_us if partial_depth is not None else 0,
            pair_mode=pair_mode.value,
            pair_complete=False,
            aligned=aligned,
            online=lookup.online,
            pair_delta_ms=lookup.delta_ms,
            drop_count=None,
            status_code=status.value,
        )
        return status, partial_rgb.data if partial_rgb is not None else None, partial_depth.data if partial_depth is not None else None, meta


def _default_config_path() -> Path:
    env = os.getenv("WVD_SDK_CONFIG")
    if env:
        return Path(env).expanduser()
    cwd_default = Path("config/receiver.json")
    if cwd_default.exists():
        return cwd_default
    package_default = Path(__file__).resolve().parents[1] / "config" / "receiver.json"
    return package_default


class Client:
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        auto_start: bool = True,
    ) -> None:
        self._store = _SdkStore()
        self._manager: Optional[_SdkReceiverManager] = None
        self._closed = False
        if auto_start:
            cfg_path = Path(config_path) if config_path is not None else _default_config_path()
            cfg = load_receiver_config(cfg_path)
            self._manager = _SdkReceiverManager(cfg, self._store)
            self._manager.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._manager is not None:
            self._manager.stop()
            self._manager = None

    def list_senders(self) -> list[SenderInfo]:
        return self._store.list_senders()

    def list_cameras(self) -> list[CameraInfo]:
        return self._store.list_cameras()

    def open_camera(self, camera_id: str) -> CameraHandle:
        text = str(camera_id).strip()
        if "/" in text:
            sender_id, cid = text.split("/", 1)
            key = self._store.resolve_camera_key(cid, sender_id=sender_id)
        else:
            key = self._store.resolve_camera_key(text)
        return CameraHandle(self._store, key)


_default_client_lock = threading.Lock()
_default_client: Optional[Client] = None


def _get_default_client() -> Client:
    global _default_client
    with _default_client_lock:
        if _default_client is None:
            _default_client = Client()
        return _default_client


def list_senders() -> list[SenderInfo]:
    return _get_default_client().list_senders()


def list_cameras() -> list[CameraInfo]:
    return _get_default_client().list_cameras()


def open_camera(camera_id: str) -> CameraHandle:
    return _get_default_client().open_camera(camera_id)
