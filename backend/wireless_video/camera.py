from __future__ import annotations

import logging
import time
from typing import Any, Optional

import cv2
import numpy as np
try:
    from pyorbbecsdk import (
        AlignFilter,
        Config,
        Context,
        OBAlignMode,
        OBError,
        OBFormat,
        OBPropertyID,
        OBSensorType,
        OBStreamType,
        Pipeline,
    )
    _SDK_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only in SDK-missing environments
    _SDK_IMPORT_ERROR = exc

    class _DummyEnum:
        RGB = "RGB"
        BGR = "BGR"
        MJPG = "MJPG"
        YUYV = "YUYV"
        NV12 = "NV12"
        NV21 = "NV21"
        UYVY = "UYVY"
        Y16 = "Y16"
        DISABLE = "DISABLE"
        COLOR_SENSOR = "COLOR_SENSOR"
        DEPTH_SENSOR = "DEPTH_SENSOR"
        COLOR_STREAM = "COLOR_STREAM"

    class _DummyFilter:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(f"pyorbbecsdk unavailable: {_SDK_IMPORT_ERROR}")

    AlignFilter = _DummyFilter  # type: ignore[assignment]
    Config = Context = OBError = OBPropertyID = OBSensorType = OBStreamType = Pipeline = object  # type: ignore[assignment]
    OBFormat = OBAlignMode = _DummyEnum()  # type: ignore[assignment]

from .config import CameraConfig
from .models import VideoFrame

LOG = logging.getLogger(__name__)


def list_connected_cameras() -> list[dict[str, str]]:
    context = Context()
    device_list = context.query_devices()
    devices: list[dict[str, str]] = []
    for index in range(device_list.get_count()):
        device = device_list.get_device_by_index(index)
        info = device.get_device_info()
        devices.append(
            {
                "index": str(index),
                "name": info.get_name() or "",
                "serial_number": info.get_serial_number() or "",
                "uid": info.get_uid() or "",
                "connection_type": info.get_connection_type() or "",
            }
        )
    return devices


def _reshape_u8(data: np.ndarray, expected_size: int, shape: tuple[int, ...]) -> Optional[np.ndarray]:
    if data.size < expected_size:
        return None
    return data[:expected_size].reshape(shape)


def _frame_to_image(frame) -> Optional[tuple[np.ndarray, str]]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    raw = frame.get_data()
    try:
        data = np.frombuffer(raw, dtype=np.uint8)
    except TypeError:
        data = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if color_format == OBFormat.RGB:
        image = _reshape_u8(data, width * height * 3, (height, width, 3))
        if image is None:
            return None
        # Keep RGB as-is to avoid per-frame color-space conversion cost.
        return image, "RGB8"
    if color_format == OBFormat.BGR:
        image = _reshape_u8(data, width * height * 3, (height, width, 3))
        if image is None:
            return None
        return image, "BGR8"
    if color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            return None
        return image, "BGR8"
    if color_format == OBFormat.YUYV:
        image = _reshape_u8(data, width * height * 2, (height, width, 2))
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2), "BGR8"
    if color_format == OBFormat.NV12:
        image = _reshape_u8(data, width * height * 3 // 2, (height * 3 // 2, width))
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV12), "BGR8"
    if color_format == OBFormat.NV21:
        image = _reshape_u8(data, width * height * 3 // 2, (height * 3 // 2, width))
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_NV21), "BGR8"
    if color_format == OBFormat.UYVY:
        image = _reshape_u8(data, width * height * 2, (height, width, 2))
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY), "BGR8"
    return None


def _frame_timestamp_ms(frame) -> int:
    try:
        system_ts_us = int(frame.get_system_timestamp_us())
        if system_ts_us > 0:
            return system_ts_us // 1000
    except Exception:
        pass
    try:
        ts_us = int(frame.get_timestamp_us())
        if ts_us > 0:
            return ts_us // 1000
    except Exception:
        pass
    return int(time.time() * 1000)


def _depth_frame_to_u16(depth_frame) -> Optional[np.ndarray]:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    raw = depth_frame.get_data()
    try:
        data = np.frombuffer(raw, dtype=np.uint16)
    except TypeError:
        data = np.asarray(raw, dtype=np.uint16).reshape(-1)
    expected = width * height
    if data.size < expected:
        return None
    return data[:expected].reshape((height, width)).copy()


def _depth_to_color(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
    if depth_u16.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    depth_mm = depth_u16.astype(np.float32) * max(float(depth_scale), 1e-6)
    valid = depth_mm > 0
    if np.any(valid):
        vals = depth_mm[valid]
        lo = float(np.percentile(vals, 2.0))
        hi = float(np.percentile(vals, 98.0))
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((depth_mm - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        norm[~valid] = 0
    else:
        norm = np.zeros(depth_u16.shape, dtype=np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def _pack_depth_u16_to_bgr(depth_u16: np.ndarray) -> np.ndarray:
    low = (depth_u16 & 0xFF).astype(np.uint8)
    high = ((depth_u16 >> 8) & 0xFF).astype(np.uint8)
    zero = np.zeros_like(low, dtype=np.uint8)
    return np.dstack((low, high, zero))


class CameraSource:
    def __init__(self) -> None:
        self._context: Optional[Context] = None
        self._pipeline: Optional[Pipeline] = None
        self._align_filter: Optional[AlignFilter] = None
        self._stream_seq: dict[str, int] = {}
        self._depth_enabled: bool = False
        self._depth_active: bool = False

    def _next_frame_id(self, stream_name: str) -> int:
        seq = self._stream_seq.get(stream_name, 0) + 1
        self._stream_seq[stream_name] = seq
        return seq

    def start(self, cfg: CameraConfig) -> bool:
        if _SDK_IMPORT_ERROR is not None:
            raise RuntimeError(f"pyorbbecsdk import failed: {_SDK_IMPORT_ERROR}") from _SDK_IMPORT_ERROR
        self.stop()
        self._context = Context()
        self._stream_seq.clear()
        self._depth_enabled = bool(cfg.enable_depth_streams)
        self._depth_active = False
        device_list = self._context.query_devices()
        candidates: list[tuple[int, str, str, str, Any]] = []
        for index in range(device_list.get_count()):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            serial = info.get_serial_number() or ""
            connection = info.get_connection_type() or ""
            uid = info.get_uid() or ""

            if cfg.device_index is not None and cfg.device_index != index:
                continue
            if cfg.serial_number and cfg.serial_number != serial:
                continue
            if cfg.device_uid and cfg.device_uid != uid:
                continue
            candidates.append((index, serial, uid, connection, device))

        if not (cfg.device_index is not None or cfg.serial_number or cfg.device_uid):
            candidates.sort(key=lambda item: (0 if item[1] else 1, 0 if "USB3" in item[3].upper() else 1, item[0]))
        if not candidates:
            filters = f"index={cfg.device_index}, serial={cfg.serial_number}, uid={cfg.device_uid}"
            available = list_connected_cameras()
            available_desc = "; ".join(
                [
                    f"index={it['index']} serial={it['serial_number'] or '<empty>'} uid={it['uid'] or '<empty>'} conn={it['connection_type'] or '<unknown>'}"
                    for it in available
                ]
            ) or "none"
            raise RuntimeError(f"No Orbbec device matches filters ({filters}), available: {available_desc}")

        last_exc: Optional[Exception] = None
        for index, serial, uid, connection, device in candidates:
            label = (
                f"index={index} serial={serial or '<empty>'} "
                f"uid={uid or '<empty>'} conn={connection or '<unknown>'}"
            )
            try:
                pipeline = Pipeline(device)
                self._start_pipeline(pipeline, cfg)
                self._pipeline = pipeline
                LOG.info("camera selected: %s", label)
                return True
            except Exception as exc:
                last_exc = exc
                LOG.warning("camera start failed on %s: %s", label, exc)
                try:
                    pipeline.stop()
                except Exception:
                    pass

        raise RuntimeError(f"No usable Orbbec device found: {last_exc}")

    def _start_pipeline(self, pipeline: Pipeline, cfg: CameraConfig) -> None:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        selected_color = None
        formats = {
            "RGB": OBFormat.RGB,
            "MJPG": OBFormat.MJPG,
            "YUYV": OBFormat.YUYV,
            "NV12": OBFormat.NV12,
            "NV21": OBFormat.NV21,
            "UYVY": OBFormat.UYVY,
        }
        for name in cfg.format_priority:
            try:
                selected_color = color_profiles.get_video_stream_profile(cfg.width, cfg.height, formats[name], cfg.fps)
                break
            except Exception:
                continue
        if selected_color is None:
            selected_color = color_profiles.get_default_video_stream_profile()

        selected_depth = None
        if self._depth_enabled:
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                selected_depth = depth_profiles.get_video_stream_profile(cfg.width, cfg.height, OBFormat.Y16, cfg.fps)
            except Exception:
                try:
                    selected_depth = depth_profiles.get_default_video_stream_profile()
                except Exception:
                    selected_depth = None

        config = Config()
        config.enable_stream(selected_color)
        if selected_depth is not None:
            config.enable_stream(selected_depth)
            self._depth_active = True
        elif self._depth_enabled:
            LOG.warning("depth stream requested but unavailable; fallback to RGB-only")
        try:
            config.set_align_mode(OBAlignMode.DISABLE)
        except Exception:
            pass
        pipeline.start(config)
        if self._depth_active:
            try:
                self._align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            except Exception as exc:
                self._align_filter = None
                LOG.warning("depth align filter init failed: %s", exc)
        else:
            self._align_filter = None
        self._apply_color_controls(cfg, pipeline)

    def _apply_color_controls(self, cfg: CameraConfig, pipeline: Optional[Pipeline] = None) -> None:
        active_pipeline = pipeline or self._pipeline
        if active_pipeline is None:
            return
        try:
            device = active_pipeline.get_device()
            device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, cfg.auto_exposure)
            if cfg.auto_exposure:
                return
            if cfg.exposure is not None:
                exposure = self._clamp_int_property(
                    device,
                    OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
                    cfg.exposure,
                )
                device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, exposure)
            if cfg.gain is not None:
                gain = self._clamp_int_property(
                    device,
                    OBPropertyID.OB_PROP_COLOR_GAIN_INT,
                    cfg.gain,
                )
                device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, gain)
        except Exception as exc:
            LOG.warning("apply color controls failed: %s", exc)

    @staticmethod
    def _clamp_int_property(device, property_id, requested: int) -> int:
        range_info = device.get_int_property_range(property_id)
        value = max(range_info.min, min(range_info.max, requested))
        if range_info.step > 1:
            value = range_info.min + ((value - range_info.min) // range_info.step) * range_info.step
        return int(value)

    def read_bundle(self, timeout_ms: int = 100) -> list[VideoFrame]:
        if self._pipeline is None:
            return []
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return []
        out: list[VideoFrame] = []

        color_frame = frames.get_color_frame()
        if color_frame is not None:
            converted = _frame_to_image(color_frame)
            if converted is not None:
                image, pixel_fmt = converted
                out.append(
                    VideoFrame(
                        frame_id=self._next_frame_id("rgb"),
                        capture_ts_ms=_frame_timestamp_ms(color_frame),
                        width=image.shape[1],
                        height=image.shape[0],
                        pixel_fmt=pixel_fmt,
                        data=image,
                        stream_name="rgb",
                    )
                )

        if not self._depth_active:
            return out

        depth_frame = frames.get_depth_frame()
        depth_raw: Optional[np.ndarray] = None
        depth_scale = 1.0
        depth_ts_ms = int(time.time() * 1000)
        if depth_frame is not None:
            depth_raw = _depth_frame_to_u16(depth_frame)
            try:
                depth_scale = float(depth_frame.get_depth_scale())
            except Exception:
                depth_scale = 1.0
            depth_ts_ms = _frame_timestamp_ms(depth_frame)

        depth_aligned: Optional[np.ndarray] = None
        if self._align_filter is not None and depth_frame is not None and color_frame is not None:
            try:
                aligned = self._align_filter.process(frames)
                if aligned is not None:
                    aligned_set = aligned.as_frame_set()
                    aligned_depth = aligned_set.get_depth_frame()
                    if aligned_depth is not None:
                        depth_aligned = _depth_frame_to_u16(aligned_depth)
            except Exception:
                depth_aligned = None

        if depth_raw is not None:
            packed_raw = _pack_depth_u16_to_bgr(depth_raw)
            out.append(
                VideoFrame(
                    frame_id=self._next_frame_id("depth_raw"),
                    capture_ts_ms=depth_ts_ms,
                    width=packed_raw.shape[1],
                    height=packed_raw.shape[0],
                    pixel_fmt="BGR8",
                    data=packed_raw,
                    stream_name="depth_raw",
                )
            )

        if depth_aligned is not None:
            packed_aligned = _pack_depth_u16_to_bgr(depth_aligned)
            out.append(
                VideoFrame(
                    frame_id=self._next_frame_id("depth_aligned_to_rgb"),
                    capture_ts_ms=depth_ts_ms,
                    width=packed_aligned.shape[1],
                    height=packed_aligned.shape[0],
                    pixel_fmt="BGR8",
                    data=packed_aligned,
                    stream_name="depth_aligned_to_rgb",
                )
            )

        depth_for_color = depth_aligned if depth_aligned is not None else depth_raw
        if depth_for_color is not None:
            depth_color = _depth_to_color(depth_for_color, depth_scale)
            out.append(
                VideoFrame(
                    frame_id=self._next_frame_id("depth_color"),
                    capture_ts_ms=depth_ts_ms,
                    width=depth_color.shape[1],
                    height=depth_color.shape[0],
                    pixel_fmt="BGR8",
                    data=depth_color,
                    stream_name="depth_color",
                )
            )
        return out

    def read(self, timeout_ms: int = 100) -> Optional[VideoFrame]:
        frames = self.read_bundle(timeout_ms)
        if not frames:
            return None
        for frame in frames:
            if frame.stream_name == "rgb":
                return frame
        return frames[0]

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except OBError:
                pass
        self._pipeline = None
        self._context = None
        self._align_filter = None
        self._depth_enabled = False
        self._depth_active = False
