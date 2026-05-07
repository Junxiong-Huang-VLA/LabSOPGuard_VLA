from __future__ import annotations

import argparse
import copy
import logging
import threading
import time
from typing import Callable

from .camera import CameraSource, list_connected_cameras
from .config import CodecConfig, SenderConfig, load_sender_config
from .encoder import Encoder
from .frame_queue import FrameQueue
from .models import VideoFrame
from .rtp import RtpH264Packetizer
from .state import RuntimeState
from .stats import StatsTracker
from .transport import TransportTx


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class SenderService:
    def __init__(
        self,
        cfg: SenderConfig,
        name: str = "cam0",
        profile_change_callback: Callable[[str, int, str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.name = name
        self.sender_id = cfg.sender_id or "sender0"
        self.camera_id = cfg.camera.camera_id or name
        self.channel_id = name
        self.camera = CameraSource()
        self.encoders: dict[str, Encoder] = {}
        self.packetizers: dict[str, RtpH264Packetizer] = {}
        self.transport = TransportTx()
        self.capture_queue: FrameQueue[VideoFrame] = FrameQueue(cfg.runtime.queue_size)
        self.stats = StatsTracker()
        self.state = RuntimeState()
        self.stop_event = threading.Event()
        self.capture_started = threading.Event()
        self._threads: list[threading.Thread] = []
        self._base_drop_divisor = max(1, cfg.runtime.drop_frame_divisor)
        self._active_drop_divisor = self._base_drop_divisor
        self._healthy_windows = 0
        self._reconnect_lock = threading.Lock()
        self._reconnect_done = threading.Event()
        self._reconnect_done.set()
        self._auto_fallback_resolution = bool(cfg.runtime.auto_fallback_resolution)
        self._auto_recover_resolution = bool(cfg.runtime.auto_recover_resolution)
        self._recover_window_sec = max(0.0, cfg.runtime.auto_recover_window_ms / 1000.0)
        self._recover_cooldown_sec = max(0.0, cfg.runtime.auto_recover_cooldown_ms / 1000.0)
        self._profile_ladder = self._build_profile_ladder(cfg.camera.width, cfg.camera.height, cfg.camera.fps)
        self._profile_index = 0
        now = time.monotonic()
        self._last_unhealthy_at = now
        self._last_congested_at = now
        self._last_tx_drop_at = now
        self._last_tx_drop_count = 0
        self._next_recover_at = now + self._recover_cooldown_sec
        self._profile_lock = threading.Lock()
        self._profile_change_callback = profile_change_callback

    def _open_pipeline(self) -> None:
        self.transport.open(
            self.cfg.network.remote_ip,
            self.cfg.network.remote_port,
            self.cfg.network.ttl,
            self.cfg.network.socket_buffer_bytes,
        )
        self.camera.start(self.cfg.camera)
        self.state.mark_success()
        self.stats.set_state(self.state.name)

    def _reconnect(self, error: Exception) -> None:
        if self.stop_event.is_set():
            return
        if not self._reconnect_lock.acquire(blocking=False):
            self._reconnect_done.wait(timeout=1.0)
            return
        self._reconnect_done.clear()
        try:
            self._last_unhealthy_at = time.monotonic()
            logging.warning("channel=%s reconnecting: %s", self.name, error)
            self.camera.stop()
            self.transport.close()
            self.state.enter_reconnecting()
            self.stats.set_state(self.state.name, str(error))
            self.stats.add_reconnect()
            while not self.stop_event.is_set():
                wait_ms = self.state.next_backoff_ms(self.cfg.runtime.reconnect_backoff_ms)
                if self.stop_event.wait(wait_ms / 1000.0):
                    return
                try:
                    self._open_pipeline()
                    self.encoders.clear()
                    self.packetizers.clear()
                    return
                except Exception as exc:
                    self._try_fallback_profile(f"RECONNECT_FAIL:{exc}")
                    logging.exception("channel=%s reconnect attempt failed", self.name)
                    self.state.enter_reconnecting()
                    self.stats.set_state(self.state.name, str(exc))
                    self.stats.add_reconnect()
        finally:
            self._reconnect_done.set()
            self._reconnect_lock.release()

    def _update_adaptive_drop(self, queue_depth: int, send_drop_count: int) -> None:
        if not self.cfg.runtime.adaptive_drop_enabled:
            self.stats.update_extra(drop_divisor=self._active_drop_divisor)
            return
        max_divisor = max(self._base_drop_divisor, self.cfg.runtime.adaptive_drop_max_divisor)
        # Only treat a full queue as congestion; near-full transient spikes
        # should not trigger aggressive frame dropping in quality-first mode.
        high_watermark = max(1, self.cfg.runtime.queue_size)
        congested = send_drop_count > 0 or queue_depth >= high_watermark
        if congested:
            self._last_congested_at = time.monotonic()
            self._active_drop_divisor = min(max_divisor, self._active_drop_divisor + 1)
            self._healthy_windows = 0
        else:
            self._healthy_windows += 1
            if (
                self._active_drop_divisor > self._base_drop_divisor
                and self._healthy_windows >= max(1, self.cfg.runtime.adaptive_drop_recover_window)
            ):
                self._active_drop_divisor -= 1
                self._healthy_windows = 0
        self.stats.update_extra(drop_divisor=self._active_drop_divisor)

    @staticmethod
    def _build_profile_ladder(width: int, height: int, fps: int) -> list[tuple[int, int, int]]:
        safe_fps = max(1, int(fps))
        base = (max(1, int(width)), max(1, int(height)), safe_fps)
        ladder = [base]
        base_pixels = base[0] * base[1]
        for cand_w, cand_h in ((1920, 1080), (1280, 720), (640, 480)):
            if cand_w * cand_h >= base_pixels:
                continue
            cand = (cand_w, cand_h, safe_fps)
            if cand not in ladder:
                ladder.append(cand)
        return ladder

    def profile_index(self) -> int:
        with self._profile_lock:
            return self._profile_index

    def _set_profile_index(self, profile_index: int, reason: str, action: str, notify: bool) -> bool:
        callback = None
        callback_index = 0
        callback_reason = ""
        with self._profile_lock:
            if not self._profile_ladder:
                return False
            target = max(0, min(int(profile_index), len(self._profile_ladder) - 1))
            if target == self._profile_index:
                return False

            self._profile_index = target
            width, height, fps = self._profile_ladder[self._profile_index]
            self.cfg.camera.width = width
            self.cfg.camera.height = height
            self.cfg.camera.fps = fps

            for encoder in self.encoders.values():
                encoder.close()
            self.encoders.clear()
            self.packetizers.clear()

            now = time.monotonic()
            self._last_unhealthy_at = now
            self._last_congested_at = now
            self._last_tx_drop_at = now
            self._next_recover_at = now + self._recover_cooldown_sec

            log_fn = logging.warning if action == "fallback" else logging.info
            log_fn(
                "channel=%s %s profile -> %dx%d@%d reason=%s",
                self.name,
                action,
                width,
                height,
                fps,
                reason,
            )

            if notify and self._profile_change_callback is not None:
                callback = self._profile_change_callback
                callback_index = self._profile_index
                callback_reason = f"{action}:{reason}"

        if callback is not None:
            try:
                callback(self.name, callback_index, callback_reason)
            except Exception:
                logging.exception("channel=%s profile change callback failed", self.name)
        return True

    def sync_profile_index(self, profile_index: int, reason: str) -> bool:
        changed = self._set_profile_index(profile_index, reason, action="sync", notify=False)
        if changed and self._threads and not self.stop_event.is_set():
            self._reconnect(RuntimeError("PROFILE_SYNC"))
        return changed

    def _try_fallback_profile(self, reason: str) -> bool:
        if not self._auto_fallback_resolution:
            return False
        return self._set_profile_index(self._profile_index + 1, reason, action="fallback", notify=True)

    def _try_recover_profile(self, reason: str) -> bool:
        if not self._auto_recover_resolution:
            return False
        if self._profile_index <= 0:
            return False
        now = time.monotonic()
        if now < self._next_recover_at:
            return False
        return self._set_profile_index(self._profile_index - 1, reason, action="recover", notify=True)

    def _codec_for_stream(self, stream_name: str) -> CodecConfig:
        if stream_name in ("depth_raw", "depth_aligned_to_rgb"):
            base = self.cfg.codec
            return CodecConfig(
                codec="libx264rgb",
                bitrate_kbps=max(base.bitrate_kbps, 4000),
                gop=base.gop,
                max_b_frames=0,
                preset=base.preset,
                tune=base.tune,
                thread_count=base.thread_count,
                thread_type=base.thread_type,
            )
        return self.cfg.codec

    def _get_encoder(self, stream_name: str, frame: VideoFrame) -> Encoder:
        encoder = self.encoders.get(stream_name)
        if encoder is not None:
            return encoder
        encoder = Encoder()
        codec_cfg = self._codec_for_stream(stream_name)
        encoder.open(codec_cfg, frame.width, frame.height, self.cfg.camera.fps)
        self.encoders[stream_name] = encoder
        return encoder

    def _get_packetizer(self, stream_name: str) -> RtpH264Packetizer:
        packetizer = self.packetizers.get(stream_name)
        if packetizer is not None:
            return packetizer
        packetizer = RtpH264Packetizer(self.cfg.network.mtu)
        self.packetizers[stream_name] = packetizer
        return packetizer

    @staticmethod
    def _stream_min_divisor(stream_name: str) -> int:
        # Keep depth bandwidth under control; RGB remains real-time first.
        if stream_name in ("depth_raw", "depth_aligned_to_rgb"):
            return 4
        if stream_name == "depth_color":
            return 2
        return 1

    def capture_thread(self) -> None:
        while not self.stop_event.is_set():
            try:
                frames = self.camera.read_bundle(self.cfg.runtime.camera_timeout_ms)
                if not frames:
                    self.state.mark_failure(self.cfg.runtime.degraded_threshold)
                    self.stats.set_state(self.state.name, "CAM_TIMEOUT")
                    self._last_unhealthy_at = time.monotonic()
                    if self.state.name == "DEGRADED" and self._try_fallback_profile("CAM_TIMEOUT"):
                        self._reconnect(RuntimeError("CAM_TIMEOUT"))
                    continue
                self.capture_started.set()
                rgb_seen = False
                for frame in frames:
                    self.capture_queue.push_latest(frame)
                    if frame.stream_name == "rgb":
                        rgb_seen = True
                if rgb_seen:
                    self.stats.mark_in()
                self.stats.set_queue_depth(self.capture_queue.size())
                self.stats.set_drop_count(self.capture_queue.drops)
                self.state.mark_success()
                self.stats.set_state(self.state.name)
            except Exception as exc:
                logging.exception("channel=%s capture failed", self.name)
                if self.stop_event.is_set():
                    break
                self._last_unhealthy_at = time.monotonic()
                self._reconnect(exc)

    def encode_send_thread(self) -> None:
        frame_count_by_stream: dict[str, int] = {}
        while not self.stop_event.is_set():
            frame = self.capture_queue.pop(100)
            if frame is None:
                continue
            stream_name = frame.stream_name or "rgb"
            encoder = self._get_encoder(stream_name, frame)
            frame_count = frame_count_by_stream.get(stream_name, 0) + 1
            frame_count_by_stream[stream_name] = frame_count
            divisor = max(self._active_drop_divisor, self._stream_min_divisor(stream_name))
            if divisor > 1 and frame_count % divisor != 0:
                continue
            try:
                packet_count = 0
                send_drop_count = 0
                packetizer = self._get_packetizer(stream_name)
                for packet in encoder.encode(
                    frame,
                    sender_id=self.sender_id,
                    camera_id=self.camera_id,
                    channel_id=self.channel_id,
                    stream_name=stream_name,
                ):
                    rtp_packets = packetizer.packetize(packet)
                    for raw in rtp_packets:
                        if self.transport.send(raw):
                            self.stats.mark_packet(len(raw))
                            packet_count += 1
                        else:
                            send_drop_count += 1
                self.stats.add_tx_drop_count(send_drop_count)
                if packet_count > 0 and stream_name == "rgb":
                    # fps_out tracks encoded frame output rate.
                    self.stats.mark_out(0)
                queue_depth = self.capture_queue.size()
                self.stats.set_queue_depth(queue_depth)
                self._update_adaptive_drop(queue_depth, send_drop_count)
            except Exception as exc:
                logging.exception("channel=%s encode/send failed", self.name)
                if self.stop_event.is_set():
                    break
                self._reconnect(exc)

    def watchdog_thread(self) -> None:
        last_log = 0.0
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now - last_log >= self.cfg.runtime.log_interval_ms / 1000.0:
                stat = self.stats.snapshot()
                if stat.tx_drop_count > self._last_tx_drop_count:
                    self._last_tx_drop_count = stat.tx_drop_count
                    self._last_tx_drop_at = now
                active_divisor = int(stat.extra.get("drop_divisor", self._active_drop_divisor))
                logging.info(
                    "channel=%s port=%d state=%s fps_in=%.1f fps_out=%.1f pkt_rate=%.1f bitrate=%.0fkbps queue=%d drops=%d tx_drop=%d div=%d reconnect=%d",
                    self.name,
                    self.cfg.network.remote_port,
                    stat.state,
                    stat.fps_in,
                    stat.fps_out,
                    stat.packet_rate,
                    stat.bitrate_kbps,
                    stat.queue_depth,
                    stat.drop_count,
                    stat.tx_drop_count,
                    active_divisor,
                    stat.reconnect_count,
                )
                recover_ready = (
                    self._profile_index > 0
                    and self._auto_recover_resolution
                    and stat.state == "RUNNING"
                    and self.state.name == "RUNNING"
                    and self._active_drop_divisor <= self._base_drop_divisor
                    and now >= self._next_recover_at
                    and (now - self._last_unhealthy_at) >= self._recover_window_sec
                    and (now - self._last_congested_at) >= self._recover_window_sec
                    and (now - self._last_tx_drop_at) >= self._recover_window_sec
                )
                if recover_ready and self._try_recover_profile("HEALTHY_WINDOW"):
                    self._reconnect(RuntimeError("PROFILE_RECOVER"))
                last_log = now
            time.sleep(self.cfg.runtime.watchdog_interval_ms / 1000.0)

    def start(self) -> None:
        if self._threads:
            return
        self.stop_event.clear()
        self.capture_started.clear()
        self._base_drop_divisor = max(1, self.cfg.runtime.drop_frame_divisor)
        self._active_drop_divisor = self._base_drop_divisor
        self._healthy_windows = 0
        now = time.monotonic()
        self._last_unhealthy_at = now
        self._last_congested_at = now
        self._last_tx_drop_at = now
        self._last_tx_drop_count = 0
        self._next_recover_at = now + self._recover_cooldown_sec
        self._reconnect_done.set()
        try:
            self._open_pipeline()
        except Exception:
            self.camera.stop()
            self.transport.close()
            for encoder in self.encoders.values():
                encoder.close()
            self.encoders.clear()
            self.packetizers.clear()
            raise
        self._threads = [
            threading.Thread(target=self.capture_thread, name=f"{self.name}_capture"),
            threading.Thread(target=self.encode_send_thread, name=f"{self.name}_encode_send"),
            threading.Thread(target=self.watchdog_thread, name=f"{self.name}_watchdog"),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()
        self.camera.stop()
        self.transport.close()
        for encoder in self.encoders.values():
            encoder.close()
        self.encoders.clear()
        self.packetizers.clear()

    def run(self) -> int:
        self.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        return 0


class MultiSenderService:
    def __init__(self, cfg: SenderConfig) -> None:
        self.cfg = cfg
        self._port_step = max(1, cfg.network.port_step)
        self._auto_discover = bool(cfg.runtime.auto_discover_cameras and not cfg.cameras)
        self._discover_interval_sec = max(0.5, cfg.runtime.auto_discover_interval_ms / 1000.0)
        self._sync_profile_across_channels = bool(cfg.runtime.sync_profile_across_channels)
        self._profile_sync_lock = threading.Lock()
        self.services = self._build_static_services(cfg) if not self._auto_discover else []
        self._dynamic_services: dict[str, SenderService] = {}
        self._channel_names: dict[str, str] = {}
        self._channel_ports: dict[str, int] = {}
        self._next_slot = 0

    def _build_static_services(self, cfg: SenderConfig) -> list[SenderService]:
        if cfg.cameras:
            cameras = [camera for camera in cfg.cameras if camera.enabled]
        else:
            cameras = [cfg.camera]
        services: list[SenderService] = []
        port_step = max(1, cfg.network.port_step)
        for index, camera in enumerate(cameras):
            channel_cfg = copy.deepcopy(cfg)
            channel_cfg.cameras = []
            channel_cfg.camera = copy.deepcopy(camera)
            channel_cfg.network.remote_port = (
                camera.remote_port if camera.remote_port is not None else cfg.network.remote_port + index * port_step
            )
            channel_name = camera.camera_id or f"cam{index}"
            if not channel_cfg.camera.camera_id:
                channel_cfg.camera.camera_id = channel_name
            services.append(
                SenderService(
                    channel_cfg,
                    name=channel_name,
                    profile_change_callback=self._on_profile_change,
                )
            )
        return services

    @staticmethod
    def _device_key(device: dict[str, str]) -> str:
        serial = (device.get("serial_number") or "").strip()
        if serial:
            return f"sn:{serial}"
        uid = (device.get("uid") or "").strip()
        if uid:
            return f"uid:{uid}"
        index = (device.get("index") or "").strip()
        return f"idx:{index}"

    def _allocate_slot(self, key: str) -> tuple[str, int]:
        if key in self._channel_names and key in self._channel_ports:
            return self._channel_names[key], self._channel_ports[key]
        slot = self._next_slot
        self._next_slot += 1
        channel_name = f"cam{slot}"
        remote_port = self.cfg.network.remote_port + slot * self._port_step
        self._channel_names[key] = channel_name
        self._channel_ports[key] = remote_port
        return channel_name, remote_port

    def _build_dynamic_service(self, device: dict[str, str]) -> tuple[str, SenderService]:
        key = self._device_key(device)
        channel_name, remote_port = self._allocate_slot(key)
        channel_cfg = copy.deepcopy(self.cfg)
        channel_cfg.cameras = []
        camera_cfg = copy.deepcopy(self.cfg.camera)
        camera_cfg.camera_id = channel_name

        serial = (device.get("serial_number") or "").strip()
        uid = (device.get("uid") or "").strip()
        index_raw = (device.get("index") or "").strip()

        camera_cfg.device_index = None
        camera_cfg.serial_number = serial or None
        camera_cfg.device_uid = uid or None
        if camera_cfg.serial_number is None and camera_cfg.device_uid is None:
            try:
                camera_cfg.device_index = int(index_raw)
            except Exception:
                camera_cfg.device_index = None

        channel_cfg.camera = camera_cfg
        channel_cfg.network.remote_port = remote_port
        return key, SenderService(channel_cfg, name=channel_name, profile_change_callback=self._on_profile_change)

    def _iter_sync_services(self) -> list[SenderService]:
        if self._auto_discover:
            return list(self._dynamic_services.values())
        return list(self.services)

    def _target_profile_index(self, exclude: str = "") -> int | None:
        target = None
        for service in self._iter_sync_services():
            if exclude and service.name == exclude:
                continue
            idx = service.profile_index()
            target = idx if target is None else max(target, idx)
        return target

    def _on_profile_change(self, source_name: str, profile_index: int, reason: str) -> None:
        if not self._sync_profile_across_channels:
            return
        if not self._profile_sync_lock.acquire(blocking=False):
            return
        try:
            targets = [service for service in self._iter_sync_services() if service.name != source_name]
        finally:
            self._profile_sync_lock.release()
        for service in targets:
            service.sync_profile_index(profile_index, f"SYNC_FROM_{source_name}:{reason}")

    def _start_service(self, service: SenderService) -> bool:
        try:
            service.start()
            cam = service.cfg.camera
            logging.info(
                "channel=%s sender=%s camera=%s started device(index=%s serial=%s uid=%s) remote=%s:%d",
                service.name,
                service.sender_id,
                service.camera_id,
                cam.device_index,
                cam.serial_number or "<empty>",
                cam.device_uid or "<empty>",
                service.cfg.network.remote_ip,
                service.cfg.network.remote_port,
            )
            return True
        except Exception:
            logging.exception("channel=%s start failed", service.name)
            return False

    def _run_static(self) -> int:
        started: list[SenderService] = []
        for service in self.services:
            if self._start_service(service):
                started.append(service)
                if self._sync_profile_across_channels:
                    target = self._target_profile_index(exclude=service.name)
                    if target is not None:
                        service.sync_profile_index(target, "JOIN_SYNC_STATIC")
        if not started:
            logging.error("no sender channel started")
            return 1
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            for service in started:
                service.stop()
        return 0

    def _reconcile_dynamic_services(self) -> None:
        try:
            devices = list_connected_cameras()
        except Exception:
            logging.exception("auto-discover query failed")
            return
        desired: dict[str, dict[str, str]] = {}
        for device in devices:
            key = self._device_key(device)
            # Keep the first occurrence for a key to avoid duplicate launches.
            desired.setdefault(key, device)

        for key, device in desired.items():
            if key in self._dynamic_services:
                continue
            dynamic_key, service = self._build_dynamic_service(device)
            if self._start_service(service):
                self._dynamic_services[dynamic_key] = service
                if self._sync_profile_across_channels:
                    target = self._target_profile_index(exclude=service.name)
                    if target is not None:
                        service.sync_profile_index(target, "JOIN_SYNC_DYNAMIC")

        stale_keys = [key for key in self._dynamic_services if key not in desired]
        for key in stale_keys:
            service = self._dynamic_services.pop(key)
            logging.info("channel=%s removed: device offline", service.name)
            service.stop()

    def _run_dynamic(self) -> int:
        logging.info(
            "auto-discover enabled: base_port=%d port_step=%d interval_ms=%d",
            self.cfg.network.remote_port,
            self._port_step,
            self.cfg.runtime.auto_discover_interval_ms,
        )
        try:
            while True:
                self._reconcile_dynamic_services()
                if not self._dynamic_services:
                    logging.warning("auto-discover: no available cameras")
                time.sleep(self._discover_interval_sec)
        except KeyboardInterrupt:
            pass
        finally:
            for service in list(self._dynamic_services.values()):
                service.stop()
            self._dynamic_services.clear()
        return 0

    def run(self) -> int:
        if self._auto_discover:
            return self._run_dynamic()
        return self._run_static()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini RTP sender")
    parser.add_argument("--config", required=True, help="sender json config")
    args = parser.parse_args()
    cfg = load_sender_config(args.config)
    if cfg.cameras or cfg.runtime.auto_discover_cameras:
        return MultiSenderService(cfg).run()
    service = SenderService(cfg, name=cfg.camera.camera_id or "cam0")
    return service.run()
