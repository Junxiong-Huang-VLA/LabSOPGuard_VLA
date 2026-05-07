from __future__ import annotations

import argparse
import copy
import heapq
import logging
import threading
import time

from .config import ReceiverConfig, load_receiver_config
from .decoder import Decoder
from .frame_queue import FrameQueue
from .metrics import start_metrics_server_from_env
from .models import FrameAssembly, VideoFrame
from .renderer import MonitorWallRenderer, Renderer
from .state import RuntimeState
from .stats import StatsTracker
from .transport import TransportRx


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class DeviceRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str, str], dict[str, object]] = {}

    def upsert(
        self,
        *,
        sender_id: str,
        camera_id: str,
        stream_channel_id: str,
        receiver_channel: str,
        listen_port: int,
        state: str,
        last_error: str,
        packets_rx: int,
        fps_in: float,
        fps_out: float,
    ) -> None:
        sid = sender_id or "unknown_sender"
        cid = camera_id or receiver_channel
        chid = stream_channel_id or receiver_channel
        key = (sid, cid, chid)
        now_ms = int(time.time() * 1000)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                record = {
                    "sender_id": sid,
                    "camera_id": cid,
                    "stream_channel_id": chid,
                    "first_seen_ms": now_ms,
                }
            record.update(
                {
                    "receiver_channel": receiver_channel,
                    "listen_port": listen_port,
                    "state": state,
                    "last_error": last_error,
                    "packets_rx": packets_rx,
                    "fps_in": round(fps_in, 2),
                    "fps_out": round(fps_out, 2),
                    "last_seen_ms": now_ms,
                }
            )
            self._records[key] = record

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            values = [dict(item) for item in self._records.values()]
        values.sort(key=lambda item: (str(item["sender_id"]), str(item["camera_id"]), str(item["stream_channel_id"])))
        return values


class ReceiverService:
    def __init__(
        self,
        cfg: ReceiverConfig,
        name: str = "rx0",
        window_name: str | None = None,
        show_stats: bool | None = None,
        registry: DeviceRegistry | None = None,
        enable_renderer: bool = True,
        expected_stream_name: str | None = "rgb",
        allowed_stream_names: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.name = name
        self.registry = registry
        self.transport = TransportRx()
        self.decoder = Decoder()
        effective_window_name = window_name or cfg.display.window_name
        effective_show_stats = cfg.display.show_stats if show_stats is None else show_stats
        self._base_window_name = effective_window_name
        self._show_stats = effective_show_stats
        self._enable_renderer = enable_renderer
        self.renderer: Renderer | None = None
        self._renderers: dict[str, Renderer] = {}
        self.decoded_queue: FrameQueue[VideoFrame] = FrameQueue(cfg.runtime.queue_size)
        self.assembly_heap: list[tuple[int, int, FrameAssembly]] = []
        self.heap_lock = threading.Lock()
        self.stats = StatsTracker()
        self.state = RuntimeState()
        self.stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_lost_packets = 0
        self._last_render_monotonic = time.monotonic()
        self._last_rx_for_stall = 0
        self._sender_id = ""
        self._camera_id = ""
        self._stream_channel_id = ""
        requested_stream = (expected_stream_name or "").strip().lower()
        self._expected_stream_name = requested_stream
        self._allowed_stream_names = {
            item.strip().lower()
            for item in (allowed_stream_names or [])
            if item and item.strip()
        }
        self._stream_name = requested_stream or "rgb"
        self._latest_by_stream: dict[str, VideoFrame] = {}
        self._latest_frame_monotonic = 0.0
        if self._enable_renderer and self._expected_stream_name:
            self.renderer = Renderer(self._base_window_name, self._show_stats)
            self._renderers[self._expected_stream_name] = self.renderer

    @property
    def expected_stream_name(self) -> str:
        return self._expected_stream_name

    def _renderer_key(self, stream_name: str) -> str:
        key = (stream_name or "rgb").strip().lower()
        return key or "rgb"

    def _renderer_window_name(self, stream_name: str) -> str:
        if self._expected_stream_name:
            return self._base_window_name
        return f"{self._base_window_name} [{stream_name}]"

    def _get_renderer(self, stream_name: str) -> Renderer | None:
        if not self._enable_renderer:
            return None
        key = self._renderer_key(stream_name)
        renderer = self._renderers.get(key)
        if renderer is not None:
            return renderer
        renderer = Renderer(self._renderer_window_name(key), self._show_stats)
        self._renderers[key] = renderer
        if self.renderer is None:
            self.renderer = renderer
        return renderer

    def _accept_stream(self, stream_name: str) -> bool:
        incoming = stream_name.strip().lower()
        if not incoming:
            incoming = "rgb"
        if self._expected_stream_name:
            return incoming == self._expected_stream_name
        if self._allowed_stream_names:
            return incoming in self._allowed_stream_names
        return True

    def _record_latest_frame(self, frame: VideoFrame) -> None:
        stream_name = (frame.stream_name or "rgb").strip().lower() or "rgb"
        self._latest_by_stream[stream_name] = frame
        self._latest_frame_monotonic = time.monotonic()

    def get_monitor_frame(
        self,
        preferred_streams: tuple[str, ...] = ("rgb", "depth_color"),
        stale_timeout_s: float = 3.0,
    ) -> tuple[VideoFrame | None, str]:
        for _ in range(16):
            item = self.decoded_queue.pop(0)
            if item is None:
                break
            self._record_latest_frame(item)
        if self._latest_frame_monotonic <= 0:
            return None, ""
        if time.monotonic() - self._latest_frame_monotonic > stale_timeout_s:
            return None, ""
        for stream_name in preferred_streams:
            key = stream_name.strip().lower()
            frame = self._latest_by_stream.get(key)
            if frame is not None:
                return frame, key
        if self._latest_by_stream:
            frame = max(self._latest_by_stream.values(), key=lambda item: int(item.capture_ts_ms))
            stream_name = (frame.stream_name or "rgb").strip().lower() or "rgb"
            return frame, stream_name
        return None, ""

    def _open(self) -> None:
        self.transport.open(
            self.cfg.network.listen_ip,
            self.cfg.network.listen_port,
            self.cfg.network.socket_buffer_bytes,
            self.cfg.runtime.recv_timeout_ms,
            self.cfg.network.depacketizer_timeout_ms,
            self.cfg.network.depacketizer_max_frames,
            self.cfg.network.depacketizer_max_fus,
        )
        self.decoder.open()
        self.state.mark_success()
        self.stats.set_state(self.state.name)

    def recv_thread(self) -> None:
        while not self.stop_event.is_set():
            try:
                pkt = self.transport.recv()
                if pkt is None:
                    self.state.mark_failure(self.cfg.runtime.degraded_threshold)
                    self.stats.set_state(self.state.name, "NET_RECV_TIMEOUT")
                    continue
                if not self._accept_stream(pkt.stream_name or ""):
                    continue
                self.stats.add_packets_rx()
                self.stats.mark_packet(len(pkt.payload))
                if pkt.sender_id:
                    self._sender_id = pkt.sender_id
                if pkt.camera_id:
                    self._camera_id = pkt.camera_id
                if pkt.channel_id:
                    self._stream_channel_id = pkt.channel_id
                if pkt.stream_name:
                    self._stream_name = pkt.stream_name.strip().lower() or self._stream_name
                self.stats.update_extra(
                    sender_id=self._sender_id,
                    camera_id=self._camera_id,
                    stream_channel_id=self._stream_channel_id,
                    stream_name=self._expected_stream_name or self._stream_name,
                )
                assembly = self.transport.push(pkt)
                lost_delta = self.transport.lost_packets - self._last_lost_packets
                if lost_delta > 0:
                    self.stats.add_packets_lost(lost_delta)
                    self._last_lost_packets = self.transport.lost_packets
                if assembly is None:
                    continue
                self.stats.mark_in()
                with self.heap_lock:
                    heapq.heappush(self.assembly_heap, (assembly.timestamp, assembly.arrival_ts_ms, assembly))
                self.state.mark_success()
                self.stats.set_state(self.state.name)
            except Exception as exc:
                logging.exception("channel=%s receive failed", self.name)
                self.state.enter_reconnecting()
                self.stats.set_state(self.state.name, str(exc))
                time.sleep(self.state.next_backoff_ms(self.cfg.runtime.reconnect_backoff_ms) / 1000.0)

    def decode_thread(self) -> None:
        jitter_ms = self.cfg.network.jitter_ms
        reorder_window = max(1, self.cfg.network.reorder_window_frames)
        while not self.stop_event.is_set():
            assembly = None
            now_ms = int(time.time() * 1000)
            with self.heap_lock:
                if self.assembly_heap:
                    _, oldest_arrival_ms, _ = self.assembly_heap[0]
                    release_due = now_ms - oldest_arrival_ms >= jitter_ms
                    release_for_reorder = len(self.assembly_heap) > reorder_window
                    if release_due or release_for_reorder:
                        _, _, assembly = heapq.heappop(self.assembly_heap)
            if assembly is None:
                time.sleep(0.005)
                continue
            try:
                for frame in self.decoder.decode(assembly):
                    if not self._accept_stream(frame.stream_name or ""):
                        continue
                    self.decoded_queue.push_latest(frame)
                    self._record_latest_frame(frame)
                    latency_ms = max(0, now_ms - frame.capture_ts_ms)
                    self.stats.mark_out(0, latency_ms=latency_ms)
                    self.stats.set_queue_depth(self.decoded_queue.size())
                    self.stats.set_drop_count(self.decoded_queue.drops)
            except Exception as exc:
                self.state.mark_failure(self.cfg.runtime.degraded_threshold)
                logging.exception("channel=%s decode failed", self.name)
                self.stats.set_state(self.state.name, f"DECODE_FAIL: {exc}")

    def watchdog_thread(self) -> None:
        last_log = 0.0
        while not self.stop_event.is_set():
            now = time.monotonic()
            if now - last_log >= self.cfg.runtime.log_interval_ms / 1000.0:
                stat = self.stats.snapshot()
                if (
                    self._enable_renderer
                    and self._renderers
                    and stat.packets_rx > self._last_rx_for_stall
                    and now - self._last_render_monotonic > 2.0
                ):
                    logging.warning(
                        "channel=%s display stalled: packets are still incoming (rx=%d) but no new frame rendered for %.1fs",
                        self.name,
                        stat.packets_rx,
                        now - self._last_render_monotonic,
                    )
                self._last_rx_for_stall = stat.packets_rx
                sender_id = str(stat.extra.get("sender_id", "")) or "unknown_sender"
                camera_id = str(stat.extra.get("camera_id", "")) or self.name
                stream_channel_id = str(stat.extra.get("stream_channel_id", "")) or self.name
                stream_name = str(stat.extra.get("stream_name", "")) or (self._expected_stream_name or "-")
                logging.info(
                    "channel=%s port=%d sender=%s camera=%s stream_ch=%s stream_type=%s state=%s fps_in=%.1f fps_out=%.1f pkt_rate=%.1f latency=%.1fms rx=%d lost=%d queue=%d",
                    self.name,
                    self.cfg.network.listen_port,
                    sender_id,
                    camera_id,
                    stream_channel_id,
                    stream_name,
                    stat.state,
                    stat.fps_in,
                    stat.fps_out,
                    stat.packet_rate,
                    stat.e2e_latency_ms_avg,
                    stat.packets_rx,
                    stat.packets_lost,
                    stat.queue_depth,
                )
                if self.registry is not None:
                    self.registry.upsert(
                        sender_id=sender_id,
                        camera_id=camera_id,
                        stream_channel_id=stream_channel_id,
                        receiver_channel=self.name,
                        listen_port=self.cfg.network.listen_port,
                        state=stat.state,
                        last_error=stat.last_error,
                        packets_rx=stat.packets_rx,
                        fps_in=stat.fps_in,
                        fps_out=stat.fps_out,
                    )
                last_log = now
            time.sleep(self.cfg.runtime.watchdog_interval_ms / 1000.0)

    def start(self) -> None:
        if self._threads:
            return
        self.stop_event.clear()
        self._open()
        self._threads = [
            threading.Thread(target=self.recv_thread, name=f"{self.name}_recv", daemon=True),
            threading.Thread(target=self.decode_thread, name=f"{self.name}_decode", daemon=True),
            threading.Thread(target=self.watchdog_thread, name=f"{self.name}_watchdog", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def step_render(self, timeout_ms: int = 50) -> bool:
        if not self._enable_renderer:
            return not self.stop_event.is_set()
        frame = self.decoded_queue.pop(timeout_ms)
        if frame is not None:
            stat = self.stats.snapshot()
            renderer = self._get_renderer(frame.stream_name)
            if renderer is None:
                return not self.stop_event.is_set()
            keep_running = renderer.render(frame, stat)
            self._last_render_monotonic = time.monotonic()
            if not keep_running:
                self.stop_event.set()
            return keep_running
        if not self._renderers:
            return not self.stop_event.is_set()
        for renderer in self._renderers.values():
            if not renderer.poll_events():
                self.stop_event.set()
                return False
        return True

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()
        self.transport.close()
        self.decoder.close()
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self.renderer = None

    def run(self) -> int:
        self.start()
        try:
            while not self.stop_event.is_set():
                self.step_render(50)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        return 0


class MultiReceiverService:
    def __init__(self, cfg: ReceiverConfig) -> None:
        self.cfg = cfg
        self.registry = DeviceRegistry()
        self.services = self._build_services(cfg, self.registry)

    @staticmethod
    def _build_services(cfg: ReceiverConfig, registry: DeviceRegistry) -> list[ReceiverService]:
        channels = [channel for channel in cfg.channels if channel.enabled]
        services: list[ReceiverService] = []
        port_step = max(1, cfg.network.port_step)

        if channels:
            for index, channel in enumerate(channels):
                channel_cfg = copy.deepcopy(cfg)
                channel_cfg.channels = []
                channel_cfg.runtime.auto_listen_port_count = 0
                channel_cfg.network.listen_port = (
                    channel.listen_port if channel.listen_port is not None else cfg.network.listen_port + index * port_step
                )
                channel_name = channel.channel_id or f"rx{index}"
                services.append(
                    ReceiverService(
                        channel_cfg,
                        name=channel_name,
                        window_name=cfg.display.window_name,
                        show_stats=cfg.display.show_stats if channel.show_stats is None else channel.show_stats,
                        registry=registry,
                        enable_renderer=False,
                        expected_stream_name=channel.stream_name,
                        allowed_stream_names=["rgb", "depth_color"] if not channel.stream_name else None,
                    )
                )
            return services

        auto_count = max(0, int(cfg.runtime.auto_listen_port_count))
        if auto_count <= 0:
            auto_count = 1

        for index in range(auto_count):
            channel_cfg = copy.deepcopy(cfg)
            channel_cfg.channels = []
            channel_cfg.runtime.auto_listen_port_count = 0
            channel_cfg.network.listen_port = cfg.network.listen_port + index * port_step
            channel_name = f"rx{index}"
            services.append(
                ReceiverService(
                    channel_cfg,
                    name=channel_name,
                    window_name=cfg.display.window_name,
                    show_stats=cfg.display.show_stats,
                    registry=registry,
                    enable_renderer=False,
                    expected_stream_name="",
                    allowed_stream_names=["rgb", "depth_color"],
                )
            )
        return services

    def run(self) -> int:
        started: list[ReceiverService] = []
        wall = MonitorWallRenderer(self.cfg.display.window_name, self.cfg.display.show_stats)
        last_registry_log = 0.0
        for service in self.services:
            try:
                service.start()
                started.append(service)
                logging.info(
                    "channel=%s receiver started listen=%s:%d monitor_window=%s expected_stream=%s",
                    service.name,
                    service.cfg.network.listen_ip,
                    service.cfg.network.listen_port,
                    self.cfg.display.window_name,
                    service.expected_stream_name or "*",
                )
            except Exception:
                logging.exception("channel=%s receiver start failed", service.name)
        if not started:
            logging.error("no receiver channel started")
            wall.close()
            return 1
        try:
            while True:
                tiles: list[dict[str, object]] = []
                for service in started:
                    frame, stream_name = service.get_monitor_frame()
                    if frame is None:
                        continue
                    stat = service.stats.snapshot()
                    sender_id = str(stat.extra.get("sender_id", "")) or frame.sender_id or "unknown_sender"
                    camera_id = str(stat.extra.get("camera_id", "")) or frame.camera_id or service.name
                    tiles.append(
                        {
                            "frame": frame,
                            "stat": stat,
                            "sender_id": sender_id,
                            "camera_id": camera_id,
                            "stream_name": stream_name,
                        }
                    )
                tiles.sort(key=lambda item: (str(item.get("sender_id", "")), str(item.get("camera_id", ""))))
                keep_running = wall.render(tiles)
                now = time.monotonic()
                if now - last_registry_log >= 5.0:
                    records = self.registry.snapshot()
                    if records:
                        summary = ", ".join(
                            [
                                f"{item['sender_id']}/{item['camera_id']}[{item['stream_channel_id']}]@{item['listen_port']}:{item['state']}"
                                for item in records[:8]
                            ]
                        )
                        logging.info("registry size=%d %s", len(records), summary)
                    last_registry_log = now
                if not keep_running:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            wall.close()
            for service in started:
                service.stop()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini RTP receiver")
    parser.add_argument("--config", required=True, help="receiver json config")
    args = parser.parse_args()
    start_metrics_server_from_env("WIRELESS_VIDEO_RECEIVER_METRICS_PORT", default_port=0)
    cfg = load_receiver_config(args.config)
    multi_mode = bool(cfg.channels) or int(cfg.runtime.auto_listen_port_count) > 0
    if multi_mode:
        return MultiReceiverService(cfg).run()
    service = ReceiverService(
        cfg,
        name="rx0",
        window_name=cfg.display.window_name,
        show_stats=cfg.display.show_stats,
        expected_stream_name="rgb",
    )
    return service.run()
