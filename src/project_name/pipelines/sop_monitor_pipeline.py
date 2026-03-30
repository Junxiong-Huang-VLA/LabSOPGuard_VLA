from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from project_name.alerting.notifier import AlertNotifier
from project_name.detection.multi_level_detector import MultiLevelDetector
from project_name.monitoring.sop_engine import SOPComplianceEngine
from project_name.perception.center_depth_extractor import CenterDepthExtractor
from project_name.perception.event_structurer import EventStructurer
from project_name.perception.object_parser import ObjectParser
from project_name.report.report_input_builder import ReportInputBuilder
from project_name.video.capture import VideoCaptureStream


class SOPMonitorPipeline:
    def __init__(
        self,
        rules: Dict[str, Any],
        confidence_threshold: float = 0.45,
        alert_cooldown_seconds: float = 0.0,
        emit_console_alerts: bool = True,
        persist_alerts: bool = True,
    ) -> None:
        self.detector = MultiLevelDetector(confidence_threshold=confidence_threshold)
        self.engine = SOPComplianceEngine(rules=rules, cooldown_seconds=alert_cooldown_seconds)
        self.notifier = AlertNotifier()
        self.emit_console_alerts = bool(emit_console_alerts)
        self.persist_alerts = bool(persist_alerts)

        self.object_parser = ObjectParser()
        self.event_structurer = EventStructurer()
        self.depth_extractor = CenterDepthExtractor()
        self.report_builder = ReportInputBuilder()

    def run(
        self,
        video_source: str,
        max_frames: int = 120,
        target_fps: float = 10.0,
        sample_id: str = "runtime_session",
        camera_id: str = "cam0",
        camera_offsets_ms: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        self.engine.reset()
        stream = VideoCaptureStream(video_source, target_fps=target_fps)

        detections_raw: List[Dict[str, Any]] = []
        violations_raw: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []

        for frame in stream.frames(max_frames=max_frames):
            det = self.detector.detect(frame)
            detections_raw.append(asdict(det))

            parsed_objects = self.object_parser.parse_batch(det.objects, image_shape=frame.frame_bgr.shape[:2])
            for obj in parsed_objects:
                depth_info = None
                if obj.center_point is not None:
                    # No synchronized depth map in default pipeline; keep report schema compatible.
                    depth_info = self.depth_extractor.extract(depth_m=None, center_point=obj.center_point)

                ev = self.event_structurer.build_detection_event(
                    sample_id=sample_id,
                    camera_id=camera_id,
                    frame_id=frame.frame_id,
                    timestamp_sec=frame.timestamp_sec,
                    obj=obj,
                    sop_step=det.actions[0] if det.actions else "unknown",
                    depth_info=depth_info,
                )
                events.append(ev)

            vlist = self.engine.update(det)
            if vlist:
                if self.emit_console_alerts:
                    self.notifier.send_console(vlist)
                rows = self.notifier.notify(vlist) if self.persist_alerts else [asdict(v) for v in vlist]
                violations_raw.extend(rows)
                for v in rows:
                    ve = self.event_structurer.build_violation_event(
                        sample_id=sample_id,
                        camera_id=camera_id,
                        frame_id=int(v.get("frame_id", frame.frame_id)),
                        timestamp_sec=float(v.get("timestamp_sec", frame.timestamp_sec)),
                        violation=v,
                        related_target=events[-1] if events else None,
                    )
                    events.append(ve)

        if camera_offsets_ms:
            events = self.event_structurer.align_multi_camera_timestamp(events, camera_offsets_ms)

        status = self.engine.build_status()
        report_input = self.report_builder.build(session_id=sample_id, events=events)
        return {
            "detections": detections_raw,
            "violations": violations_raw,
            "events": events,
            "status": status,
            "report_input": report_input,
        }
