from __future__ import annotations

import base64
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
_SRC_ROOT = _PROJECT_ROOT / 'src'
for _p in (str(_SRC_ROOT), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from project_name.detection.multi_level_detector import DetectionEvent, MultiLevelDetector
from project_name.monitoring.sop_engine import SOPComplianceEngine, ViolationEvent
from project_name.video.capture import FramePacket

logger = logging.getLogger(__name__)

RULES_PATH = _PROJECT_ROOT / 'configs' / 'sop' / 'rules.yaml'
RUNTIME_CONFIG_PATH = _PROJECT_ROOT / 'configs' / 'model' / 'detection_runtime.yaml'

SEVERITY_DISPLAY_MAP = {
    'critical': 'Critical',
    'high': 'Critical',
    'major': 'Major',
    'medium': 'Major',
    'minor': 'Minor',
    'warning': 'Warning',
    'low': 'Warning',
}
SEVERITY_COLOR_MAP = {
    'Critical': (0, 0, 255),
    'Major': (0, 140, 255),
    'Minor': (0, 255, 255),
    'Warning': (255, 255, 0),
}
STEP_LABELS = {
    'wear_gloves': 'Wear gloves',
    'wear_goggles': 'Wear goggles',
    'wear_lab_coat': 'Wear lab coat',
    'verify_label': 'Verify reagent label',
    'pipette_transfer': 'Pipette transfer',
    'cap_container': 'Cap container',
    'dispose_waste': 'Dispose waste',
}


@dataclass
class AdaptiveLabConstraint:
    constraint_id: str
    description: str
    severity: str
    enabled: bool = True
    sop_step: str = ''


@dataclass
class RealTimeDetection:
    frame_id: int
    timestamp: float
    objects: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    ppe: Dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.0
    layer_outputs: Dict[str, Any] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)


class _FallbackDetector:
    def detect(self, frame: FramePacket) -> DetectionEvent:
        objects = [
            {'label': 'sample_container', 'bbox': [120, 90, 250, 260], 'score': 0.78},
            {'label': 'pipette', 'bbox': [270, 120, 380, 220], 'score': 0.66},
        ]
        return DetectionEvent(
            frame_id=frame.frame_id,
            timestamp_sec=frame.timestamp_sec,
            ppe={'wear_gloves': False, 'wear_goggles': False, 'wear_lab_coat': False},
            objects=objects,
            actions=['verify_label', 'pipette_transfer'],
            confidence=0.5,
            layer_outputs={
                'layer1_realtime_pose': {
                    'model': 'fallback_heuristic',
                    'backend': 'fallback_heuristic',
                    'objects_count': len(objects),
                    'pose_keypoints_17': None,
                    'pose_instances': 0,
                },
                'layer2_action_analysis': {
                    'model': 'fallback',
                    'window_size': 0,
                    'actions': ['verify_label', 'pipette_transfer'],
                    'action_confidence': 0.5,
                    'backend': 'fallback',
                },
            },
        )


class AdaptiveLabMonitor:
    def __init__(
        self,
        rules_path: Path = RULES_PATH,
        runtime_config_path: Path = RUNTIME_CONFIG_PATH,
        confidence_threshold: float = 0.45,
        alert_cooldown_seconds: float = 1.0,
    ) -> None:
        self.rules_path = Path(rules_path)
        self.runtime_config_path = Path(runtime_config_path)
        self.confidence_threshold = float(confidence_threshold)
        self.alert_cooldown_seconds = float(alert_cooldown_seconds)
        self.rules = self._load_rules()
        self.constraints = self._build_constraints()
        self.engine = SOPComplianceEngine(self.rules, cooldown_seconds=self.alert_cooldown_seconds)
        self.detection_history: List[RealTimeDetection] = []
        self.violation_history: List[Dict[str, Any]] = []
        self.is_active = False
        self.current_frame_id = 0
        self.started_at: Optional[float] = None
        self.latest_detection: Optional[RealTimeDetection] = None
        self.latest_status: Dict[str, Any] = self.engine.build_status()
        self._detector: Optional[Any] = None
        self._lock = threading.Lock()

    def _load_rules(self) -> Dict[str, Any]:
        if not self.rules_path.exists():
            logger.warning('rules file missing: %s', self.rules_path)
            return {}
        return yaml.safe_load(self.rules_path.read_text(encoding='utf-8')) or {}

    def _build_constraints(self) -> List[AdaptiveLabConstraint]:
        violation_rules = self.rules.get('violation_rules', {}) if isinstance(self.rules, dict) else {}
        constraints: List[AdaptiveLabConstraint] = []
        if isinstance(violation_rules, dict):
            for rule_id, payload in violation_rules.items():
                if not isinstance(payload, dict):
                    continue
                constraints.append(
                    AdaptiveLabConstraint(
                        constraint_id=str(rule_id),
                        description=str(payload.get('message') or payload.get('condition') or rule_id),
                        severity=self._normalize_severity(str(payload.get('severity', 'warning'))),
                        sop_step=str(payload.get('sop_step', '')),
                    )
                )
        return constraints

    @staticmethod
    def _normalize_severity(value: str) -> str:
        key = str(value or '').strip().lower()
        return SEVERITY_DISPLAY_MAP.get(key, 'Warning')

    def _resolve_local_model_path(self, model_name: str) -> Optional[Path]:
        model_ref = Path(str(model_name).strip())
        if not str(model_ref):
            return None
        if model_ref.is_absolute():
            return model_ref if model_ref.exists() else None

        candidates = [
            self.runtime_config_path.parent / model_ref,
            self.runtime_config_path.parent.parent / model_ref,
            _PROJECT_ROOT / model_ref,
            _PROJECT_ROOT.parent / model_ref,
            Path.cwd() / model_ref,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _resolved_runtime_config_path(self) -> Path:
        if not self.runtime_config_path.exists():
            return self.runtime_config_path

        runtime_cfg = yaml.safe_load(self.runtime_config_path.read_text(encoding='utf-8')) or {}
        if not isinstance(runtime_cfg, dict):
            return self.runtime_config_path

        backend = str(runtime_cfg.get('backend', 'ultralytics')).strip().lower()
        model_name = str(runtime_cfg.get('model', '')).strip()
        if backend != 'ultralytics' or not model_name:
            return self.runtime_config_path

        resolved_model = self._resolve_local_model_path(model_name)
        if resolved_model is None:
            raise FileNotFoundError(f'local detector weight not found: {model_name}')

        if Path(model_name).is_absolute():
            return self.runtime_config_path

        runtime_cfg['model'] = str(resolved_model)
        cache_dir = _PROJECT_ROOT / '.runtime_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        resolved_cfg_path = cache_dir / 'adaptive_monitor_runtime.yaml'
        resolved_cfg_path.write_text(
            yaml.safe_dump(runtime_cfg, sort_keys=False, allow_unicode=False),
            encoding='utf-8',
        )
        return resolved_cfg_path

    def _ensure_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        try:
            runtime_config_path = self._resolved_runtime_config_path()
            self._detector = MultiLevelDetector(
                confidence_threshold=self.confidence_threshold,
                runtime_config_path=str(runtime_config_path),
            )
            logger.info('adaptive monitor detector ready: backend=%s model=%s', self._detector.backend, self._detector.model_name)
        except Exception as exc:
            logger.exception('failed to initialize detector, using fallback detector: %s', exc)
            self._detector = _FallbackDetector()
        return self._detector

    def start_monitoring(self) -> None:
        with self._lock:
            self.engine.reset()
            self.detection_history.clear()
            self.violation_history.clear()
            self.latest_detection = None
            self.latest_status = self.engine.build_status()
            self.current_frame_id = 0
            self.started_at = time.perf_counter()
            self.is_active = True

    def stop_monitoring(self) -> None:
        with self._lock:
            self.is_active = False

    def reset_statistics(self) -> None:
        with self._lock:
            self.engine.reset()
            self.detection_history.clear()
            self.violation_history.clear()
            self.latest_detection = None
            self.latest_status = self.engine.build_status()
            self.current_frame_id = 0
            self.started_at = time.perf_counter() if self.is_active else None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        with self._lock:
            if not self.is_active:
                return frame, []

            self.current_frame_id += 1
            frame_id = int(self.current_frame_id)
            started_at = self.started_at or time.perf_counter()
            timestamp_sec = max(0.0, time.perf_counter() - started_at)

            packet = FramePacket(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                frame_bgr=frame,
                source='adaptive_monitor',
            )
            detector = self._ensure_detector()
            det = detector.detect(packet)
            violations_raw = self.engine.update(det)
            violations = [self._serialize_violation(v, det) for v in violations_raw]

            detection = RealTimeDetection(
                frame_id=det.frame_id,
                timestamp=det.timestamp_sec,
                objects=list(det.objects or []),
                actions=list(det.actions or []),
                ppe=dict(det.ppe or {}),
                confidence=float(det.confidence),
                layer_outputs=dict(det.layer_outputs or {}),
                violations=violations,
            )
            self.detection_history.append(detection)
            self.latest_detection = detection
            self.latest_status = self.engine.build_status()
            if violations:
                self.violation_history.extend(violations)

            if len(self.detection_history) > 600:
                self.detection_history = self.detection_history[-300:]
            if len(self.violation_history) > 600:
                self.violation_history = self.violation_history[-300:]

            visual_frame = self._draw_visualization(frame, detection)
            return visual_frame, violations

    def _serialize_violation(self, violation: ViolationEvent, detection: DetectionEvent) -> Dict[str, Any]:
        severity = self._normalize_severity(violation.severity)
        return {
            'constraint_id': violation.rule_id,
            'description': violation.message,
            'severity': severity,
            'confidence': max(0.1, min(1.0, float(detection.confidence))),
            'timestamp': float(violation.timestamp_sec),
            'timestamp_sec': float(violation.timestamp_sec),
            'frame_id': int(violation.frame_id),
            'recommendation': self._recommendation_for_rule(violation.rule_id),
        }

    @staticmethod
    def _recommendation_for_rule(rule_id: str) -> str:
        recommendations = {
            'missing_ppe': 'Wear all required PPE before continuing the operation.',
            'reagent_unverified': 'Verify the reagent label before transfer starts.',
            'unsafe_transfer_zone': 'Move the transfer task back to the approved working zone.',
            'container_not_closed': 'Close the container before ending the experiment.',
            'waste_not_disposed': 'Dispose waste according to SOP before session closeout.',
        }
        return recommendations.get(str(rule_id), 'Review the SOP step and correct the violation before continuing.')

    def _draw_visualization(self, frame: np.ndarray, detection: RealTimeDetection) -> np.ndarray:
        visual = frame.copy()
        height, width = visual.shape[:2]

        for obj in detection.objects:
            bbox = obj.get('bbox') or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            score = float(obj.get('score', obj.get('confidence', 0.0)))
            label = str(obj.get('label', 'object'))
            if label in {'glove', 'goggles', 'lab_coat'}:
                color = (0, 255, 0)
            elif label in {'person', 'human', 'operator', 'worker'}:
                color = (255, 200, 0)
            else:
                color = (255, 255, 0)
            cv2.rectangle(visual, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                visual,
                f'{label} {score:.2f}',
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(visual, (0, 0), (width, 58), (15, 18, 28), -1)
        backend = self._extract_backend(detection.layer_outputs)
        compliance_ratio = float(self.latest_status.get('compliance_ratio', 0.0))
        ppe_ok = all(bool(v) for v in detection.ppe.values()) if detection.ppe else False
        header_left = f'Frame {detection.frame_id} | {detection.timestamp:.2f}s | {backend}'
        header_right = f'Compliance {compliance_ratio * 100:.1f}% | PPE {"OK" if ppe_ok else "CHECK"}'
        cv2.putText(visual, header_left, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(visual, header_right, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 200), 1, cv2.LINE_AA)

        if detection.violations:
            top = 72
            for violation in detection.violations[:3]:
                severity = str(violation.get('severity', 'Warning'))
                color = SEVERITY_COLOR_MAP.get(severity, (255, 255, 255))
                text = f'{severity}: {violation.get("description", "Violation detected")}'
                cv2.rectangle(visual, (18, top - 18), (width - 18, top + 10), (20, 20, 20), -1)
                cv2.rectangle(visual, (18, top - 18), (width - 18, top + 10), color, 2)
                cv2.putText(visual, text[:96], (28, top), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                top += 34

        return visual

    @staticmethod
    def _extract_backend(layer_outputs: Dict[str, Any]) -> str:
        layer1 = layer_outputs.get('layer1_realtime_pose', {}) if isinstance(layer_outputs, dict) else {}
        if isinstance(layer1, dict):
            return str(layer1.get('backend') or layer1.get('model') or 'unknown')
        return 'unknown'

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            severity_stats: Dict[str, int] = {}
            for violation in self.violation_history:
                severity = str(violation.get('severity', 'Warning'))
                severity_stats[severity] = severity_stats.get(severity, 0) + 1

            latest_detection = self.latest_detection
            latest_backend = self._extract_backend(latest_detection.layer_outputs) if latest_detection else 'unknown'
            step_state = dict(getattr(self.engine, 'step_state', {}) or {})
            recent_violations = list(self.violation_history[-10:])
            return {
                'total_frames': self.current_frame_id,
                'total_violations': len(self.violation_history),
                'severity_distribution': severity_stats,
                'active_constraints': len([c for c in self.constraints if c.enabled]),
                'is_active': self.is_active,
                'step_state': step_state,
                'completed_steps': list(self.latest_status.get('completed_steps', [])),
                'pending_steps': list(self.latest_status.get('pending_steps', [])),
                'compliance_ratio': float(self.latest_status.get('compliance_ratio', 0.0)),
                'latest_actions': list(latest_detection.actions) if latest_detection else [],
                'latest_ppe': dict(latest_detection.ppe) if latest_detection else {},
                'latest_objects_count': len(latest_detection.objects) if latest_detection else 0,
                'latest_backend': latest_backend,
                'recent_violations': recent_violations,
                'step_labels': STEP_LABELS,
            }


monitor_instance = AdaptiveLabMonitor()


def get_monitor() -> AdaptiveLabMonitor:
    return monitor_instance


async def process_video_frame(frame_data: bytes) -> Dict[str, Any]:
    try:
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {'error': 'Unable to decode frame payload.'}

        monitor = get_monitor()
        visual_frame, violations = monitor.process_frame(frame)
        ok, buffer = cv2.imencode('.jpg', visual_frame)
        if not ok:
            return {'error': 'Unable to encode annotated frame.'}

        stats = monitor.get_statistics()
        latest_detection = monitor.latest_detection
        return {
            'success': True,
            'visual_frame': base64.b64encode(buffer).decode('utf-8'),
            'violations': violations,
            'statistics': stats,
            'detection': {
                'actions': list(latest_detection.actions) if latest_detection else [],
                'objects': list(latest_detection.objects) if latest_detection else [],
                'ppe': dict(latest_detection.ppe) if latest_detection else {},
                'confidence': float(latest_detection.confidence) if latest_detection else 0.0,
                'layer_outputs': dict(latest_detection.layer_outputs) if latest_detection else {},
            },
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception('adaptive monitor frame processing failed: %s', exc)
        return {'error': str(exc)}
