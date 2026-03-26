from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from project_name.common.config import load_yaml
from project_name.utils.spatial import camera_to_robot_xyz
from project_name.video.capture import VideoCaptureStream

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None


POSE_CSV_FIELDS = [
    "sample_id",
    "camera_id",
    "frame_id",
    "timestamp",
    "video_path",
    "image_path",
    "detections_2d",
    "keypoints_3d_camera",
    "keypoints_3d_base",
    "grasp_target",
    "pour_target",
    "vision_quality",
    "warnings",
]


@dataclass
class PoseConfig:
    model_path: str = "yolo26s-pose.pt"
    device: str = "cuda:0"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 960
    depth_window_size: int = 5
    min_valid_depth_ratio: float = 0.2
    max_depth_m: float = 3.0
    depth_unit: str = "auto"


def _load_json_or_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if p.suffix.lower() in {".json"}:
        return json.loads(p.read_text(encoding="utf-8"))
    return load_yaml(p)


def load_camera_info(path: str | Path | None, fallback_cfg: Dict[str, Any]) -> Dict[str, float]:
    """Load camera intrinsics from file or fallback config."""
    if path:
        raw = _load_json_or_yaml(path)
    else:
        raw = fallback_cfg

    # Supported schemas:
    # 1) {fx,fy,cx,cy}
    # 2) {intrinsics:{fx,fy,cx,cy}}
    # 3) {camera:{intrinsics:{fx,fy,cx,cy}}}
    cand = raw
    if "camera" in raw and isinstance(raw.get("camera"), dict):
        cand = raw["camera"]
    if "intrinsics" in cand and isinstance(cand.get("intrinsics"), dict):
        cand = cand["intrinsics"]

    keys = ("fx", "fy", "cx", "cy")
    if not all(k in cand for k in keys):
        raise ValueError("camera intrinsics must include fx, fy, cx, cy")
    return {k: float(cand[k]) for k in keys}


def load_extrinsics(path: str | Path | None, fallback_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load camera->base extrinsics from file or fallback config."""
    if path:
        raw = _load_json_or_yaml(path)
    else:
        raw = fallback_cfg

    if "camera" in raw and isinstance(raw.get("camera"), dict):
        cam = raw["camera"]
        if "hand_eye_extrinsics" in cam:
            return cam["hand_eye_extrinsics"]
    if "extrinsics" in raw:
        return raw["extrinsics"]
    return raw


class PoseDepthMVP:
    """MVP vision chain: 2D pose -> depth fusion -> 3D camera -> 3D base."""

    def __init__(
        self,
        cfg: PoseConfig,
        keypoint_names_cfg: Dict[str, List[str]],
        class_alias_cfg: Dict[str, List[str]],
        camera_intrinsics: Dict[str, float],
        extrinsics: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.keypoint_names_cfg = keypoint_names_cfg
        self.class_alias_cfg = class_alias_cfg
        self.camera_intrinsics = camera_intrinsics
        self.extrinsics = extrinsics or {}

        self.model = None
        self.init_warnings: List[str] = []
        if YOLO is None:
            self.init_warnings.append("ultralytics_not_available")
        else:
            try:
                self.model = YOLO(cfg.model_path)
            except Exception as exc:
                self.model = None
                self.init_warnings.append(f"pose_model_load_failed:{exc}")

    def _canonical_class(self, label: str) -> str:
        label_l = label.lower()
        for canonical, aliases in self.class_alias_cfg.items():
            if label_l == canonical.lower() or label_l in {a.lower() for a in aliases}:
                return canonical
        return label

    def _kp_names_for(self, class_name: str, count: int) -> List[str]:
        names = self.keypoint_names_cfg.get(class_name, self.keypoint_names_cfg.get("default", []))
        if not names:
            names = [f"kp_{i}" for i in range(count)]
        if len(names) < count:
            names = names + [f"kp_{i}" for i in range(len(names), count)]
        return names[:count]

    def infer_2d(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None:
            return []

        preds = self.model.predict(
            source=frame_bgr,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=False,
        )
        detections: List[Dict[str, Any]] = []

        for result in preds:
            names = result.names or {}
            boxes = result.boxes
            kobj = getattr(result, "keypoints", None)
            kp_xy = getattr(kobj, "xy", None) if kobj is not None else None
            kp_conf = getattr(kobj, "conf", None) if kobj is not None else None

            for i, box in enumerate(boxes):
                cls_id = int(box.cls.item())
                raw_name = str(names.get(cls_id, str(cls_id)))
                class_name = self._canonical_class(raw_name)
                score = float(box.conf.item())
                bbox = [int(v) for v in box.xyxy[0].tolist()]

                keypoints_2d: List[Dict[str, Any]] = []
                if kp_xy is not None and len(kp_xy) > i:
                    xy = kp_xy[i].cpu().numpy()
                    cf = (
                        kp_conf[i].cpu().numpy()
                        if (kp_conf is not None and len(kp_conf) > i)
                        else np.ones((xy.shape[0],), dtype=np.float32)
                    )
                    names_k = self._kp_names_for(class_name, int(xy.shape[0]))
                    for j in range(int(xy.shape[0])):
                        u = float(xy[j][0])
                        v = float(xy[j][1])
                        c = float(cf[j]) if np.isfinite(cf[j]) else 0.0
                        keypoints_2d.append(
                            {
                                "name": names_k[j],
                                "u": u,
                                "v": v,
                                "conf": c,
                                "valid": bool(c > 0.05),
                            }
                        )

                detections.append(
                    {
                        "bbox": bbox,
                        "class_name": class_name,
                        "score": score,
                        "keypoints_2d": keypoints_2d,
                    }
                )
        return detections

    def _depth_to_m(self, depth_raw: np.ndarray) -> np.ndarray:
        d = depth_raw.astype(np.float32)
        if self.cfg.depth_unit == "m":
            return d
        if self.cfg.depth_unit == "mm":
            return d / 1000.0
        valid = d[np.isfinite(d) & (d > 0)]
        if valid.size == 0:
            return d
        return d / 1000.0 if float(np.median(valid)) > 10.0 else d

    def _sample_depth(self, depth_m: np.ndarray, u: float, v: float) -> Dict[str, Any]:
        h, w = depth_m.shape[:2]
        ui = int(round(u))
        vi = int(round(v))
        r = max(1, int(self.cfg.depth_window_size // 2))
        x1, x2 = max(0, ui - r), min(w, ui + r + 1)
        y1, y2 = max(0, vi - r), min(h, vi + r + 1)
        patch = depth_m[y1:y2, x1:x2]

        valid = patch[
            np.isfinite(patch) & (patch > 0.0) & (patch < float(self.cfg.max_depth_m))
        ]
        ratio = float(valid.size / patch.size) if patch.size else 0.0
        if valid.size == 0:
            return {
                "depth_m": 0.0,
                "valid": False,
                "valid_depth_ratio": ratio,
                "depth_std": 0.0,
                "source": "window_median_empty",
            }
        return {
            "depth_m": float(np.median(valid)),
            "valid": bool(ratio >= float(self.cfg.min_valid_depth_ratio)),
            "valid_depth_ratio": ratio,
            "depth_std": float(np.std(valid)),
            "source": "window_median",
        }

    def _to_camera_xyz(self, u: float, v: float, z: float) -> Optional[List[float]]:
        if z <= 0.0:
            return None
        fx = float(self.camera_intrinsics["fx"])
        fy = float(self.camera_intrinsics["fy"])
        cx = float(self.camera_intrinsics["cx"])
        cy = float(self.camera_intrinsics["cy"])
        if fx == 0.0 or fy == 0.0:
            return None
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return [float(x), float(y), float(z)]

    def fuse_depth_and_3d(
        self,
        detections_2d: List[Dict[str, Any]],
        depth_raw: Optional[np.ndarray],
        export_base_frame: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        kps_cam: List[Dict[str, Any]] = []
        kps_base: List[Dict[str, Any]] = []
        warnings: List[str] = []

        depth_m = self._depth_to_m(depth_raw) if depth_raw is not None else None
        if depth_m is None:
            warnings.append("depth_missing")

        total_points = 0
        valid_points = 0

        for det_idx, det in enumerate(detections_2d):
            class_name = str(det.get("class_name", "unknown"))
            for kp in det.get("keypoints_2d", []):
                total_points += 1
                u = float(kp.get("u", 0.0))
                v = float(kp.get("v", 0.0))
                dmeta = {
                    "depth_m": 0.0,
                    "valid": False,
                    "valid_depth_ratio": 0.0,
                    "depth_std": 0.0,
                    "source": "no_depth",
                }
                if depth_m is not None:
                    dmeta = self._sample_depth(depth_m, u, v)
                xyz_cam = self._to_camera_xyz(u, v, float(dmeta["depth_m"]))
                if xyz_cam is not None and bool(dmeta["valid"]):
                    valid_points += 1

                cam_item = {
                    "det_index": det_idx,
                    "class_name": class_name,
                    "name": kp.get("name"),
                    "u": u,
                    "v": v,
                    "xyz": xyz_cam,
                    "valid": bool(dmeta["valid"]) and xyz_cam is not None,
                    "valid_depth_ratio": float(dmeta["valid_depth_ratio"]),
                    "depth_std": float(dmeta["depth_std"]),
                    "source": dmeta["source"],
                }
                kps_cam.append(cam_item)

                if export_base_frame:
                    xyz_base = camera_to_robot_xyz(
                        xyz_cam,
                        self.extrinsics if self.extrinsics else None,
                    )
                    base_item = dict(cam_item)
                    base_item["xyz"] = xyz_base
                    base_item["valid"] = bool(base_item["valid"] and xyz_base is not None)
                    kps_base.append(base_item)

        quality = {
            "total_keypoints": total_points,
            "valid_keypoints": valid_points,
            "valid_ratio": float(valid_points / total_points) if total_points else 0.0,
            "warnings": warnings,
        }
        return kps_cam, kps_base, quality

    @staticmethod
    def _pick_target(
        points: List[Dict[str, Any]],
        class_name: str,
        preferred_names: List[str],
    ) -> Optional[Dict[str, Any]]:
        for pname in preferred_names:
            for p in points:
                if (
                    p.get("class_name") == class_name
                    and p.get("name") == pname
                    and p.get("valid")
                    and p.get("xyz") is not None
                ):
                    return {"name": pname, "xyz": p["xyz"], "frame": class_name}
        return None

    def build_action_targets(
        self,
        keypoints_3d_base: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        grasp = self._pick_target(
            keypoints_3d_base,
            "titration_tool",
            ["tool_grasp_point", "tool_body_center", "tool_tip"],
        )
        pour = self._pick_target(
            keypoints_3d_base,
            "target_vessel",
            ["vessel_center", "vessel_rim_left", "vessel_rim_right"],
        )
        if grasp:
            grasp["orientation_hint"] = {"type": "identity_quaternion", "xyzw": [0.0, 0.0, 0.0, 1.0]}
        if pour:
            pour["orientation_hint"] = {"type": "identity_quaternion", "xyzw": [0.0, 0.0, 0.0, 1.0]}
        return grasp, pour

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        detections_2d: List[Dict[str, Any]],
        out_path: str | Path,
    ) -> None:
        canvas = frame_bgr.copy()
        for det in detections_2d:
            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            name = str(det.get("class_name", "obj"))
            score = float(det.get("score", 0.0))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                canvas,
                f"{name}:{score:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            for kp in det.get("keypoints_2d", []):
                if not kp.get("valid", False):
                    continue
                u, v = int(round(float(kp["u"]))), int(round(float(kp["v"])))
                cv2.circle(canvas, (u, v), 3, (0, 170, 255), -1)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), canvas)


def export_pose_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_pose_csv(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POSE_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.get("sample_id"),
                    "camera_id": row.get("camera_id"),
                    "frame_id": row.get("frame_id"),
                    "timestamp": row.get("timestamp"),
                    "video_path": row.get("video_path"),
                    "image_path": row.get("image_path"),
                    "detections_2d": json.dumps(row.get("detections_2d", []), ensure_ascii=False),
                    "keypoints_3d_camera": json.dumps(row.get("keypoints_3d_camera", []), ensure_ascii=False),
                    "keypoints_3d_base": json.dumps(row.get("keypoints_3d_base", []), ensure_ascii=False),
                    "grasp_target": json.dumps(row.get("grasp_target"), ensure_ascii=False),
                    "pour_target": json.dumps(row.get("pour_target"), ensure_ascii=False),
                    "vision_quality": json.dumps(row.get("vision_quality", {}), ensure_ascii=False),
                    "warnings": json.dumps(row.get("warnings", []), ensure_ascii=False),
                }
            )


def _load_depth_frame(depth_path: str, frame_id: int) -> Optional[np.ndarray]:
    p = Path(depth_path)
    if not p.exists():
        return None
    if p.suffix.lower() == ".npy":
        return np.load(str(p))
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def run_pose_export_batch(
    mvp: PoseDepthMVP,
    sources: List[Dict[str, Any]],
    max_frames: int,
    target_fps: float,
    export_base_frame: bool,
    debug_overlay: bool,
    overlay_dir: str | Path,
    logger: Any,
) -> List[Dict[str, Any]]:
    """Run pose-depth-3D export for manifest or single video sources."""
    rows: List[Dict[str, Any]] = []
    overlay_root = Path(overlay_dir)

    for idx, source in enumerate(sources):
        sample_id = str(source.get("sample_id", f"sample_{idx:04d}"))
        camera_id = str(source.get("camera_id", "cam0"))
        video_path = source.get("video_path") or source.get("rgb_path")
        image_path = source.get("image_path")
        depth_path = source.get("depth_path")
        if not video_path and not image_path:
            logger.warning("pose export skip sample without input path: %s", sample_id)
            continue

        if image_path:
            frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                logger.warning("pose export image decode failed: %s", image_path)
                continue
            detections_2d = mvp.infer_2d(frame_bgr)
            depth_raw = _load_depth_frame(str(depth_path), 0) if depth_path else None
            kps_cam, kps_base, quality = mvp.fuse_depth_and_3d(
                detections_2d=detections_2d,
                depth_raw=depth_raw,
                export_base_frame=export_base_frame,
            )
            grasp_target, pour_target = mvp.build_action_targets(kps_base)
            row = {
                "sample_id": sample_id,
                "camera_id": camera_id,
                "frame_id": 0,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "video_path": None,
                "image_path": image_path,
                "detections_2d": detections_2d,
                "keypoints_3d_camera": kps_cam,
                "keypoints_3d_base": kps_base,
                "grasp_target": grasp_target,
                "pour_target": pour_target,
                "vision_quality": quality,
                "warnings": quality.get("warnings", []),
            }
            rows.append(row)
            if debug_overlay:
                mvp.draw_overlay(
                    frame_bgr,
                    detections_2d,
                    overlay_root / sample_id / "frame_000000.jpg",
                )
            continue

        stream = VideoCaptureStream(str(video_path), target_fps=target_fps)
        for packet in stream.frames(max_frames=max_frames):
            detections_2d = mvp.infer_2d(packet.frame_bgr)
            depth_raw = _load_depth_frame(str(depth_path), packet.frame_id) if depth_path else None
            kps_cam, kps_base, quality = mvp.fuse_depth_and_3d(
                detections_2d=detections_2d,
                depth_raw=depth_raw,
                export_base_frame=export_base_frame,
            )
            grasp_target, pour_target = mvp.build_action_targets(kps_base)
            row = {
                "sample_id": sample_id,
                "camera_id": camera_id,
                "frame_id": packet.frame_id,
                "timestamp": f"{packet.timestamp_sec:.3f}",
                "video_path": video_path,
                "image_path": None,
                "detections_2d": detections_2d,
                "keypoints_3d_camera": kps_cam,
                "keypoints_3d_base": kps_base,
                "grasp_target": grasp_target,
                "pour_target": pour_target,
                "vision_quality": quality,
                "warnings": quality.get("warnings", []),
            }
            rows.append(row)
            if debug_overlay:
                mvp.draw_overlay(
                    packet.frame_bgr,
                    detections_2d,
                    overlay_root / sample_id / f"frame_{packet.frame_id:06d}.jpg",
                )
        logger.info("pose export [%d/%d] done: %s", idx + 1, len(sources), sample_id)
    return rows
