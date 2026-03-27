from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2


try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None


def run_hand_detection(
    video_path: str,
    output_dir: Path,
    enable_video_export: bool = True,
    target_fps: float = 12.0,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 25.0
    step = max(1, int(round(src_fps / target_fps)))

    hand_json = output_dir / "hand_detection.json"
    annotated_path = output_dir / "hand_annotated.mp4"

    frame_results: List[Dict[str, Any]] = []
    total_frames = 0
    hand_frames = 0
    max_hands = 0

    writer = None
    mp_hands = None
    mp_draw = None
    hands = None
    hand_backend = "disabled"
    warnings: List[str] = []

    if mp is not None:
        # MediaPipe package API may differ by version:
        # - legacy: mp.solutions.hands
        # - tasks-only builds: no "solutions" export
        if hasattr(mp, "solutions"):
            mp_hands = mp.solutions.hands
            mp_draw = mp.solutions.drawing_utils
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            hand_backend = "mediapipe_solutions"
        else:
            warnings.append(
                "mediapipe.solutions is unavailable in current mediapipe build; hand detection skipped."
            )
            hand_backend = "mediapipe_no_solutions"
    else:
        warnings.append("mediapipe package is not installed; hand detection skipped.")

    try:
        frame_id = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % step != 0:
                continue

            total_frames += 1
            ts = frame_id / src_fps
            hands_count = 0
            landmarks_payload: List[List[float]] = []

            if hands is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    hands_count = len(res.multi_hand_landmarks)
                    for hand_lm in res.multi_hand_landmarks:
                        points = [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                        landmarks_payload.append(points)
                        if enable_video_export and mp_draw and mp_hands:
                            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            if hands is None:
                cv2.putText(
                    frame,
                    "Hand detection skipped (mediapipe backend unavailable)",
                    (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 180, 255),
                    2,
                )

            if hands_count > 0:
                hand_frames += 1
            max_hands = max(max_hands, hands_count)

            frame_results.append(
                {
                    "frame_id": frame_id,
                    "timestamp": ts,
                    "hands_count": hands_count,
                    "landmarks": landmarks_payload,
                }
            )

            if enable_video_export:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(annotated_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(target_fps),
                        (w, h),
                        True,
                    )
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if hands is not None:
            hands.close()

    summary = {
        "total_sampled_frames": total_frames,
        "frames_with_hands": hand_frames,
        "hand_presence_ratio": float(hand_frames / total_frames) if total_frames else 0.0,
        "max_hands_in_frame": max_hands,
        "mediapipe_enabled": hands is not None,
        "hand_backend": hand_backend,
        "warnings": warnings,
    }

    payload = {
        "video_path": video_path,
        "summary": summary,
        "frame_results": frame_results,
        "annotated_video": str(annotated_path).replace("\\", "/") if enable_video_export else None,
        "warnings": warnings,
    }
    hand_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
