from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2


try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None


HAND_CONNECTIONS: List[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

DEFAULT_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _resolve_ffmpeg_executable() -> Optional[str]:
    cli_ffmpeg = shutil.which("ffmpeg")
    if cli_ffmpeg:
        return cli_ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore

        ffmpeg_bin = get_ffmpeg_exe()
        if ffmpeg_bin and Path(ffmpeg_bin).exists():
            return str(ffmpeg_bin)
    except Exception:
        return None
    return None


def _open_video_writer(path: Path, fps: float, size: tuple[int, int], warnings: List[str]):
    width, height = size
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            float(max(fps, 1.0)),
            (width, height),
            True,
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    warnings.append("annotated video writer init failed; export disabled.")
    return None, ""


def _draw_annotation_overlay(
    frame,
    *,
    timestamp_sec: float,
    sampled_index: int,
    hands_count: int,
    backend: str,
    fallback_mode: bool,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (42, 120, 218), 2)
    cv2.rectangle(frame, (0, max(0, h - 46)), (w, h), (22, 34, 48), -1)
    line1 = f"LabSOPGuard Annotated | t={timestamp_sec:.2f}s | sample={sampled_index:04d}"
    line2 = f"hands={hands_count} | backend={backend}"
    if fallback_mode:
        line2 += " | local-fallback"
    cv2.putText(frame, line1[:96], (12, max(16, h - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 251, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, line2[:96], (12, max(16, h - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (167, 220, 255), 1, cv2.LINE_AA)


def _try_transcode_web_mp4(path: Path, warnings: List[str]) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    ffmpeg_bin = _resolve_ffmpeg_executable()
    if not ffmpeg_bin:
        warnings.append("ffmpeg unavailable; keep original video encoding.")
        return False

    tmp_path = path.with_name(f"{path.stem}_web.mp4")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=300)
        if proc.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size <= 0:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            warnings.append("ffmpeg transcode failed; keep original video encoding.")
            return False
        path.unlink(missing_ok=True)
        tmp_path.replace(path)
        return True
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        warnings.append(f"ffmpeg transcode exception: {exc}")
        return False


def _draw_hand_landmarks_basic(frame, points: List[List[float]]) -> None:
    if not points:
        return
    h, w = frame.shape[:2]
    pix: List[tuple[int, int]] = []
    for p in points:
        if not isinstance(p, list) or len(p) < 2:
            pix.append((-1, -1))
            continue
        x = int(max(0, min(w - 1, round(float(p[0]) * w))))
        y = int(max(0, min(h - 1, round(float(p[1]) * h))))
        pix.append((x, y))

    for a, b in HAND_CONNECTIONS:
        if a >= len(pix) or b >= len(pix):
            continue
        ax, ay = pix[a]
        bx, by = pix[b]
        if ax < 0 or ay < 0 or bx < 0 or by < 0:
            continue
        cv2.line(frame, (ax, ay), (bx, by), (80, 235, 138), 2, cv2.LINE_AA)

    for x, y in pix:
        if x < 0 or y < 0:
            continue
        cv2.circle(frame, (x, y), 2, (29, 219, 242), -1, cv2.LINE_AA)


def _ensure_hand_landmarker_model(warnings: List[str]) -> Optional[Path]:
    custom_model = str(os.getenv("HAND_LANDMARKER_MODEL", "")).strip()
    if custom_model:
        custom_path = Path(custom_model).expanduser().resolve()
        if custom_path.exists() and custom_path.is_file():
            return custom_path
        warnings.append(f"HAND_LANDMARKER_MODEL not found: {custom_path}")

    model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
    if model_path.exists() and model_path.is_file():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(DEFAULT_TASK_MODEL_URL, timeout=30) as resp:  # nosec B310
            data = resp.read()
        if not data:
            warnings.append("hand landmarker model download returned empty content.")
            return None
        model_path.write_bytes(data)
        return model_path
    except Exception as exc:
        warnings.append(f"hand landmarker model download failed: {exc}")
        return None


def run_hand_detection(
    video_path: str,
    output_dir: Path,
    enable_video_export: bool = True,
    target_fps: float = 12.0,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 25.0
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_total_frames = total_frames_raw if total_frames_raw > 0 else 0
    step = max(1, int(round(src_fps / target_fps)))

    hand_json = output_dir / "hand_detection.json"
    annotated_path = output_dir / "hand_annotated.mp4"

    frame_results: List[Dict[str, Any]] = []
    sampled_frames = 0
    hand_frames = 0
    max_hands = 0
    video_export_active = bool(enable_video_export)

    writer = None
    writer_codec = ""
    mp_hands = None
    mp_draw = None
    hands = None
    tasks_landmarker = None
    hand_backend = "disabled"
    warnings: List[str] = []

    if mp is not None:
        # MediaPipe package API may differ by version:
        # - legacy: mp.solutions.hands
        # - tasks builds: use HandLandmarker task model
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
            try:
                from mediapipe.tasks import python as mp_tasks  # type: ignore
                from mediapipe.tasks.python import vision  # type: ignore

                model_path = _ensure_hand_landmarker_model(warnings)
                if model_path and model_path.exists():
                    options = vision.HandLandmarkerOptions(
                        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
                        running_mode=vision.RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.45,
                        min_hand_presence_confidence=0.45,
                        min_tracking_confidence=0.45,
                    )
                    tasks_landmarker = vision.HandLandmarker.create_from_options(options)
                    hand_backend = "mediapipe_tasks"
                else:
                    warnings.append("hand landmarker model unavailable; hand detection skipped.")
                    hand_backend = "mediapipe_tasks_model_missing"
            except Exception as exc:
                warnings.append(f"mediapipe tasks hand detector init failed: {exc}")
                hand_backend = "mediapipe_tasks_init_failed"
    else:
        warnings.append("mediapipe package is not installed; hand detection skipped.")

    try:
        frame_id = -1
        last_reported = 0
        report_interval = max(10, int(round(src_fps)))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if progress_callback and source_total_frames > 0:
                processed = frame_id + 1
                if (processed - last_reported) >= report_interval:
                    last_reported = processed
                    progress_callback(processed, source_total_frames, sampled_frames, "手部检测中：读取视频帧。")
            if frame_id % step != 0:
                continue

            sampled_frames += 1
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
                        if video_export_active and mp_draw and mp_hands:
                            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            elif tasks_landmarker is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                task_result = tasks_landmarker.detect(mp_image)
                hand_list = list(task_result.hand_landmarks) if getattr(task_result, "hand_landmarks", None) else []
                hands_count = len(hand_list)
                for hand_lm in hand_list:
                    points = [[lm.x, lm.y, lm.z] for lm in hand_lm]
                    landmarks_payload.append(points)
                    if video_export_active:
                        _draw_hand_landmarks_basic(frame, points)

            if hands_count > 0:
                hand_frames += 1
            max_hands = max(max_hands, hands_count)

            _draw_annotation_overlay(
                frame,
                timestamp_sec=ts,
                sampled_index=sampled_frames,
                hands_count=hands_count,
                backend=hand_backend,
                fallback_mode=(hands is None and tasks_landmarker is None),
            )

            frame_results.append(
                {
                    "frame_id": frame_id,
                    "timestamp": ts,
                    "hands_count": hands_count,
                    "landmarks": landmarks_payload,
                }
            )

            if video_export_active:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer, writer_codec = _open_video_writer(
                        path=annotated_path,
                        fps=float(target_fps),
                        size=(w, h),
                        warnings=warnings,
                    )
                    if writer is None:
                        video_export_active = False
                if writer is not None:
                    writer.write(frame)
            if progress_callback:
                total_text = str(source_total_frames) if source_total_frames > 0 else "?"
                progress_callback(
                    frame_id + 1,
                    source_total_frames,
                    sampled_frames,
                    f"手部检测中：已采样 {sampled_frames} 帧（源帧总数 {total_text}）。",
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if hands is not None:
            hands.close()
        if tasks_landmarker is not None:
            tasks_landmarker.close()

    transcoded_for_web = False
    if video_export_active and annotated_path.exists() and annotated_path.stat().st_size > 0:
        transcoded_for_web = _try_transcode_web_mp4(annotated_path, warnings)

    summary = {
        "total_sampled_frames": sampled_frames,
        "frames_with_hands": hand_frames,
        "hand_presence_ratio": float(hand_frames / sampled_frames) if sampled_frames else 0.0,
        "max_hands_in_frame": max_hands,
        "mediapipe_enabled": (hands is not None or tasks_landmarker is not None),
        "hand_backend": hand_backend,
        "writer_codec": writer_codec or "none",
        "web_transcoded": bool(transcoded_for_web),
        "warnings": warnings,
    }

    payload = {
        "video_path": video_path,
        "summary": summary,
        "frame_results": frame_results,
        "annotated_video": (
            str(annotated_path).replace("\\", "/")
            if (video_export_active and annotated_path.exists() and annotated_path.stat().st_size > 0)
            else None
        ),
        "warnings": warnings,
    }
    hand_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
