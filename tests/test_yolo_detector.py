from __future__ import annotations

from key_action_indexer.yolo_detector import (
    EXPERIMENT_CONTEXT_LABELS,
    INTERACTION_OBJECT_LABELS,
    _detections_from_model,
    _detections_from_model_batch,
    _ffmpeg_chunk_sec_for_scan,
    _ffmpeg_scale_width_for_scan,
    _ffmpeg_worker_count_for_scan,
    _resolve_ffmpeg_sparse_mode_for_scan,
    _use_sparse_scan_for_scan,
    canonical_yolo_label,
    filter_implausible_detections,
    find_hand_object_interactions,
    normalize_yolo_detection,
    resolve_adaptive_yolo_imgsz,
    scan_yolo_video,
)


def test_canonical_yolo_label_normalizes_lab_synonyms() -> None:
    assert canonical_yolo_label("glove") == "gloved_hand"
    assert canonical_yolo_label("sample bottle") == "sample_bottle"
    assert canonical_yolo_label("tube-cap") == "tube_cap"
    assert canonical_yolo_label("magnetic stir bar") == "magnetic_stir_bar"
    assert canonical_yolo_label("equipment panel") == "panel"


def test_experiment_context_keeps_magnetic_stir_bar_in_21_detection_classes() -> None:
    assert "magnetic_stir_bar" in INTERACTION_OBJECT_LABELS
    assert "magnetic_stir_bar" in EXPERIMENT_CONTEXT_LABELS
    assert len(EXPERIMENT_CONTEXT_LABELS) == 21


def test_find_hand_object_interactions_scores_nearby_hand_and_container() -> None:
    detections = [
        normalize_yolo_detection({"label": "gloved_hand", "confidence": 0.9, "bbox": [10, 10, 40, 40]}),
        normalize_yolo_detection({"label": "container", "confidence": 0.8, "bbox": [35, 12, 70, 42]}),
        normalize_yolo_detection({"label": "balance", "confidence": 0.8, "bbox": [200, 200, 260, 260]}),
    ]

    interactions = find_hand_object_interactions(detections, frame_width=320, frame_height=240)

    assert interactions
    assert interactions[0]["object_label"] == "container"
    assert interactions[0]["score"] > 0.1


def test_find_hand_object_interactions_uses_object_overlap_for_weighing_paper() -> None:
    detections = [
        normalize_yolo_detection({"label": "gloved_hand", "confidence": 0.92, "bbox": [100, 100, 220, 220]}),
        normalize_yolo_detection({"label": "paper", "confidence": 0.88, "bbox": [200, 200, 240, 240]}),
    ]

    interactions = find_hand_object_interactions(detections, frame_width=640, frame_height=480)

    assert interactions
    assert interactions[0]["object_label"] == "paper"
    assert interactions[0]["iou"] < 0.05
    assert interactions[0]["object_overlap_ratio"] >= 0.25
    assert interactions[0]["score"] > 0.5


def test_find_hand_object_interactions_includes_magnetic_stir_bar() -> None:
    detections = [
        normalize_yolo_detection({"label": "hand", "confidence": 0.9, "bbox": [20, 20, 80, 80]}),
        normalize_yolo_detection({"label": "magnetic_stir_bar", "confidence": 0.86, "bbox": [60, 54, 105, 70]}),
    ]

    interactions = find_hand_object_interactions(detections, frame_width=320, frame_height=240)

    assert interactions
    assert interactions[0]["object_label"] == "magnetic_stir_bar"


def test_implausible_detection_filter_suppresses_flat_blue_background_for_all_classes() -> None:
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (185, 135, 45)  # BGR: blue/cyan workbench-like region.
    detections = [
        {"label": "gloved_hand", "confidence": 0.82, "bbox": [20, 20, 210, 130]},
        {"label": "beaker", "confidence": 0.81, "bbox": [25, 30, 230, 145]},
        {"label": "sample_bottle_blue", "confidence": 0.84, "bbox": [250, 40, 282, 108]},
    ]

    kept, ignored = filter_implausible_detections(
        detections,
        frame=frame,
        source_view="third_person",
    )

    assert [item["label"] for item in kept] == ["sample_bottle_blue"]
    assert {item["label"] for item in ignored} == {"gloved_hand", "beaker"}


def test_first_person_low_confidence_blue_hand_region_is_rejected() -> None:
    import numpy as np

    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    frame[:, :] = (155, 110, 70)
    frame[255:460, 250:565] = (105, 75, 50)

    kept, ignored = filter_implausible_detections(
        [{"label": "gloved_hand", "confidence": 0.39, "bbox": [252, 248, 563, 465]}],
        frame=frame,
        source_view="first_person",
    )

    assert kept == []
    assert ignored[0]["ignore_reason"] == "implausible_hand_bbox_or_background"


def test_first_person_high_confidence_bottom_hand_survives_blue_background_filter() -> None:
    import numpy as np

    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    frame[:, :] = (155, 110, 70)
    frame[388:539, 242:615] = (115, 74, 45)

    kept, ignored = filter_implausible_detections(
        [{"label": "gloved_hand", "confidence": 0.94, "bbox": [242, 387, 615, 538]}],
        frame=frame,
        source_view="first_person",
    )

    assert [item["label"] for item in kept] == ["gloved_hand"]
    assert ignored == []


def test_low_resolution_first_person_blue_beaker_region_is_not_a_hand() -> None:
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = (120, 88, 26)
    frame[242:440, 242:491] = (92, 80, 11)
    frame[260:434, 322:500] = (94, 82, 12)

    kept, ignored = filter_implausible_detections(
        [{"label": "gloved_hand", "confidence": 0.744, "bbox": [242, 242, 490, 440]}],
        frame=frame,
        source_view="first_person",
    )

    assert kept == []
    assert ignored[0]["ignore_reason"] == "implausible_hand_bbox_or_background"


def test_implausible_detection_filter_rejects_blue_workbench_as_physical_objects() -> None:
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (185, 135, 45)
    detections = [
        {"label": "paper", "confidence": 0.91, "bbox": [45, 55, 145, 115]},
        {"label": "sample_bottle", "confidence": 0.88, "bbox": [160, 45, 230, 120]},
        {"label": "balance", "confidence": 0.84, "bbox": [70, 130, 235, 210]},
    ]

    kept, ignored = filter_implausible_detections(detections, frame=frame, source_view="third_person")

    assert kept == []
    assert {item["label"] for item in ignored} == {"paper", "sample_bottle", "balance"}


def test_implausible_detection_filter_keeps_real_white_weighing_paper_on_blue_workbench() -> None:
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (185, 135, 45)
    frame[70:122, 86:162] = (242, 244, 246)
    frame[70:122, 86] = (210, 210, 210)
    frame[70:122, 161] = (210, 210, 210)
    frame[70, 86:162] = (210, 210, 210)
    frame[121, 86:162] = (210, 210, 210)

    kept, ignored = filter_implausible_detections(
        [{"label": "paper", "confidence": 0.82, "bbox": [86, 70, 162, 122]}],
        frame=frame,
        source_view="third_person",
    )

    assert [item["label"] for item in kept] == ["paper"]
    assert ignored == []


def test_blue_workbench_hand_and_paper_do_not_create_interaction() -> None:
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (185, 135, 45)
    detections = [
        {"label": "gloved_hand", "confidence": 0.93, "bbox": [42, 52, 140, 116]},
        {"label": "paper", "confidence": 0.91, "bbox": [105, 56, 176, 120]},
    ]

    interactions = find_hand_object_interactions(detections, frame=frame, source_view="third_person")

    assert interactions == []


def test_third_person_low_confidence_edge_hand_does_not_drive_interaction() -> None:
    import numpy as np

    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    detections = [
        {"label": "gloved_hand", "confidence": 0.51, "bbox": [2, 212, 151, 354]},
        {"label": "container", "confidence": 0.76, "bbox": [0, 154, 90, 285]},
    ]

    interactions = find_hand_object_interactions(
        detections,
        frame=frame,
        source_view="third_person",
    )

    assert interactions == []


def test_adaptive_yolo_imgsz_uses_stable_lab_default_for_low_resolution_uploads() -> None:
    assert resolve_adaptive_yolo_imgsz(640, 480) == 960
    assert resolve_adaptive_yolo_imgsz(960, 540) == 960
    assert resolve_adaptive_yolo_imgsz(1920, 1080) == 1280


def test_explicit_yolo_imgsz_is_rounded_and_honored() -> None:
    assert resolve_adaptive_yolo_imgsz(640, 480, configured_imgsz=1000) == 1024
    assert resolve_adaptive_yolo_imgsz(1920, 1080, configured_imgsz=640) == 640


def test_long_video_coarse_ffmpeg_defaults_are_parallel(monkeypatch) -> None:
    for name in (
        "KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_WORKERS",
        "KEY_ACTION_YOLO_COARSE_FFMPEG_WORKERS",
        "KEY_ACTION_YOLO_FFMPEG_WORKERS",
        "KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_CHUNK_SEC",
        "KEY_ACTION_YOLO_COARSE_FFMPEG_CHUNK_SEC",
        "KEY_ACTION_YOLO_FFMPEG_CHUNK_SEC",
    ):
        monkeypatch.delenv(name, raising=False)
    model_ref = {"scan_role": "long_video_coarse"}

    assert _ffmpeg_worker_count_for_scan(model_ref) == 4
    assert _ffmpeg_chunk_sec_for_scan(model_ref) == 600.0


def test_long_video_coarse_defaults_to_ffmpeg_chunks(monkeypatch) -> None:
    for name in (
        "KEY_ACTION_FAST_LOCATE_COARSE_SEEK_SCAN",
        "KEY_ACTION_YOLO_COARSE_SEEK_SCAN",
        "KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_SPARSE_MODE",
        "KEY_ACTION_YOLO_COARSE_FFMPEG_SPARSE_MODE",
        "KEY_ACTION_YOLO_FFMPEG_SPARSE_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    model_ref = {"scan_role": "long_video_coarse"}

    assert _use_sparse_scan_for_scan(2.0, None, 3600.0, model_ref) is True
    assert _resolve_ffmpeg_sparse_mode_for_scan(2.0, 3600.0, model_ref) == "chunks"


def test_long_video_coarse_legacy_seek_requires_explicit_per_frame_mode(monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_SPARSE_MODE", "per_frame_seek")

    model_ref = {"scan_role": "long_video_coarse"}

    assert _resolve_ffmpeg_sparse_mode_for_scan(0.2, 180.0, model_ref) == "seek"


def test_fast_locate_coarse_seek_default_stays_as_seek(monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_SPARSE_MODE", "seek")

    model_ref = {"scan_role": "long_video_coarse"}

    assert _resolve_ffmpeg_sparse_mode_for_scan(0.2, 180.0, model_ref) == "seek"


def test_scan_yolo_video_records_requested_and_actual_device_for_mock_rows(tmp_path) -> None:
    timings = []

    rows = scan_yolo_video(
        video_path=tmp_path / "missing.mp4",
        mock_rows=[
            {
                "time_sec": 0.0,
                "detections": [],
                "source_view": "first_person",
            }
        ],
        sample_fps=1.0,
        device="auto",
        timing_callback=timings.append,
    )

    assert rows[0]["requested_yolo_device"] == "auto"
    assert rows[0]["actual_yolo_device"] == "dry_run"
    assert timings[0]["requested_device"] == "auto"
    assert timings[0]["actual_device"] == "dry_run"


def test_ffmpeg_scale_width_can_be_tuned_per_scan_role(monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_YOLO_FFMPEG_SCALE_WIDTH", "960")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_SCALE_WIDTH", "640")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_FINE_FFMPEG_SCALE_WIDTH", "960")

    assert _ffmpeg_scale_width_for_scan({"scan_role": "long_video_coarse"}, 1280) == 640
    assert _ffmpeg_scale_width_for_scan({"scan_role": "micro_refine"}, 1280) == 960
    assert _ffmpeg_scale_width_for_scan({"scan_role": "other"}, 1280) == 960


def test_micro_refine_ffmpeg_defaults_are_window_sized(monkeypatch) -> None:
    for name in (
        "KEY_ACTION_FAST_LOCATE_FINE_FFMPEG_WORKERS",
        "KEY_ACTION_YOLO_FINE_FFMPEG_WORKERS",
        "KEY_ACTION_YOLO_FFMPEG_WORKERS",
        "KEY_ACTION_FAST_LOCATE_FINE_FFMPEG_CHUNK_SEC",
        "KEY_ACTION_YOLO_FINE_FFMPEG_CHUNK_SEC",
        "KEY_ACTION_YOLO_FFMPEG_CHUNK_SEC",
    ):
        monkeypatch.delenv(name, raising=False)
    model_ref = {"scan_role": "micro_refine"}

    assert _ffmpeg_worker_count_for_scan(model_ref) == 4
    assert _ffmpeg_chunk_sec_for_scan(model_ref) == 30.0


def test_yolo_predict_receives_adaptive_imgsz() -> None:
    import numpy as np

    class Box:
        cls = [0]
        conf = [0.91]
        xyxy = [20, 20, 58, 92]

    class Result:
        names = {0: "sample_bottle"}
        boxes = [Box()]

    class Model:
        def __init__(self) -> None:
            self.kwargs = {}

        def predict(self, **kwargs):
            self.kwargs = kwargs
            return [Result()]

    model = Model()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    _detections_from_model(model, frame, conf=0.25, iou=0.45, device="cpu", imgsz=960)

    assert model.kwargs["imgsz"] == 960


def test_yolo_batch_fallback_records_real_predict_calls() -> None:
    class Result:
        names = {}
        boxes = []

    class Model:
        def __init__(self) -> None:
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            if isinstance(kwargs.get("source"), list):
                raise RuntimeError("batch predict unavailable")
            return [Result()]

    model = Model()
    stats = {}

    detections = _detections_from_model_batch(
        model,
        [object(), object()],
        conf=0.25,
        iou=0.45,
        device="cpu",
        stats=stats,
    )

    assert detections == [[], []]
    assert len(model.calls) == 3
    assert stats["actual_batch_sizes"] == [2]
    assert stats["batch_predict_attempts"] == 1
    assert stats["batch_predict_calls"] == 0
    assert stats["batch_fallback_count"] == 1
    assert stats["frame_predict_calls"] == 2
    assert stats["predict_call_count"] == 3
