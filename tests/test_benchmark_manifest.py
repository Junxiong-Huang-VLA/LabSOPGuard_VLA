"""Tests for benchmark manifest construction + validation (P0.1) and the
no-hardcoded-expected-count guard (P0.3).

No GPU, no video decode — these tests only exercise manifest logic and a
static source-tree scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from key_action_indexer import benchmark_manifest as bm


def _make_view(tmp_path: Path, role: str, camera: str, with_ts: bool = True) -> bm.ViewSpec:
    seg = tmp_path / f"sender_{camera}" / "2026-05-22" / "000_virtual"
    seg.mkdir(parents=True)
    video = seg / "rgb.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # token bytes; never decoded
    header = "frame_id,packet_system_timestamp_us,width,height" if with_ts else "frame_id,width,height"
    (seg / "frames.csv").write_text(header + "\n1,1779433040743025,1920,1080\n", encoding="utf-8")
    (seg / "meta.json").write_text(json.dumps({
        "camera_id": camera, "segment_start_us": 1779433040743025,
        "rgb_actual_fps": 28.49, "rgb_fps": 30,
    }), encoding="utf-8")
    return bm.ViewSpec(role=role, video_path=video,
                       frames_csv_path=seg / "frames.csv",
                       meta_path=seg / "meta.json", camera_id=camera)


def _weights(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"PK\x03\x04fake-weights")
    return p


def test_discover_assigns_roles_explicitly(tmp_path):
    _make_view(tmp_path, "first_person", "cam02")
    _make_view(tmp_path, "third_person", "cam01")
    specs = bm.discover_virtual_camera_pair(
        tmp_path, first_person_camera="cam02", third_person_camera="cam01")
    assert specs["first_person"].camera_id == "cam02"
    assert specs["third_person"].camera_id == "cam01"


def test_build_manifest_wires_weights(tmp_path):
    specs = {
        "first_person": _make_view(tmp_path, "first_person", "cam02"),
        "third_person": _make_view(tmp_path, "third_person", "cam01"),
    }
    w1 = _weights(tmp_path, "first_lab.pt")
    w3 = _weights(tmp_path, "third_lab.pt")
    m = bm.build_session_manifest(
        session_id="s1", specs=specs, output_dir=tmp_path / "out",
        first_person_weights=w1, third_person_weights=w3)
    dc = m["detection_config"]
    assert dc["detector_backend"] == "yolo"
    assert dc["yolo_first_person_model_path"] == str(w1)
    assert dc["yolo_third_person_model_path"] == str(w3)


def test_validation_passes_for_good_manifest(tmp_path):
    specs = {
        "first_person": _make_view(tmp_path, "first_person", "cam02"),
        "third_person": _make_view(tmp_path, "third_person", "cam01"),
    }
    m = bm.build_session_manifest(
        session_id="s1", specs=specs, output_dir=tmp_path / "out",
        first_person_weights=_weights(tmp_path, "first_lab.pt"),
        third_person_weights=_weights(tmp_path, "third_lab.pt"))
    v = bm.validate_manifest_for_benchmark(m)
    assert v.ok, v.errors


def test_validation_rejects_generic_coco_weights(tmp_path):
    specs = {
        "first_person": _make_view(tmp_path, "first_person", "cam02"),
        "third_person": _make_view(tmp_path, "third_person", "cam01"),
    }
    # COCO fallback weight name — the exact trap we must catch
    coco = _weights(tmp_path, "yolo26s.pt")
    m = bm.build_session_manifest(
        session_id="s1", specs=specs, output_dir=tmp_path / "out",
        first_person_weights=coco,
        third_person_weights=_weights(tmp_path, "third_lab.pt"))
    v = bm.validate_manifest_for_benchmark(m)
    assert not v.ok
    assert any("generic COCO" in e for e in v.errors)


def test_validation_rejects_missing_weights(tmp_path):
    specs = {
        "first_person": _make_view(tmp_path, "first_person", "cam02"),
        "third_person": _make_view(tmp_path, "third_person", "cam01"),
    }
    m = bm.build_session_manifest(
        session_id="s1", specs=specs, output_dir=tmp_path / "out",
        first_person_weights=_weights(tmp_path, "first_lab.pt"),
        third_person_weights=tmp_path / "does_not_exist.pt")
    v = bm.validate_manifest_for_benchmark(m)
    assert not v.ok
    assert any("does not exist" in e for e in v.errors)


def test_validation_rejects_missing_timestamp_field(tmp_path):
    specs = {
        "first_person": _make_view(tmp_path, "first_person", "cam02", with_ts=False),
        "third_person": _make_view(tmp_path, "third_person", "cam01"),
    }
    m = bm.build_session_manifest(
        session_id="s1", specs=specs, output_dir=tmp_path / "out",
        first_person_weights=_weights(tmp_path, "first_lab.pt"),
        third_person_weights=_weights(tmp_path, "third_lab.pt"))
    v = bm.validate_manifest_for_benchmark(m)
    assert not v.ok
    assert any("timestamp field" in e for e in v.errors)


def test_validation_warns_when_fine_fps_below_coarse(tmp_path):
    specs = {
        "first_person": _make_view(tmp_path, "first_person", "cam02"),
        "third_person": _make_view(tmp_path, "third_person", "cam01"),
    }
    m = bm.build_session_manifest(
        session_id="s1", specs=specs, output_dir=tmp_path / "out",
        first_person_weights=_weights(tmp_path, "first_lab.pt"),
        third_person_weights=_weights(tmp_path, "third_lab.pt"),
        coarse_sample_fps=6.0, fine_sample_fps=1.0)  # inverted on purpose
    v = bm.validate_manifest_for_benchmark(m)
    assert v.ok  # warning, not error
    assert any("fine scan" in w for w in v.warnings)
