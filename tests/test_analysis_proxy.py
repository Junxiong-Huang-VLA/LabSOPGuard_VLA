from __future__ import annotations

import subprocess
from pathlib import Path

from key_action_indexer.analysis_proxy import (
    analysis_proxy_cache_payload,
    build_analysis_proxy_for_source,
    build_analysis_proxies,
)
from key_action_indexer.schemas import SessionManifest, SessionVideos, VideoSource


def _source(path: Path, name: str = "third_person") -> VideoSource:
    return VideoSource(name=name, path=str(path), start_time="2026-05-18T00:00:00+08:00", fps=30.0)


def test_analysis_proxy_skips_without_ffmpeg(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"not a real video")
    monkeypatch.setattr("key_action_indexer.analysis_proxy.shutil.which", lambda _name: None)

    proxy_source, meta = build_analysis_proxy_for_source(
        _source(source_path),
        view="third_person",
        proxy_root=tmp_path / "proxy",
        enabled=True,
        dry_run=False,
    )

    assert proxy_source.path == str(source_path)
    assert meta["status"] == "skipped_no_ffmpeg"
    assert meta["proxy_used"] is False
    assert meta["timing_row"]["pipeline_stage"] == "analysis_proxy_prepare"


def test_analysis_proxy_builds_and_reuses_cached_proxy(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video bytes")
    monkeypatch.setattr("key_action_indexer.analysis_proxy.shutil.which", lambda _name: "ffmpeg")

    def fake_run(cmd, check, stdout, stderr):
        Path(cmd[-1]).write_bytes(b"proxy bytes")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("key_action_indexer.analysis_proxy.subprocess.run", fake_run)
    proxy_source, meta = build_analysis_proxy_for_source(
        _source(source_path),
        view="third_person",
        proxy_root=tmp_path / "proxy",
        enabled=True,
        dry_run=False,
    )
    cached_proxy_source, cached_meta = build_analysis_proxy_for_source(
        _source(source_path),
        view="third_person",
        proxy_root=tmp_path / "proxy",
        enabled=True,
        dry_run=False,
    )

    assert meta["status"] == "built"
    assert Path(proxy_source.path).exists()
    assert cached_meta["status"] == "cache_hit"
    assert cached_proxy_source.path == proxy_source.path


def test_analysis_proxy_existing_only_skips_cache_miss(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video bytes")
    monkeypatch.setenv("KEY_ACTION_ANALYSIS_PROXY_EXISTING_ONLY", "1")
    monkeypatch.setattr("key_action_indexer.analysis_proxy.shutil.which", lambda _name: "ffmpeg")

    proxy_source, meta = build_analysis_proxy_for_source(
        _source(source_path),
        view="third_person",
        proxy_root=tmp_path / "proxy",
        enabled=True,
        dry_run=False,
    )

    assert proxy_source.path == str(source_path)
    assert meta["status"] == "skipped_existing_proxy_missing"
    assert meta["proxy_used"] is False
    assert meta["timing_row"]["scan_backend"] == "analysis_proxy_existing_only_cache_miss"


def test_analysis_proxy_cache_key_tracks_video_file_not_timeline_metadata(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video bytes")
    monkeypatch.setattr("key_action_indexer.analysis_proxy.shutil.which", lambda _name: "ffmpeg")

    def fake_run(cmd, check, stdout, stderr):
        Path(cmd[-1]).write_bytes(b"proxy bytes")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("key_action_indexer.analysis_proxy.subprocess.run", fake_run)
    proxy_source, meta = build_analysis_proxy_for_source(
        _source(source_path),
        view="third_person",
        proxy_root=tmp_path / "proxy",
        enabled=True,
        dry_run=False,
    )
    shifted = VideoSource(
        name="third_person",
        path=str(source_path),
        start_time="2026-05-18T01:00:00+08:00",
        fps=24.0,
        offset_sec=12.5,
        role="third_person",
        camera_id="renamed_camera",
    )
    cached_proxy_source, cached_meta = build_analysis_proxy_for_source(
        shifted,
        view="third_person",
        proxy_root=tmp_path / "proxy",
        enabled=True,
        dry_run=False,
    )

    assert meta["status"] == "built"
    assert cached_meta["status"] == "cache_hit"
    assert cached_proxy_source.path == proxy_source.path


def test_analysis_proxy_cache_payload_is_stable(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"video bytes")
    monkeypatch.setattr("key_action_indexer.analysis_proxy.shutil.which", lambda _name: "ffmpeg")

    def fake_run(cmd, check, stdout, stderr):
        Path(cmd[-1]).write_bytes(b"proxy bytes")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("key_action_indexer.analysis_proxy.subprocess.run", fake_run)
    manifest = SessionManifest(
        session_id="proxy-test",
        session_start_time="2026-05-18T00:00:00+08:00",
        videos=SessionVideos(third_person=_source(source_path, "third_person")),
    )
    _sources, summary = build_analysis_proxies(
        manifest,
        proxy_root=tmp_path / "proxy",
        views=["third_person"],
        enabled=True,
        dry_run=False,
    )
    payload = analysis_proxy_cache_payload(summary)

    assert payload["enabled"] is True
    assert payload["views"]["third_person"]["status_class"] == "proxy"
    assert "timing_rows" not in payload
