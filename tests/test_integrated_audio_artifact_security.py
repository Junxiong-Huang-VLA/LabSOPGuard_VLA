from __future__ import annotations

from pathlib import Path

import integrated_system.app_integrated as app_integrated


def test_temp_audio_artifact_token_flow(tmp_path: Path) -> None:
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"fake-audio")

    token = app_integrated._register_temp_audio_artifact(audio, ttl_sec=120)
    assert token

    resolved = app_integrated._resolve_temp_audio_artifact(token)
    assert resolved is not None
    assert resolved == audio.resolve()


def test_temp_audio_legacy_path_must_be_registered(tmp_path: Path) -> None:
    audio = tmp_path / "legacy.mp3"
    audio.write_bytes(b"fake-audio")

    not_registered = app_integrated._resolve_temp_audio_artifact_by_legacy_path(str(audio))
    assert not_registered is None

    token = app_integrated._register_temp_audio_artifact(audio, ttl_sec=120)
    assert token
    registered = app_integrated._resolve_temp_audio_artifact_by_legacy_path(str(audio))
    assert registered == audio.resolve()


def test_temp_audio_artifact_rejects_non_audio_suffix(tmp_path: Path) -> None:
    text_file = tmp_path / "secret.txt"
    text_file.write_text("not audio", encoding="utf-8")

    token = app_integrated._register_temp_audio_artifact(text_file, ttl_sec=120)
    assert token is None
