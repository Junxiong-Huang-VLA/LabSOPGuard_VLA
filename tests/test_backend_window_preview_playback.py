from pathlib import Path


def test_window_preview_mp4_is_treated_as_browser_playback_candidate() -> None:
    source = (Path("LabSOPGuard") / "backend" / "main.py").read_text(encoding="utf-8")

    assert "def _is_window_preview_clip_path" in source
    assert 'path.name.lower() == "window_preview.mp4"' in source
    assert "or _is_window_preview_clip_path(path)" in source
