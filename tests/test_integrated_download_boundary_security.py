from __future__ import annotations

from pathlib import Path

import integrated_system.app_integrated as app_integrated


def _build_client(monkeypatch, project_root: Path, token: str = "test-token"):
    monkeypatch.setattr(app_integrated, "_bootstrap_tasks_from_disk", lambda: 0)
    monkeypatch.setattr(app_integrated, "_load_alert_notify_state", lambda: None)
    monkeypatch.setattr(app_integrated, "_load_audit_events_from_disk", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_integrated, "_append_audit_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_integrated, "PROJECT_ROOT", project_root)

    settings = app_integrated.IntegratedSettings(api_token=token)
    app = app_integrated.create_app(settings=settings)
    app.config.update(TESTING=True)
    return app.test_client(), {"X-API-Token": token}


def test_download_blocks_output_path_outside_task_dir(monkeypatch, tmp_path: Path):
    client, auth = _build_client(monkeypatch, tmp_path)

    out_dir = tmp_path / "outputs" / "task-sec-1"
    out_dir.mkdir(parents=True, exist_ok=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("top-secret", encoding="utf-8")

    with app_integrated.TASKS_LOCK:
        app_integrated.TASKS.clear()
        app_integrated.TASKS["task-sec-1"] = {
            "task_id": "task-sec-1",
            "status": "completed",
            "output_dir": app_integrated._safe_rel(out_dir),
            "outputs": {"report": app_integrated._safe_rel(secret)},
            "options": {},
        }

    resp = client.get("/api/download/task-sec-1/pdf", headers=auth)
    assert resp.status_code == 404


def test_download_bundle_ignores_paths_outside_task_dir(monkeypatch, tmp_path: Path):
    client, auth = _build_client(monkeypatch, tmp_path)

    out_dir = tmp_path / "outputs" / "task-sec-2"
    out_dir.mkdir(parents=True, exist_ok=True)
    secret = tmp_path / "other" / "sensitive.log"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("sensitive", encoding="utf-8")

    with app_integrated.TASKS_LOCK:
        app_integrated.TASKS.clear()
        app_integrated.TASKS["task-sec-2"] = {
            "task_id": "task-sec-2",
            "status": "completed",
            "output_dir": app_integrated._safe_rel(out_dir),
            "outputs": {"report": app_integrated._safe_rel(secret)},
            "options": {},
        }

    resp = client.get("/api/download_bundle/task-sec-2", headers=auth)
    assert resp.status_code == 404


def test_download_allows_file_within_task_dir(monkeypatch, tmp_path: Path):
    client, auth = _build_client(monkeypatch, tmp_path)

    out_dir = tmp_path / "outputs" / "task-sec-3"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "integrated_analysis_report.pdf"
    report.write_bytes(b"%PDF-1.4\n%test\n")

    with app_integrated.TASKS_LOCK:
        app_integrated.TASKS.clear()
        app_integrated.TASKS["task-sec-3"] = {
            "task_id": "task-sec-3",
            "status": "completed",
            "output_dir": app_integrated._safe_rel(out_dir),
            "outputs": {},
            "options": {},
        }

    resp = client.get("/api/download/task-sec-3/pdf", headers=auth)
    assert resp.status_code == 200
