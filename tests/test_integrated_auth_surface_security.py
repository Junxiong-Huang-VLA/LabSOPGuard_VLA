from __future__ import annotations

import integrated_system.app_integrated as app_integrated


def _build_client(monkeypatch, token: str = "test-token"):
    monkeypatch.setattr(app_integrated, "_bootstrap_tasks_from_disk", lambda: 0)
    monkeypatch.setattr(app_integrated, "_load_alert_notify_state", lambda: None)
    monkeypatch.setattr(app_integrated, "_load_audit_events_from_disk", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_integrated, "_append_audit_event", lambda *args, **kwargs: None)

    settings = app_integrated.IntegratedSettings(api_token=token)
    app = app_integrated.create_app(settings=settings)
    app.config.update(TESTING=True)
    return app.test_client(), {"X-API-Token": token}


def test_adaptive_monitor_write_endpoints_require_auth(monkeypatch):
    class _Monitor:
        def start_monitoring(self):
            return None

        def stop_monitoring(self):
            return None

        def reset_statistics(self):
            return None

    monkeypatch.setattr(app_integrated, "get_monitor", lambda: _Monitor())
    client, auth = _build_client(monkeypatch)

    for path in [
        "/api/adaptive_monitor/start",
        "/api/adaptive_monitor/stop",
        "/api/adaptive_monitor/reset",
        "/api/adaptive_monitor/process_frame",
    ]:
        resp = client.post(path, json={})
        assert resp.status_code == 401

    assert client.post("/api/adaptive_monitor/start", json={}, headers=auth).status_code == 200
    assert client.post("/api/adaptive_monitor/stop", json={}, headers=auth).status_code == 200
    assert client.post("/api/adaptive_monitor/reset", json={}, headers=auth).status_code == 200
    assert client.post("/api/adaptive_monitor/process_frame", json={}, headers=auth).status_code == 400


def test_adaptive_monitor_read_endpoints_require_auth(monkeypatch):
    class _Constraint:
        constraint_id = "c1"
        description = "constraint"
        severity = "high"
        enabled = True

    class _Monitor:
        violation_history = []
        constraints = [_Constraint()]

        def get_statistics(self):
            return {"ok": True}

    monkeypatch.setattr(app_integrated, "get_monitor", lambda: _Monitor())
    client, auth = _build_client(monkeypatch)

    assert client.get("/api/adaptive_monitor/status").status_code == 401
    assert client.get("/api/adaptive_monitor/report").status_code == 401

    assert client.get("/api/adaptive_monitor/status", headers=auth).status_code == 200
    assert client.get("/api/adaptive_monitor/report", headers=auth).status_code == 200


def test_voice_endpoints_require_auth(monkeypatch):
    client, auth = _build_client(monkeypatch)

    targets = [
        ("/api/tts/speak", {"text": ""}),
        ("/api/tts/stream", {"text": ""}),
        ("/api/asr/transcribe", {"audio": ""}),
        ("/api/voice/chat", {"audio": ""}),
    ]
    for path, payload in targets:
        resp = client.post(path, json=payload)
        assert resp.status_code == 401

    assert client.post("/api/tts/speak", json={"text": ""}, headers=auth).status_code == 400
    assert client.post("/api/tts/stream", json={"text": ""}, headers=auth).status_code == 400
    assert client.post("/api/asr/transcribe", json={"audio": ""}, headers=auth).status_code == 400
    assert client.post("/api/voice/chat", json={"audio": ""}, headers=auth).status_code == 400
