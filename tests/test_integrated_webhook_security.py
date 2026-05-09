from __future__ import annotations

import types

import integrated_system.app_integrated as app_integrated


def test_validate_monitor_webhook_url_blocks_private_ip_by_default(monkeypatch):
    monkeypatch.delenv("INTEGRATED_MONITOR_WEBHOOK_ALLOW_PRIVATE", raising=False)
    monkeypatch.delenv("INTEGRATED_MONITOR_WEBHOOK_ALLOWED_CIDRS", raising=False)
    try:
        app_integrated._validate_monitor_webhook_url("https://127.0.0.1:8443/hook")
        assert False, "expected ValueError for private IP"
    except ValueError as exc:
        assert "private" in str(exc).lower() or "local" in str(exc).lower()


def test_validate_monitor_webhook_url_allows_public_ip(monkeypatch):
    monkeypatch.delenv("INTEGRATED_MONITOR_WEBHOOK_ALLOW_PRIVATE", raising=False)
    monkeypatch.delenv("INTEGRATED_MONITOR_WEBHOOK_ALLOWED_CIDRS", raising=False)
    value = app_integrated._validate_monitor_webhook_url("https://8.8.8.8/hook")
    assert value == "https://8.8.8.8/hook"


def test_validate_monitor_webhook_url_respects_allowed_host_whitelist(monkeypatch):
    def fake_getaddrinfo(host, port, proto=0):
        return [(None, None, None, None, ("93.184.216.34", port))]

    monkeypatch.setattr(app_integrated.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setenv("INTEGRATED_MONITOR_WEBHOOK_ALLOWED_HOSTS", "hooks.example.com,*.trusted.example")
    monkeypatch.delenv("INTEGRATED_MONITOR_WEBHOOK_ALLOWED_CIDRS", raising=False)
    monkeypatch.delenv("INTEGRATED_MONITOR_WEBHOOK_ALLOW_PRIVATE", raising=False)

    assert (
        app_integrated._validate_monitor_webhook_url("https://hooks.example.com/hook")
        == "https://hooks.example.com/hook"
    )
    assert (
        app_integrated._validate_monitor_webhook_url("https://sub.trusted.example/hook")
        == "https://sub.trusted.example/hook"
    )
    try:
        app_integrated._validate_monitor_webhook_url("https://evil.example/hook")
        assert False, "expected ValueError for disallowed host"
    except ValueError as exc:
        assert "allowed host whitelist" in str(exc)


def test_maybe_send_monitor_webhooks_blocks_invalid_url_without_network(monkeypatch):
    called: list[str] = []

    def fake_urlopen(*args, **kwargs):
        called.append("called")
        return types.SimpleNamespace(status=200)

    monkeypatch.setattr(app_integrated.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("INTEGRATED_MONITOR_WEBHOOK_ALLOW_PRIVATE", raising=False)
    monkeypatch.delenv("INTEGRATED_MONITOR_WEBHOOK_ALLOWED_CIDRS", raising=False)

    settings = types.SimpleNamespace(
        monitor_webhook_url="https://127.0.0.1:9443/hook",
        monitor_webhook_min_level="low",
        monitor_webhook_cooldown_sec=60,
    )
    payload = {
        "alerts": [{"type": "runtime_failure", "task_id": "task-1", "level": "high"}],
        "task_stats": {"running": 0},
    }
    result = app_integrated._maybe_send_monitor_webhooks(settings, payload)
    assert result.get("blocked") is True
    assert result.get("sent") == 0
    assert called == []
