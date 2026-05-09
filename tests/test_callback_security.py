from __future__ import annotations

import pytest

from backend.callback_security import validate_callback_url


pytestmark = pytest.mark.unit


def _env(values: dict[str, str]):
    def env_get(name: str, default=None):
        return values.get(name, default)

    return env_get


def test_callback_security_blocks_private_direct_ip_by_default():
    with pytest.raises(ValueError, match="private or local IP"):
        validate_callback_url("https://127.0.0.1/hook", env_get=_env({}))


def test_callback_security_respects_wildcard_host_and_custom_prefix():
    def fake_resolver(host, port, proto=0):
        return [(None, None, None, None, ("93.184.216.34", port))]

    env_get = _env({"INTEGRATED_MONITOR_WEBHOOK_ALLOWED_HOSTS": "*.trusted.example"})

    assert (
        validate_callback_url(
            "https://ops.trusted.example/hook",
            env_get=env_get,
            resolver=fake_resolver,
            env_prefix="INTEGRATED_MONITOR_WEBHOOK",
            field_name="monitor_webhook_url",
            cidr_error_prefix="monitor webhook",
        )
        == "https://ops.trusted.example/hook"
    )
    with pytest.raises(ValueError, match="allowed host whitelist"):
        validate_callback_url(
            "https://evil.example/hook",
            env_get=env_get,
            resolver=fake_resolver,
            env_prefix="INTEGRATED_MONITOR_WEBHOOK",
            field_name="monitor_webhook_url",
            cidr_error_prefix="monitor webhook",
        )


def test_callback_security_allows_private_ip_only_when_explicitly_enabled():
    assert (
        validate_callback_url(
            "https://10.0.0.5/hook",
            env_get=_env({"REALITYLOOP_CALLBACK_ALLOW_PRIVATE": "true"}),
        )
        == "https://10.0.0.5/hook"
    )
