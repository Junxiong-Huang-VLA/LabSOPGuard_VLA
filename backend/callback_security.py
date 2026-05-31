from __future__ import annotations

import ipaddress
import logging
import socket
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlsplit

import requests


EnvGetter = Callable[[str, Optional[str]], Optional[str]]
Resolver = Callable[..., list]
PostCallable = Callable[..., Any]


def _env_bool(env_get: EnvGetter, name: str, default: str = "false") -> bool:
    return str(env_get(name, default) or default).strip().lower() in {"1", "true", "yes", "on"}


def _is_blocked_ip(addr: ipaddress._BaseAddress) -> bool:
    return any(
        [
            addr.is_private,
            addr.is_loopback,
            addr.is_link_local,
            addr.is_multicast,
            addr.is_reserved,
            addr.is_unspecified,
        ]
    )


def _parse_allowed_hosts(env_get: EnvGetter, env_name: str) -> List[str]:
    raw = env_get(env_name, "") or ""
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def host_matches_allowed_patterns(host: str, patterns: List[str]) -> bool:
    if not patterns:
        return True
    normalized = host.strip().lower()
    for pattern in patterns:
        if pattern.startswith("*."):
            suffix = pattern[1:]
            if normalized.endswith(suffix) and normalized != pattern[2:]:
                return True
        elif normalized == pattern:
            return True
    return False


def _parse_allowed_cidrs(env_get: EnvGetter, env_name: str, error_prefix: str) -> List[ipaddress._BaseNetwork]:
    raw = env_get(env_name, "") or ""
    cidrs: List[ipaddress._BaseNetwork] = []
    for item in [token.strip() for token in raw.split(",") if token.strip()]:
        try:
            cidrs.append(ipaddress.ip_network(item, strict=False))
        except ValueError as exc:
            raise ValueError(f"invalid {error_prefix} CIDR: {item}") from exc
    return cidrs


def _ip_in_allowed_cidrs(addr: ipaddress._BaseAddress, cidrs: List[ipaddress._BaseNetwork]) -> bool:
    if not cidrs:
        return False
    return any(addr in network for network in cidrs)


def validate_callback_url(
    value: str,
    *,
    env_get: EnvGetter,
    resolver: Resolver = socket.getaddrinfo,
    env_prefix: str = "REALITYLOOP_CALLBACK",
    field_name: str = "callback_url",
    cidr_error_prefix: str = "callback",
) -> str:
    callback_url = str(value or "").strip()
    if not callback_url:
        raise ValueError(f"{field_name} is empty")
    if len(callback_url) > 2048:
        raise ValueError(f"{field_name} is too long")

    parsed = urlsplit(callback_url)
    scheme = (parsed.scheme or "").lower()
    allow_http = _env_bool(env_get, f"{env_prefix}_ALLOW_HTTP")
    allowed_schemes = {"https", "http"} if allow_http else {"https"}
    if scheme not in allowed_schemes:
        raise ValueError(f"{field_name} scheme must be one of: {', '.join(sorted(allowed_schemes))}")
    if not parsed.hostname:
        raise ValueError(f"{field_name} hostname is required")

    host = parsed.hostname.strip().lower()
    if host in {"localhost", "localhost.localdomain"}:
        raise ValueError(f"{field_name} host is not allowed")

    allowed_hosts = _parse_allowed_hosts(env_get, f"{env_prefix}_ALLOWED_HOSTS")
    if allowed_hosts and not host_matches_allowed_patterns(host, allowed_hosts):
        raise ValueError(f"{field_name} host is not in allowed host whitelist")

    allow_private = _env_bool(env_get, f"{env_prefix}_ALLOW_PRIVATE")
    allowed_cidrs = _parse_allowed_cidrs(env_get, f"{env_prefix}_ALLOWED_CIDRS", cidr_error_prefix)
    try:
        direct_ip = ipaddress.ip_address(host)
    except ValueError:
        direct_ip = None

    if direct_ip is not None:
        allowed_by_cidr = _ip_in_allowed_cidrs(direct_ip, allowed_cidrs)
        if allowed_cidrs and not allowed_by_cidr:
            raise ValueError(f"{field_name} IP is not in allowed CIDR whitelist")
        if _is_blocked_ip(direct_ip) and not allow_private and not allowed_by_cidr:
            raise ValueError(f"{field_name} points to a private or local IP")
        return callback_url

    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        resolved = resolver(host, port, proto=socket.IPPROTO_TCP)
    except Exception as exc:
        raise ValueError(f"{field_name} host resolve failed: {exc}") from exc
    if not resolved:
        raise ValueError(f"{field_name} host resolve returned no addresses")

    resolved_addrs: List[ipaddress._BaseAddress] = []
    for item in resolved:
        sockaddr = item[4]
        if not sockaddr:
            continue
        try:
            addr = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        resolved_addrs.append(addr)
    if not resolved_addrs:
        raise ValueError(f"{field_name} resolve yielded no valid IP addresses")
    if allowed_cidrs and not any(_ip_in_allowed_cidrs(addr, allowed_cidrs) for addr in resolved_addrs):
        raise ValueError(f"{field_name} resolved IPs are not in allowed CIDR whitelist")
    for addr in resolved_addrs:
        allowed_by_cidr = _ip_in_allowed_cidrs(addr, allowed_cidrs)
        if _is_blocked_ip(addr) and not allow_private and not allowed_by_cidr:
            raise ValueError(f"{field_name} resolves to private/local address")
    return callback_url


def send_callback_notification(
    callback_url: str,
    task_id: str,
    violations: List[Dict[str, Any]],
    *,
    env_get: EnvGetter,
    resolver: Resolver = socket.getaddrinfo,
    post: PostCallable = requests.post,
    logger: logging.Logger | None = None,
) -> None:
    payload = {
        "task_id": task_id,
        "violations_count": len(violations),
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
    }
    log = logger or logging.getLogger(__name__)
    try:
        safe_callback_url = validate_callback_url(callback_url, env_get=env_get, resolver=resolver)
        post(safe_callback_url, json=payload, timeout=10, allow_redirects=False)
    except ValueError as exc:
        log.warning("callback url blocked: %s", exc)
    except Exception as exc:
        log.warning("callback notification failed: %s", exc)
