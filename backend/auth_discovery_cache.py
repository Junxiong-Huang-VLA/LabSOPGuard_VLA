from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException


OIDC_DISCOVERY_CACHE: Dict[str, Dict[str, Any]] = {}
JWKS_CACHE: Dict[str, Dict[str, Any]] = {}

EnvGetter = Callable[[str, Optional[str]], Optional[str]]
HttpGet = Callable[..., Any]
RedisEnabledGetter = Callable[[], bool]
RedisClientGetter = Callable[[], Any]


def auth_cache_dir(project_root: Path, *, env_get: EnvGetter) -> Path:
    raw = env_get("REALITYLOOP_AUTH_CACHE_DIR", None)
    path = Path(raw) if raw else project_root / "outputs" / "auth_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def auth_cache_path(prefix: str, key: str, project_root: Path, *, env_get: EnvGetter) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
    return auth_cache_dir(project_root, env_get=env_get) / f"{prefix}_{digest}.json"


def read_auth_disk_cache(prefix: str, key: str, project_root: Path, *, env_get: EnvGetter) -> Optional[Dict[str, Any]]:
    path = auth_cache_path(prefix, key, project_root, env_get=env_get)
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if float(cached.get("expires_at") or 0) <= time.time():
        return None
    payload = cached.get("payload")
    return payload if isinstance(payload, dict) else None


def auth_redis_key(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
    return f"realityloop:auth_cache:{prefix}:{digest}"


def read_auth_shared_cache(
    prefix: str,
    key: str,
    *,
    redis_enabled: RedisEnabledGetter,
    redis_client: RedisClientGetter,
) -> Optional[Dict[str, Any]]:
    if not redis_enabled():
        return None
    client = redis_client()
    if client is None:
        return None
    try:
        raw = client.get(auth_redis_key(prefix, key))
        if not raw:
            return None
        cached = json.loads(raw)
        payload = cached.get("payload")
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def write_auth_shared_cache(
    prefix: str,
    key: str,
    payload: Dict[str, Any],
    *,
    redis_enabled: RedisEnabledGetter,
    redis_client: RedisClientGetter,
    ttl_sec: int = 3600,
) -> None:
    if not redis_enabled():
        return
    client = redis_client()
    if client is None:
        return
    try:
        client.setex(
            auth_redis_key(prefix, key),
            ttl_sec,
            json.dumps(
                {
                    "schema_version": "auth_distributed_cache.v1",
                    "cache_key": key,
                    "payload": payload,
                },
                ensure_ascii=False,
            ),
        )
    except Exception:
        return


def write_auth_disk_cache(
    prefix: str,
    key: str,
    payload: Dict[str, Any],
    project_root: Path,
    *,
    env_get: EnvGetter,
    ttl_sec: int = 3600,
) -> None:
    path = auth_cache_path(prefix, key, project_root, env_get=env_get)
    cache_payload = {
        "schema_version": "auth_discovery_cache.v1",
        "cache_key": key,
        "cache_prefix": prefix,
        "key_hash": hashlib.sha1(key.encode("utf-8")).hexdigest()[:24],
        "expires_at": time.time() + ttl_sec,
        "payload": payload,
    }
    path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def oidc_discovery(
    issuer_url: str,
    *,
    project_root: Path,
    env_get: EnvGetter,
    http_get: HttpGet,
    redis_enabled: RedisEnabledGetter,
    redis_client: RedisClientGetter,
) -> Dict[str, Any]:
    issuer = issuer_url.rstrip("/")
    cached = OIDC_DISCOVERY_CACHE.get(issuer)
    if cached and cached.get("expires_at", 0) > time.time():
        return cached["payload"]
    shared_cached = read_auth_shared_cache(
        "oidc",
        issuer,
        redis_enabled=redis_enabled,
        redis_client=redis_client,
    )
    if shared_cached:
        OIDC_DISCOVERY_CACHE[issuer] = {"payload": shared_cached, "expires_at": time.time() + 3600}
        write_auth_disk_cache("oidc", issuer, shared_cached, project_root, env_get=env_get)
        return shared_cached
    disk_cached = read_auth_disk_cache("oidc", issuer, project_root, env_get=env_get)
    if disk_cached:
        OIDC_DISCOVERY_CACHE[issuer] = {"payload": disk_cached, "expires_at": time.time() + 3600}
        write_auth_shared_cache(
            "oidc",
            issuer,
            disk_cached,
            redis_enabled=redis_enabled,
            redis_client=redis_client,
        )
        return disk_cached
    url = f"{issuer}/.well-known/openid-configuration"
    try:
        response = http_get(url, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"OIDC discovery failed: {exc}") from exc
    OIDC_DISCOVERY_CACHE[issuer] = {"payload": payload, "expires_at": time.time() + 3600}
    write_auth_shared_cache(
        "oidc",
        issuer,
        payload,
        redis_enabled=redis_enabled,
        redis_client=redis_client,
    )
    write_auth_disk_cache("oidc", issuer, payload, project_root, env_get=env_get)
    return payload


def jwks_payload(
    *,
    project_root: Path,
    env_get: EnvGetter,
    http_get: HttpGet,
    redis_enabled: RedisEnabledGetter,
    redis_client: RedisClientGetter,
) -> Dict[str, Any]:
    jwks_url = env_get("REALITYLOOP_JWKS_URL", None)
    issuer = env_get("REALITYLOOP_OAUTH_ISSUER_URL", None) or env_get("REALITYLOOP_JWT_ISSUER", None)
    if not jwks_url and issuer:
        jwks_url = oidc_discovery(
            issuer,
            project_root=project_root,
            env_get=env_get,
            http_get=http_get,
            redis_enabled=redis_enabled,
            redis_client=redis_client,
        ).get("jwks_uri")
    if not jwks_url:
        raise HTTPException(status_code=401, detail="JWKS URL or OAuth issuer is not configured")
    cached = JWKS_CACHE.get(jwks_url)
    if cached and cached.get("expires_at", 0) > time.time():
        return cached["payload"]
    shared_cached = read_auth_shared_cache(
        "jwks",
        jwks_url,
        redis_enabled=redis_enabled,
        redis_client=redis_client,
    )
    if shared_cached:
        JWKS_CACHE[jwks_url] = {"payload": shared_cached, "expires_at": time.time() + 3600}
        write_auth_disk_cache("jwks", jwks_url, shared_cached, project_root, env_get=env_get)
        return shared_cached
    disk_cached = read_auth_disk_cache("jwks", jwks_url, project_root, env_get=env_get)
    if disk_cached:
        JWKS_CACHE[jwks_url] = {"payload": disk_cached, "expires_at": time.time() + 3600}
        write_auth_shared_cache(
            "jwks",
            jwks_url,
            disk_cached,
            redis_enabled=redis_enabled,
            redis_client=redis_client,
        )
        return disk_cached
    try:
        response = http_get(jwks_url, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"JWKS fetch failed: {exc}") from exc
    JWKS_CACHE[jwks_url] = {"payload": payload, "expires_at": time.time() + 3600}
    write_auth_shared_cache(
        "jwks",
        jwks_url,
        payload,
        redis_enabled=redis_enabled,
        redis_client=redis_client,
    )
    write_auth_disk_cache("jwks", jwks_url, payload, project_root, env_get=env_get)
    return payload


def refresh_auth_cache_entry(
    path: Path,
    *,
    project_root: Path,
    env_get: EnvGetter,
    http_get: HttpGet,
    redis_enabled: RedisEnabledGetter,
    redis_client: RedisClientGetter,
) -> None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        prefix = cached.get("cache_prefix")
        key = cached.get("cache_key")
    except Exception:
        return
    if prefix not in {"oidc", "jwks"} or not key:
        return
    try:
        if prefix == "oidc":
            response = http_get(f"{str(key).rstrip('/')}/.well-known/openid-configuration", timeout=5)
        else:
            response = http_get(str(key), timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return
    if prefix == "oidc":
        OIDC_DISCOVERY_CACHE[str(key).rstrip("/")] = {"payload": payload, "expires_at": time.time() + 3600}
    else:
        JWKS_CACHE[str(key)] = {"payload": payload, "expires_at": time.time() + 3600}
    write_auth_shared_cache(
        prefix,
        str(key),
        payload,
        redis_enabled=redis_enabled,
        redis_client=redis_client,
    )
    write_auth_disk_cache(prefix, str(key), payload, project_root, env_get=env_get)


async def auth_cache_refresh_loop(
    *,
    project_root: Path,
    env_get: EnvGetter,
    http_get: HttpGet,
    redis_enabled: RedisEnabledGetter,
    redis_client: RedisClientGetter,
    logger: logging.Logger,
) -> None:
    interval = max(60, int(env_get("REALITYLOOP_AUTH_CACHE_REFRESH_INTERVAL_SEC", "900") or "900"))
    while True:
        try:
            for path in auth_cache_dir(project_root, env_get=env_get).glob("*.json"):
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    expires_at = float(cached.get("expires_at") or 0)
                except Exception:
                    continue
                if expires_at - time.time() <= interval:
                    await asyncio.to_thread(
                        refresh_auth_cache_entry,
                        path,
                        project_root=project_root,
                        env_get=env_get,
                        http_get=http_get,
                        redis_enabled=redis_enabled,
                        redis_client=redis_client,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Auth cache background refresh failed")
        await asyncio.sleep(interval)
