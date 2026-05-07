import json
import sqlite3
import sys
import base64
import hashlib
import hmac
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from labsopguard.material_publishing import SemanticMaterialPublisher, slugify, stable_name
from labsopguard.material_publishing.archive_planner import ArchivePlanner
from labsopguard.material_publishing.semantic_enhancer import QwenVlmDisplayNameEnhancer
from labsopguard.material_publishing.uploaders import uploader_for
from labsopguard.material_maintenance import (
    rebuild_workspace_published_materials_index,
    query_workspace_published_materials,
    record_workspace_published_material_click,
)


def _jwt(payload: dict, secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}

    def enc(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    signing_input = f"{enc(header)}.{enc(payload)}"
    sig = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(sig).decode('ascii').rstrip('=')}"


def _jwt_rs256(payload: dict, private_key, kid: str = "kid-test") -> str:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    header = {"alg": "RS256", "typ": "JWT", "kid": kid}

    def enc(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    signing_input = f"{enc(header)}.{enc(payload)}"
    sig = private_key.sign(signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input}.{base64.urlsafe_b64encode(sig).decode('ascii').rstrip('=')}"


def _rsa_jwk(private_key, kid: str = "kid-test") -> dict:
    numbers = private_key.public_key().public_numbers()

    def enc_int(value: int) -> str:
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return {"kty": "RSA", "kid": kid, "alg": "RS256", "use": "sig", "n": enc_int(numbers.n), "e": enc_int(numbers.e)}


def _rsa_x5c_jwk(private_key, kid: str = "kid-test") -> dict:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.x509.oid import NameOID

    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "RealityLoop Test Issuer")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=5))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
        .sign(private_key, hashes.SHA256())
    )
    der = cert.public_bytes(serialization.Encoding.DER)
    return {
        "kty": "RSA",
        "kid": kid,
        "alg": "RS256",
        "use": "sig",
        "x5c": [base64.b64encode(der).decode("ascii")],
    }


def _write_event_asset(exp_dir: Path, event_id: str = "evt_001") -> Path:
    event_dir = exp_dir / "materials" / "events" / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    for name in ["clip.mp4", "preview.jpg", "keyframe_01.jpg", "keyframe_02.jpg", "keyframe_03.jpg"]:
        (event_dir / name).write_bytes(f"{name}:{event_id}".encode("utf-8"))
    event = {
        "event_id": event_id,
        "experiment_id": exp_dir.name,
        "event_type": "liquid_transfer",
        "actor_name": "operator_a",
        "start_time_sec": 96.2,
        "end_time_sec": 103.5,
        "involved_objects": ["bottle", "beaker"],
        "dominant_object": "bottle",
        "source_container": {"track_id": "trk_1", "class_name": "bottle", "display_name": "试剂瓶", "role_confidence": 0.9},
        "target_container": {"track_id": "trk_2", "class_name": "beaker", "display_name": "烧杯", "role_confidence": 0.8},
        "evidence_grade": "medium",
        "review_status": "candidate_review",
        "direction_status": "candidate",
    }
    asset = {
        "event_id": event_id,
        "clip_path": str(event_dir / "clip.mp4"),
        "preview_path": str(event_dir / "preview.jpg"),
        "keyframe_paths": [str(event_dir / f"keyframe_{idx:02d}.jpg") for idx in range(1, 4)],
        "event_json_path": str(event_dir / "event.json"),
        "overlay_mode": "event_selective",
        "asset_status": "ready",
        "quality_score": 91.0,
        "quality_grade": "excellent",
        "quality_reasons": ["good_keyframe_coverage"],
    }
    payload = {"event": {**event, "asset_pack": asset}, "asset_pack": asset}
    (event_dir / "event.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (exp_dir / "experiment.json").write_text(json.dumps({"experiment_id": exp_dir.name, "title": "固体称量实验"}, ensure_ascii=False), encoding="utf-8")
    return event_dir


def test_material_publishing_naming_rules():
    event = {
        "event_type": "liquid_transfer",
        "start_time_sec": 96.2,
        "end_time_sec": 103.5,
        "source_container": {"class_name": "bottle"},
        "target_container": {"class_name": "beaker"},
    }
    assert slugify("固体称量实验", fallback="experiment") == "experiment"
    assert stable_name("solid weighing", event) == "solid_weighing__liquid_transfer__bottle_to_beaker__t096_103"


def test_archive_path_generation(tmp_path: Path):
    planner = ArchivePlanner(tmp_path)
    plan = planner.plan(
        event={"event_id": "evt_001", "event_type": "liquid_transfer", "start_time_sec": 96.2, "end_time_sec": 103.5},
        stable_name="solid_weighing__liquid_transfer__bottle_to_beaker__t096_103",
        actor_name="operator_a",
    )
    assert "published_materials/operator_a/liquid_transfer" in plan.relative_publish_dir
    assert plan.material_publish_path.name == "material_publish.json"


def test_publish_generates_material_publish_manifest_and_index(tmp_path: Path):
    _write_event_asset(tmp_path)
    result = SemanticMaterialPublisher(tmp_path, experiment_id="exp_publish").publish()
    assert result["published_materials"]["total"] == 1
    item = result["published_materials"]["items"][0]
    assert Path(item["published_paths"]["clip"]).exists()
    assert Path(item["published_paths"]["material_publish"]).exists()
    assert item["extra"]["quality_score"] == 91.0
    assert item["extra"]["quality_grade"] == "excellent"
    assert item["extra"]["quality_reasons"] == ["good_keyframe_coverage"]
    assert (tmp_path / "upload_manifest.json").exists()
    manifest = json.loads((tmp_path / "upload_manifest.json").read_text(encoding="utf-8"))
    assert manifest["items"][0]["recommended_remote_path"].endswith(f"{item['stable_name']}/")
    conn = sqlite3.connect(tmp_path / "material_index.sqlite")
    try:
        row = conn.execute("SELECT published_path, material_publish_path FROM event_materials WHERE event_id='evt_001'").fetchone()
    finally:
        conn.close()
    assert row and row[0] and row[1]


def test_publish_is_idempotent(tmp_path: Path):
    _write_event_asset(tmp_path)
    publisher = SemanticMaterialPublisher(tmp_path, experiment_id="exp_publish")
    first = publisher.publish()
    second = publisher.publish()
    assert first["published_materials"]["total"] == second["published_materials"]["total"] == 1
    conn = sqlite3.connect(tmp_path / "material_index.sqlite")
    try:
        count = conn.execute("SELECT COUNT(*) FROM event_materials WHERE event_id='evt_001'").fetchone()[0]
    finally:
        conn.close()
    assert count == 1


def test_material_publish_api(tmp_path: Path):
    import backend.main as main

    exp_id = "_pytest_material_publish_api"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    if exp_dir.exists():
        import shutil

        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    _write_event_asset(exp_dir)
    client = TestClient(main.app)
    response = client.post(f"/api/v1/experiments/{exp_id}/materials/publish")
    assert response.status_code == 200
    assert response.json()["published_total"] == 1
    listed = client.get(f"/api/v1/experiments/{exp_id}/materials/published")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    manifest = client.get(f"/api/v1/experiments/{exp_id}/materials/upload-manifest")
    assert manifest.status_code == 200
    assert len(manifest.json()["items"]) == 1


def test_published_materials_api_limit_preserves_total_and_reports_returned():
    import shutil
    import backend.main as main

    exp_id = "_pytest_material_published_limit_api"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    main._EXPERIMENTS.pop(exp_id, None)
    try:
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "experiment.json").write_text(
            json.dumps({"experiment_id": exp_id, "experiment_name": "Published limit API regression"}),
            encoding="utf-8",
        )
        items = [
            {
                "event_id": f"evt_{idx:03d}",
                "material_id": f"mat_evt_{idx:03d}",
                "display_name": f"Material {idx}",
                "time_start": float(idx),
                "time_end": float(idx + 1),
                "published_paths": {"clip": str(exp_dir / "published_materials" / f"clip_{idx:03d}.mp4")},
            }
            for idx in range(8)
        ]
        (exp_dir / "published_materials.json").write_text(
            json.dumps(
                {
                    "schema_version": "published_materials.v1",
                    "experiment_id": exp_id,
                    "total": len(items),
                    "items": items,
                }
            ),
            encoding="utf-8",
        )

        client = TestClient(main.app)
        limited = client.get(f"/api/v1/experiments/{exp_id}/materials/published?limit=6")
        assert limited.status_code == 200
        limited_payload = limited.json()
        assert limited_payload["total"] == 8
        assert limited_payload["returned"] == 6
        assert len(limited_payload["items"]) == 6
        assert [item["event_id"] for item in limited_payload["items"]] == [f"evt_{idx:03d}" for idx in range(6)]

        oversized = client.get(f"/api/v1/experiments/{exp_id}/materials/published?limit=20")
        assert oversized.status_code == 200
        oversized_payload = oversized.json()
        assert oversized_payload["total"] == 8
        assert oversized_payload["returned"] == 8
        assert len(oversized_payload["items"]) == 8
    finally:
        main._EXPERIMENTS.pop(exp_id, None)
        if exp_dir.exists():
            shutil.rmtree(exp_dir)


def test_material_timeline_limit_and_etag_cache():
    import shutil
    import backend.main as main

    exp_id = "_pytest_material_timeline_limit_api"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    main._EXPERIMENTS.pop(exp_id, None)
    try:
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "experiment.json").write_text(
            json.dumps({"experiment_id": exp_id, "experiment_name": "Timeline limit API regression"}),
            encoding="utf-8",
        )
        material_stream = [
            {"item_id": f"mat_{idx}", "timestamp_sec": float(idx), "camera_id": "cam_a"}
            for idx in range(5)
        ]
        preprocessing = {
            "detected_changes": [
                {"event_id": "evt_1", "timestamp_sec": 5.0, "metadata": {"camera_id": "cam_a"}},
                {"event_id": "evt_2", "timestamp_sec": 6.0, "metadata": {"camera_id": "cam_b"}},
            ],
            "key_clips": [{"clip_id": "clip_1"}],
        }
        (exp_dir / "material_stream.json").write_text(json.dumps(material_stream), encoding="utf-8")
        (exp_dir / "preprocessing.json").write_text(json.dumps(preprocessing), encoding="utf-8")

        client = TestClient(main.app)
        first = client.get(f"/api/v1/experiments/{exp_id}/materials/timeline?limit=3")
        assert first.status_code == 200
        payload = first.json()
        assert payload["material_count"] == 5
        assert payload["event_count"] == 2
        assert payload["total_items"] == 7
        assert payload["returned"] == 3
        assert len(payload["items"]) == 3
        assert first.headers.get("etag")

        cached = client.get(
            f"/api/v1/experiments/{exp_id}/materials/timeline?limit=3",
            headers={"If-None-Match": first.headers["etag"]},
        )
        assert cached.status_code == 304
    finally:
        main._EXPERIMENTS.pop(exp_id, None)
        if exp_dir.exists():
            shutil.rmtree(exp_dir)


def test_publish_updates_official_step_bundle(tmp_path: Path):
    _write_event_asset(tmp_path)
    (tmp_path / "official_steps.json").write_text(
        json.dumps(
            {
                "schema_version": "official_steps.v1",
                "experiment_id": "exp_publish",
                "official_steps": [
                    {
                        "official_step_id": "official_1",
                        "linked_event_ids": ["evt_001"],
                        "evidence_bundle": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "step_review_log.json").write_text(
        json.dumps({"schema_version": "step_review_log.v1", "governance_decisions": []}),
        encoding="utf-8",
    )
    result = SemanticMaterialPublisher(tmp_path, experiment_id="exp_publish").publish()
    official = json.loads((tmp_path / "official_steps.json").read_text(encoding="utf-8"))
    log = json.loads((tmp_path / "step_review_log.json").read_text(encoding="utf-8"))
    refs = official["official_steps"][0]["evidence_bundle"]["published_material_refs"]
    assert result["published_materials"]["official_step_updates"]["updated_count"] == 1
    assert refs[0]["event_id"] == "evt_001"
    assert refs[0]["material_publish_path"]
    assert log["governance_decisions"][0]["decision"] == "publish_material_refs_update"


def test_local_uploader_and_workspace_published_index(tmp_path: Path):
    exp_dir = tmp_path / "experiments" / "exp_a"
    exp_dir.mkdir(parents=True)
    _write_event_asset(exp_dir)
    SemanticMaterialPublisher(exp_dir, experiment_id="exp_a").publish()
    upload_result = uploader_for("local").upload_manifest(exp_dir / "upload_manifest.json", destination_root=tmp_path / "uploaded")
    assert upload_result.uploaded_count >= 2
    manifest = json.loads((exp_dir / "upload_manifest.json").read_text(encoding="utf-8"))
    assert manifest["items"][0]["remote_url"]
    index = rebuild_workspace_published_materials_index(tmp_path / "experiments", tmp_path / "published_index.json")
    assert index["total"] == 1
    query = query_workspace_published_materials(tmp_path / "published_index.json", event_type="liquid_transfer")
    assert query["total"] == 1
    fts_query = query_workspace_published_materials(tmp_path / "published_index.json", text="liquid_transfer")
    assert fts_query["total"] == 1


def test_workspace_published_query_sort_cursor_and_permissions(tmp_path: Path):
    root = tmp_path / "experiments"
    exp_a = root / "exp_a"
    exp_b = root / "exp_b"
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)
    _write_event_asset(exp_a, "evt_a")
    _write_event_asset(exp_b, "evt_b")
    (exp_b / "official_steps.json").write_text(
        json.dumps({"official_steps": [{"official_step_id": "official_b", "linked_event_ids": ["evt_b"]}]}),
        encoding="utf-8",
    )
    payload = json.loads((exp_b / "materials" / "events" / "evt_b" / "event.json").read_text(encoding="utf-8"))
    payload["event"]["actor_name"] = "operator_b"
    payload["event"]["start_time_sec"] = 12.0
    payload["event"]["end_time_sec"] = 15.0
    (exp_b / "materials" / "events" / "evt_b" / "event.json").write_text(json.dumps(payload), encoding="utf-8")
    SemanticMaterialPublisher(exp_a, experiment_id="exp_a").publish()
    SemanticMaterialPublisher(exp_b, experiment_id="exp_b").publish()
    index_path = tmp_path / "published_index.sqlite"
    rebuild_workspace_published_materials_index(root, index_path)

    first = query_workspace_published_materials(index_path, limit=1, sort_by="time_start", sort_order="asc")
    assert first["total"] == 1
    assert first["next_cursor"]
    second = query_workspace_published_materials(index_path, limit=1, cursor=first["next_cursor"], sort_by="time_start", sort_order="asc")
    assert second["total"] == 1
    assert first["items"][0]["material_id"] != second["items"][0]["material_id"]

    scoped = query_workspace_published_materials(index_path, operator_role="reviewer", allowed_experiment_ids=["exp_b"])
    assert scoped["total"] == 1
    assert scoped["items"][0]["experiment_id"] == "exp_b"
    denied = query_workspace_published_materials(index_path, operator_role="reviewer")
    assert denied["total"] == 0
    actor_scoped = query_workspace_published_materials(index_path, operator_role="reviewer", actor_name="operator_b")
    assert actor_scoped["total"] == 1
    assert first["next_cursor"]
    cursor_payload = json.loads(base64.urlsafe_b64decode(first["next_cursor"] + "=" * (-len(first["next_cursor"]) % 4)).decode("utf-8"))
    assert cursor_payload["v"] == 2
    assert "sort_value" in cursor_payload
    assert cursor_payload["filters_hash"] == first["cursor"]["filters_hash"]

    ranked = query_workspace_published_materials(index_path, limit=2, sort_by="business_score", sort_order="desc")
    assert ranked["sort"]["sort_by"] == "business_score"
    assert ranked["items"][0]["experiment_id"] == "exp_b"
    assert ranked["items"][0]["business_score"] >= ranked["items"][1]["business_score"]


def test_workspace_published_relevance_and_header_auth(tmp_path: Path):
    import backend.main as main

    exp_id = "_pytest_published_auth"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    if exp_dir.exists():
        import shutil

        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    _write_event_asset(exp_dir)
    SemanticMaterialPublisher(exp_dir, experiment_id=exp_id).publish()
    main._workspace_published_materials_index_path().unlink(missing_ok=True)
    client = TestClient(main.app)

    denied = client.get("/api/v1/materials/published", headers={"X-Operator-Role": "reviewer"})
    assert denied.status_code == 200
    assert denied.json()["total"] == 0

    allowed = client.get(
        "/api/v1/materials/published?text=liquid_transfer&sort_by=relevance",
        headers={"X-Operator-Role": "reviewer", "X-Allowed-Experiments": exp_id},
    )
    assert allowed.status_code == 200
    payload = allowed.json()
    assert payload["total"] >= 1
    assert payload["sort"]["sort_by"] == "relevance"
    assert payload["items"][0]["relevance_score"] is not None


def test_workspace_published_jwt_auth_and_keyset_cursor(tmp_path: Path, monkeypatch):
    import backend.main as main

    exp_id = "_pytest_published_jwt"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    if exp_dir.exists():
        import shutil

        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    _write_event_asset(exp_dir)
    SemanticMaterialPublisher(exp_dir, experiment_id=exp_id).publish()
    main._workspace_published_materials_index_path().unlink(missing_ok=True)
    secret = "pytest-jwt-secret"
    monkeypatch.setenv("REALITYLOOP_JWT_SECRET", secret)
    token = _jwt(
        {
            "sub": "operator.jwt",
            "role": "reviewer",
            "allowed_experiments": [exp_id],
            "exp": int(time.time()) + 3600,
        },
        secret,
    )
    client = TestClient(main.app)
    response = client.get(
        "/api/v1/materials/published?limit=1&sort_by=business_score&sort_order=desc",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["permission_filter"]["operator_role"] == "reviewer"
    assert payload["permission_filter"]["allowed_experiment_ids"] == [exp_id]
    assert payload["items"][0]["business_score"] is not None

    bad = client.get("/api/v1/materials/published", headers={"Authorization": "Bearer invalid.token.value"})
    assert bad.status_code == 401


def test_workspace_published_rs256_oidc_discovery_auth(tmp_path: Path, monkeypatch):
    import backend.main as main
    from cryptography.hazmat.primitives.asymmetric import rsa

    exp_id = "_pytest_published_rs256"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    if exp_dir.exists():
        import shutil

        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    _write_event_asset(exp_dir)
    SemanticMaterialPublisher(exp_dir, experiment_id=exp_id).publish()
    main._workspace_published_materials_index_path().unlink(missing_ok=True)

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = _rsa_jwk(private_key)
    issuer = "https://issuer.example.test"
    monkeypatch.setenv("REALITYLOOP_AUTH_CACHE_DIR", str(tmp_path / "auth_cache_rs256"))
    monkeypatch.setenv("REALITYLOOP_OAUTH_ISSUER_URL", issuer)
    monkeypatch.setenv("REALITYLOOP_JWT_ISSUER", issuer)
    main._OIDC_DISCOVERY_CACHE.clear()
    main._JWKS_CACHE.clear()

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, timeout):
        if url.endswith("/.well-known/openid-configuration"):
            return FakeResponse({"issuer": issuer, "jwks_uri": f"{issuer}/jwks"})
        if url.endswith("/jwks"):
            return FakeResponse({"keys": [jwk]})
        raise AssertionError(url)

    monkeypatch.setattr(main.requests, "get", fake_get)
    token = _jwt_rs256(
        {
            "sub": "rs256-user",
            "iss": issuer,
            "scope": f"role:reviewer experiment:{exp_id}",
            "exp": int(time.time()) + 3600,
        },
        private_key,
    )
    client = TestClient(main.app)
    response = client.get("/api/v1/materials/published", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["permission_filter"]["allowed_experiment_ids"] == [exp_id]


def test_workspace_published_rs256_x5c_and_persistent_discovery_cache(tmp_path: Path, monkeypatch):
    import backend.main as main
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa

    exp_id = "_pytest_published_x5c"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    if exp_dir.exists():
        import shutil

        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    _write_event_asset(exp_dir)
    SemanticMaterialPublisher(exp_dir, experiment_id=exp_id).publish()
    main._workspace_published_materials_index_path().unlink(missing_ok=True)

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = _rsa_x5c_jwk(private_key, kid="kid-x5c")
    issuer = "https://issuer-x5c.example.test"
    cache_dir = tmp_path / "auth_cache"
    root_cert = x509.load_der_x509_certificate(base64.b64decode(jwk["x5c"][0]))
    monkeypatch.setenv("REALITYLOOP_AUTH_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("REALITYLOOP_OAUTH_ISSUER_URL", issuer)
    monkeypatch.setenv("REALITYLOOP_JWT_ISSUER", issuer)
    monkeypatch.setenv("REALITYLOOP_JWKS_X5C_TRUSTED_FINGERPRINTS", root_cert.fingerprint(hashes.SHA256()).hex())
    main._OIDC_DISCOVERY_CACHE.clear()
    main._JWKS_CACHE.clear()

    calls = {"count": 0}

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, timeout):
        calls["count"] += 1
        if url.endswith("/.well-known/openid-configuration"):
            return FakeResponse({"issuer": issuer, "jwks_uri": f"{issuer}/jwks"})
        if url.endswith("/jwks"):
            return FakeResponse({"keys": [jwk]})
        raise AssertionError(url)

    monkeypatch.setattr(main.requests, "get", fake_get)
    token = _jwt_rs256(
        {
            "sub": "x5c-user",
            "iss": issuer,
            "scope": f"role:reviewer experiment:{exp_id}",
            "exp": int(time.time()) + 3600,
        },
        private_key,
        kid="kid-x5c",
    )
    client = TestClient(main.app)
    first = client.get("/api/v1/materials/published", headers={"Authorization": f"Bearer {token}"})
    assert first.status_code == 200
    assert first.json()["total"] == 1
    assert calls["count"] == 2
    assert list(cache_dir.glob("oidc_*.json"))
    assert list(cache_dir.glob("jwks_*.json"))

    main._OIDC_DISCOVERY_CACHE.clear()
    main._JWKS_CACHE.clear()

    def fail_get(url, timeout):
        raise AssertionError("network should not be used when disk cache is valid")

    monkeypatch.setattr(main.requests, "get", fail_get)
    second = client.get("/api/v1/materials/published", headers={"Authorization": f"Bearer {token}"})
    assert second.status_code == 200
    assert second.json()["total"] == 1


def test_business_score_uses_configurable_usage_profile(tmp_path: Path):
    root = tmp_path / "experiments"
    exp_a = root / "exp_a"
    exp_b = root / "exp_b"
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)
    _write_event_asset(exp_a, "evt_a")
    _write_event_asset(exp_b, "evt_b")
    (exp_a / "experiment.json").write_text(json.dumps({"experiment_id": "exp_a", "experiment_type": "priority_type"}), encoding="utf-8")
    (exp_b / "experiment.json").write_text(json.dumps({"experiment_id": "exp_b", "experiment_type": "normal_type"}), encoding="utf-8")
    (exp_a / "material_usage.json").write_text(json.dumps({"mat_evt_a": {"click_count": 12}}), encoding="utf-8")
    (exp_b / "official_steps.json").write_text(json.dumps({"official_steps": [{"linked_event_ids": ["evt_b"]}]}), encoding="utf-8")
    (exp_a / "step_review_log.json").write_text(json.dumps({"review_decisions": [{"linked_event_ids": ["evt_a"]}]}), encoding="utf-8")
    SemanticMaterialPublisher(exp_a, experiment_id="exp_a").publish()
    SemanticMaterialPublisher(exp_b, experiment_id="exp_b").publish()
    profile_path = tmp_path / "scoring.json"
    profile_path.write_text(
        json.dumps(
            {
                "weights": {
                    "click_count": 0.20,
                    "experiment_type_priority": 1.0,
                    "official_linked": 0.01,
                    "official_usage_count": 0.01,
                },
                "experiment_type_priorities": {"priority_type": 2.0},
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "published_index.sqlite"
    rebuild_workspace_published_materials_index(root, index_path)
    clicked = record_workspace_published_material_click(index_path, "exp_a:mat_evt_a", experiments_root=root)
    assert clicked["updated"] is True
    assert clicked["usage_event_id"].startswith("usage_")
    usage = json.loads((exp_a / "material_usage.json").read_text(encoding="utf-8"))
    assert usage["items"][0]["click_count"] == 13
    assert usage["items"][0]["last_clicked_at"]
    conn = sqlite3.connect(index_path)
    try:
        usage_count = conn.execute("SELECT click_count FROM material_usage_counts WHERE material_id='exp_a:mat_evt_a'").fetchone()[0]
        usage_events = conn.execute("SELECT COUNT(*) FROM material_usage_events WHERE material_id='exp_a:mat_evt_a'").fetchone()[0]
    finally:
        conn.close()
    assert usage_count == 13
    assert usage_events == 1
    ranked = query_workspace_published_materials(index_path, limit=2, sort_by="business_score", sort_order="desc", scoring_profile_path=profile_path)
    assert ranked["items"][0]["experiment_id"] == "exp_a"
    assert ranked["items"][0]["click_count"] >= 13
    assert ranked["scoring_profile"]["profile_hash"]


def test_display_name_uses_existing_qwen_summary(tmp_path: Path):
    event_dir = _write_event_asset(tmp_path)
    payload = json.loads((event_dir / "event.json").read_text(encoding="utf-8"))
    payload["event"].pop("display_name", None)
    payload["event"]["qwen_summary"] = "operator pours liquid into beaker"
    (event_dir / "event.json").write_text(json.dumps(payload), encoding="utf-8")
    item = SemanticMaterialPublisher(tmp_path, experiment_id="exp_publish").publish()["published_materials"]["items"][0]
    assert "operator pours liquid" in item["display_name"]
    assert item["extra"]["display_name_source"] == "qwen_vlm_summary"


def test_display_name_can_call_live_qwen_model(tmp_path: Path, monkeypatch):
    event_dir = _write_event_asset(tmp_path)
    payload = json.loads((event_dir / "event.json").read_text(encoding="utf-8"))
    payload["event"].pop("display_name", None)
    (event_dir / "event.json").write_text(json.dumps(payload), encoding="utf-8")

    def fake_call(model, image_path, prompt, *, timeout_sec, retries):
        return {
            "structured_result": {"display_name": "固体称量实验-液体转移-倒入烧杯"},
            "source": "test",
            "response_time_ms": 1,
        }

    import labsopguard.qwen_writeback as qwen_writeback

    monkeypatch.setattr(qwen_writeback, "_call_frame_model", fake_call)
    enhancer = QwenVlmDisplayNameEnhancer(enable_live=True, model="fake-qwen")
    item = SemanticMaterialPublisher(tmp_path, experiment_id="exp_publish", display_name_enhancer=enhancer).publish()["published_materials"]["items"][0]
    assert item["display_name"] == "固体称量实验-液体转移-倒入烧杯"
    assert item["extra"]["display_name_source"] == "qwen_vlm_live"
