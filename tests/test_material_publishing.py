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
    check_workspace_published_materials_lifecycle,
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


def test_published_materials_gate_requires_dual_view_keyframe_and_clip(monkeypatch):
    import backend.main as main

    monkeypatch.setenv("KEY_ACTION_REQUIRE_RELIABLE_DUAL_VIEW_ALIGNMENT", "0")
    payload = {
        "schema_version": "published_materials.v1",
        "items": [
            {"candidate_id": "complete-third-clip", "micro_segment_id": "micro-1", "canonical_action_type": "hand-paper", "view": "third_person", "asset_kind": "关键片段"},
            {"candidate_id": "complete-third-frame", "micro_segment_id": "micro-1", "canonical_action_type": "hand-paper", "view": "third_person", "asset_kind": "关键帧"},
            {"candidate_id": "complete-first-clip", "micro_segment_id": "micro-1", "canonical_action_type": "hand-paper", "view": "first_person", "asset_kind": "关键片段"},
            {"candidate_id": "complete-first-frame", "micro_segment_id": "micro-1", "canonical_action_type": "hand-paper", "view": "first_person", "asset_kind": "关键帧"},
            {"candidate_id": "first-only-clip", "micro_segment_id": "micro-2", "canonical_action_type": "hand-bottle", "view": "first_person", "asset_kind": "关键片段"},
            {"candidate_id": "first-only-frame", "micro_segment_id": "micro-2", "canonical_action_type": "hand-bottle", "view": "first_person", "asset_kind": "关键帧"},
        ],
    }

    result = main._apply_published_material_alignment_gate("exp-test", payload)

    assert result["total"] == 4
    assert {item["candidate_id"] for item in result["items"]} == {
        "complete-third-clip",
        "complete-third-frame",
        "complete-first-clip",
        "complete-first-frame",
    }
    assert result["dual_view_quality_gate"]["hidden_item_count"] == 2
    assert result["grouped_items"][0]["view"] == "dual_view"
    assert result["grouped_items"][0]["clip_count"] == 2
    assert result["grouped_items"][0]["keyframe_count"] == 2


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


def test_material_reference_api_adds_canonical_taxonomy():
    import shutil
    import backend.main as main

    exp_id = "_pytest_material_taxonomy_api"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    ref_root = exp_dir / "material_references"
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    main._EXPERIMENTS.pop(exp_id, None)
    try:
        frame_dir = ref_root / "\u5173\u952e\u5e27"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame = frame_dir / "hand_bottle.jpg"
        frame.write_bytes(b"jpg")
        (exp_dir / "experiment.json").write_text(
            json.dumps({"experiment_id": exp_id, "title": "taxonomy api"}),
            encoding="utf-8",
        )
        row = {
            "schema_version": "material_reference.item.v1",
            "asset_kind": "\u5173\u952e\u5e27",
            "material_type": "\u5173\u952e\u5e27",
            "action_name": "hand reagent bottle operation",
            "stored_file": str(frame),
            "stored_filename": frame.name,
            "primary_object": "reagent_bottle",
            "review_status": "accepted",
            "approved_at": "2026-05-08T10:00:00+08:00",
            "formal_material_reference": True,
        }
        (ref_root / "\u7d20\u6750\u7d22\u5f15.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        client = TestClient(main.app)
        response = client.get(f"/api/v1/experiments/{exp_id}/materials/published")
        assert response.status_code == 200
        item = response.json()["items"][0]
        assert item["canonical_action_type"] == "hand-bottle"
        assert item["canonical_object"] == "bottle"
        assert item["sop_phase"] == "reagent-bottle-handling"
        assert item["best_score"] > 0
        assert "YOLO" in item["best_reason"]
        diagnostics = client.get(f"/api/v1/experiments/{exp_id}/materials/diagnostics")
        assert diagnostics.status_code == 200
        calibration = diagnostics.json()["taxonomy_calibration"]
        assert calibration["per_action"]["hand-bottle"]["formal_material_total"] == 1
        assert (exp_dir / "material_taxonomy_calibration.json").exists()
    finally:
        main._EXPERIMENTS.pop(exp_id, None)
        if exp_dir.exists():
            shutil.rmtree(exp_dir)


def test_material_candidate_disposition_requires_reason_and_hides_from_pending():
    import shutil
    import backend.main as main

    exp_id = "_pytest_material_candidate_disposition"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    queue = exp_dir / "_material_review_queue"
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    main._EXPERIMENTS.pop(exp_id, None)
    try:
        frame_dir = queue / "\u5173\u952e\u5e27"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame = frame_dir / "bad_candidate.jpg"
        frame.write_bytes(b"jpg")
        (exp_dir / "experiment.json").write_text(json.dumps({"experiment_id": exp_id, "title": "candidate disposition"}), encoding="utf-8")
        rows = [
            {
                "schema_version": "material_reference.item.v1",
                "candidate_id": "candidate_bad_frame",
                "candidate_group_id": "group_bad",
                "asset_kind": "\u5173\u952e\u5e27",
                "material_type": "\u5173\u952e\u5e27",
                "stored_file": str(frame),
                "stored_filename": frame.name,
                "primary_object": "balance",
                "candidate_status": "pending",
                "review_status": "pending",
            }
        ]
        (queue / "\u7d20\u6750\u5019\u9009\u7d22\u5f15.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

        client = TestClient(main.app)
        missing_reason = client.post(
            f"/api/v1/experiments/{exp_id}/materials/candidates/group_bad/decision",
            json={"decision": "false_positive"},
        )
        assert missing_reason.status_code == 400

        decided = client.post(
            f"/api/v1/experiments/{exp_id}/materials/candidates/group_bad/decision",
            json={"decision": "false_positive", "reason_code": "wrong_object", "notes": "not a balance action"},
        )
        assert decided.status_code == 200
        payload = decided.json()["candidates"]
        assert payload["pending_total"] == 0
        assert payload["rejected_total"] == 1
        assert payload["items"][0]["status"] == "rejected"
        updated = json.loads((queue / "\u7d20\u6750\u5019\u9009\u7d22\u5f15.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert updated["candidate_status"] == "rejected"
        assert updated["rejection_reason_code"] == "wrong_object"
        assert (queue / "review_log.jsonl").exists()
    finally:
        main._EXPERIMENTS.pop(exp_id, None)
        if exp_dir.exists():
            shutil.rmtree(exp_dir)


def test_material_candidate_confirm_and_rename_keep_candidate_reviewable():
    import shutil
    import backend.main as main

    exp_id = "_pytest_material_candidate_confirm_rename"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    queue = exp_dir / "_material_review_queue"
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    main._EXPERIMENTS.pop(exp_id, None)
    try:
        frame_dir = queue / "\u5173\u952e\u5e27"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame = frame_dir / "candidate.jpg"
        frame.write_bytes(b"jpg")
        (exp_dir / "experiment.json").write_text(json.dumps({"experiment_id": exp_id, "title": "candidate confirm"}), encoding="utf-8")
        rows = [
            {
                "schema_version": "material_reference.item.v1",
                "candidate_id": "candidate_frame",
                "candidate_group_id": "group_confirm",
                "asset_kind": "\u5173\u952e\u5e27",
                "material_type": "\u5173\u952e\u5e27",
                "stored_file": str(frame),
                "stored_filename": frame.name,
                "candidate_status": "pending",
                "review_status": "pending",
                "display_title": "原始候选",
            }
        ]
        (queue / "\u7d20\u6750\u5019\u9009\u7d22\u5f15.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

        client = TestClient(main.app)
        confirmed = client.post(
            f"/api/v1/experiments/{exp_id}/materials/candidates/group_confirm/decision",
            json={"decision": "confirmed", "candidate_ids": ["candidate_frame"], "notes": "人工确认"},
        )
        assert confirmed.status_code == 200
        payload = confirmed.json()["candidates"]
        assert payload["items"][0]["status"] == "confirmed"
        assert payload["items"][0]["review_status"] == "confirmed"

        renamed = client.patch(
            f"/api/v1/experiments/{exp_id}/materials/candidates/group_confirm/rename",
            json={"display_title": "手部与称量纸操作", "candidate_ids": ["candidate_frame"]},
        )
        assert renamed.status_code == 200
        renamed_payload = renamed.json()["candidates"]
        assert renamed_payload["items"][0]["display_title"] == "手部与称量纸操作"
        updated = json.loads((queue / "\u7d20\u6750\u5019\u9009\u7d22\u5f15.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert updated["human_display_title"] == "手部与称量纸操作"
        assert updated["rename_scope"] == "display_only"
    finally:
        main._EXPERIMENTS.pop(exp_id, None)
        if exp_dir.exists():
            shutil.rmtree(exp_dir)


def test_experiment_published_materials_prefers_global_formal_delivery_folder():
    import shutil
    import backend.main as main

    exp_id = "_pytest_formal_material_delivery_api"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    delivery_root = main.PROJECT_ROOT / "outputs" / "material_references" / "正式素材交付测试_20260508"
    for path in (exp_dir, delivery_root):
        if path.exists():
            shutil.rmtree(path)
    main._EXPERIMENTS.pop(exp_id, None)
    try:
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "experiment.json").write_text(
            json.dumps(
                {
                    "experiment_id": exp_id,
                    "title": "正式素材交付测试",
                    "created_at": "2026-05-08T09:30:00+08:00",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        keyframe_dir = delivery_root / "关键帧"
        clip_dir = delivery_root / "关键片段"
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        clip_dir.mkdir(parents=True, exist_ok=True)
        frame = keyframe_dir / "手与天平操作_20260508.jpg"
        clip = clip_dir / "手与天平操作_20260508.mp4"
        frame.write_bytes(b"jpg")
        clip.write_bytes(b"mp4")
        rows = [
            {
                "schema_version": "material_reference.item.v1",
                "asset_kind": "关键帧",
                "material_type": "关键帧",
                "action_name": "手与天平操作",
                "stored_file": str(frame),
                "stored_filename": frame.name,
                "primary_object": "balance",
                "review_status": "accepted",
            },
            {
                "schema_version": "material_reference.item.v1",
                "asset_kind": "关键片段",
                "material_type": "关键片段",
                "action_name": "手与天平操作",
                "stored_file": str(clip),
                "stored_filename": clip.name,
                "primary_object": "balance",
                "review_status": "accepted",
            },
        ]
        (delivery_root / "素材索引.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        (delivery_root / "manifest.json").write_text(
            json.dumps({"formal_material_references": str(delivery_root)}, ensure_ascii=False),
            encoding="utf-8",
        )

        client = TestClient(main.app)
        response = client.get(f"/api/v1/experiments/{exp_id}/materials/published")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 2
        assert payload["source"] == str(delivery_root)
        frame_item = next(item for item in payload["items"] if item["asset_kind"] == "关键帧")
        clip_item = next(item for item in payload["items"] if item["asset_kind"] == "关键片段")
        assert frame_item["frame_path"] == str(frame)
        assert clip_item["clip_file_path"] == str(clip)
        assert frame_item["preview_url"].startswith(f"/api/v1/experiments/{exp_id}/material-references/files/")
        assert clip_item["clip_url"].startswith(f"/api/v1/experiments/{exp_id}/material-references/files/")
        assert client.get(frame_item["preview_url"]).status_code == 200
        assert client.get(clip_item["clip_url"]).status_code == 200

        reindex = client.post("/api/v1/materials/published/reindex", headers={"X-Operator-Role": "admin"})
        assert reindex.status_code == 200
        workspace = client.get("/api/v1/materials/published?limit=500", headers={"X-Operator-Role": "admin"})
        assert workspace.status_code == 200
        workspace_items = [item for item in workspace.json()["items"] if item.get("experiment_id") == exp_id]
        assert len(workspace_items) == 2
        workspace_frame = next(item for item in workspace_items if item["preview_path"])
        workspace_clip = next(item for item in workspace_items if item["clip_path"])
        assert workspace_frame["preview_url"].startswith(f"/api/v1/experiments/{exp_id}/material-references/files/")
        assert workspace_clip["clip_url"].startswith(f"/api/v1/experiments/{exp_id}/material-references/files/")
        assert client.get(workspace_frame["preview_url"]).status_code == 200
        assert client.get(workspace_clip["clip_url"]).status_code == 200
    finally:
        main._EXPERIMENTS.pop(exp_id, None)
        for path in (exp_dir, delivery_root):
            if path.exists():
                shutil.rmtree(path)


def test_material_reference_items_backfill_real_dual_view_micro_clips(tmp_path: Path, monkeypatch):
    import backend.main as main

    keyframe_kind = "\u5173\u952e\u5e27"
    clip_kind = "\u5173\u952e\u7247\u6bb5"
    exp_id = "_pytest_material_clip_backfill_contract"
    main.PROJECT_ROOT = tmp_path
    exp_dir = tmp_path / "outputs" / "experiments" / exp_id
    ref_root = exp_dir / "material_references"
    keyframe_root = ref_root / keyframe_kind
    clip_root = exp_dir / "key_action_index" / "clips" / "micro"
    keyframe_root.mkdir(parents=True)
    clip_root.mkdir(parents=True)
    monkeypatch.setenv("KEY_ACTION_BACKFILL_SOURCE_MICRO_CLIPS", "1")

    rows = []
    for view in ("third_person", "first_person"):
        frame = keyframe_root / f"seg_000001_micro_001_{view}.jpg"
        source_clip = clip_root / f"seg_000001_micro_001_{view}.mp4"
        frame.write_bytes(b"frame")
        source_clip.write_bytes(b"clip")
        rows.append(
            {
                "schema_version": "material_reference.item.v1",
                "asset_kind": keyframe_kind,
                "material_type": keyframe_kind,
                "stored_file": str(frame),
                "stored_filename": frame.name,
                "candidate_status": "approved",
                "review_status": "accepted",
                "approved_at": "2026-05-20T00:00:00+08:00",
                "formal_material_reference": True,
                "material_group_id": "seg_000001_micro_001_merged",
                "micro_segment_id": "seg_000001_micro_001_merged",
                "parent_segment_id": "seg_000001",
                "segment_id": "seg_000001",
                "canonical_action_type": "hand-paper",
                "display_title": "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c",
                "view": view,
                "time_start": 10.0,
                "time_end": 12.0,
                "quality_score": 0.9,
            }
        )
    (ref_root / "\u7d20\u6750\u7d22\u5f15.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    payload = main._material_reference_items(exp_dir, exp_id)
    items = payload["items"]
    clips = [item for item in items if item.get("asset_kind") == clip_kind]
    frames = [item for item in items if item.get("asset_kind") == keyframe_kind]

    assert len(items) == 4
    assert {item["view"] for item in frames} == {"third_person", "first_person"}
    assert {item["view"] for item in clips} == {"third_person", "first_person"}
    assert all(Path(item["clip_file_path"]).parent == clip_root for item in clips)
    assert all(item["source_micro_clip_backfilled"] is True for item in clips)
    assert all(item["clip_url"].startswith(f"/api/v1/experiments/{exp_id}/files/") for item in clips)

    groups = payload["grouped_items"]
    assert len(groups) == 1
    assert groups[0]["clip_count"] == 2
    assert groups[0]["keyframe_count"] == 2
    assert groups[0]["view"] == "dual_view"


def test_workspace_published_materials_indexes_formal_material_references_without_publish_json(tmp_path: Path):
    experiments_root = tmp_path / "outputs" / "experiments"
    exp_id = "exp_formal_workspace_index"
    exp_dir = experiments_root / exp_id
    delivery_root = tmp_path / "outputs" / "material_references" / "正式素材交付测试_20260508"
    keyframe_dir = delivery_root / "关键帧"
    clip_dir = delivery_root / "关键片段"
    report_dir = delivery_root / "专业报告"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": exp_id,
                "title": "正式素材交付测试",
                "created_at": "2026-05-08T09:30:00+08:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rows = []
    for index, (primary_object, action_object) in enumerate((("container", "容器"), ("beaker", "烧杯")), start=1):
        frame = keyframe_dir / f"手与{action_object}操作_20260508_{index:02d}.jpg"
        clip = clip_dir / f"手与{action_object}操作_20260508.mp4"
        frame.write_bytes(b"jpg")
        clip.write_bytes(b"mp4")
        for asset_kind, path in (("关键帧", frame), ("关键片段", clip)):
            rows.append(
                {
                    "schema_version": "material_reference.item.v1",
                    "asset_kind": asset_kind,
                    "material_type": asset_kind,
                    "action_name": f"手与{action_object}操作",
                    "candidate_id": f"candidate_{primary_object}_{asset_kind}",
                    "micro_segment_id": f"micro_{index}",
                    "stored_file": str(path),
                    "stored_filename": path.name,
                    "primary_object": primary_object,
                    "review_status": "accepted",
                    "start_sec": float(index),
                    "end_sec": float(index + 1),
                    "yolo_recheck": {
                        "status": "passed",
                        "primary_object": primary_object,
                        "valid_evidence_count": 3,
                    },
                    "vlm_semantics": {
                        "status": "aligned",
                        "model": "qwen3.6-plus",
                        "description": "戴手套操作烧杯并进行 pouring liquid" if primary_object == "beaker" else "戴手套操作容器",
                        "physical_action": "pouring_liquid" if primary_object == "beaker" else "handling_container",
                        "confirmed_objects": [primary_object, "gloved_hand"],
                    },
                }
            )
    report = report_dir / "professional_report_qwen36max.pdf"
    report.write_bytes(b"%PDF")
    rows.append(
        {
            "schema_version": "material_reference.item.v1",
            "asset_kind": "专业报告",
            "material_type": "专业报告",
            "role": "professional_report_pdf",
            "stored_file": str(report),
            "file_name": report.name,
            "review_status": "accepted",
        }
    )
    (delivery_root / "素材索引.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    index_path = tmp_path / "published_materials.sqlite"
    rebuild = rebuild_workspace_published_materials_index(experiments_root, index_path)
    assert rebuild["total"] == 4
    assert rebuild["experiments"][0]["source"] == "formal_material_references"
    assert rebuild["experiments"][0]["published_count"] == 4

    queried = query_workspace_published_materials(index_path, limit=10)
    assert queried["total"] == 4
    assert {item["event_type"] for item in queried["items"]} == {"手与容器操作", "手与烧杯操作"}
    assert all(item["event_type"] != "专业报告" for item in queried["items"])
    assert sum(1 for item in queried["items"] if item["preview_path"]) == 2
    assert sum(1 for item in queried["items"] if item["clip_path"]) == 2
    assert query_workspace_published_materials(index_path, text="烧杯", limit=10)["total"] == 2
    assert query_workspace_published_materials(index_path, text="容器操作", limit=10)["total"] == 2
    assert query_workspace_published_materials(index_path, text="戴手套操作", limit=10)["total"] == 4
    assert query_workspace_published_materials(index_path, text="pouring liquid", limit=10)["total"] == 2


def test_workspace_published_lifecycle_health_rebuilds_stale_formal_index(tmp_path: Path):
    experiments_root = tmp_path / "experiments"
    exp_id = "exp_formal_lifecycle"
    exp_dir = experiments_root / exp_id
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(
        json.dumps({"experiment_id": exp_id, "title": "Lifecycle Formal", "created_at": "2026-05-08T00:00:00Z"}),
        encoding="utf-8",
    )
    ref_root = tmp_path / "material_references" / "Lifecycle_Formal_20260508"
    (ref_root / "关键帧").mkdir(parents=True)
    (ref_root / "关键片段").mkdir(parents=True)

    def write_material(filename: str, asset_kind: str, index: int) -> dict:
        folder = "关键帧" if asset_kind == "关键帧" else "关键片段"
        suffix = ".jpg" if asset_kind == "关键帧" else ".mp4"
        stored_filename = f"{filename}{suffix}"
        (ref_root / folder / stored_filename).write_bytes(b"material")
        return {
            "candidate_id": f"candidate_{index}",
            "asset_kind": asset_kind,
            "stored_file": f"{folder}/{stored_filename}",
            "action_name": "手与烧杯操作",
            "primary_object": "beaker",
            "vlm_semantics": {"description": "戴手套操作烧杯并进行 pouring liquid"},
            "yolo_recheck": {"status": "passed", "primary_object": "beaker", "valid_evidence_count": 2},
        }

    rows = [
        write_material("frame_1", "关键帧", 1),
        write_material("frame_2", "关键帧", 2),
        write_material("clip_1", "关键片段", 3),
        write_material("clip_2", "关键片段", 4),
        {"asset_kind": "专业报告", "stored_file": "专业报告/report.md"},
    ]
    index_jsonl = ref_root / "素材索引.jsonl"
    index_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    index_path = tmp_path / "published_materials.sqlite"
    rebuild_workspace_published_materials_index(experiments_root, index_path)

    healthy = check_workspace_published_materials_lifecycle(experiments_root, index_path)
    assert healthy["status"] == "ok"
    assert healthy["sqlite_count"] == 4
    assert healthy["expected_indexable_count"] == 4
    assert healthy["formal_jsonl_material_count"] == 4
    assert healthy["formal_report_count"] == 1

    rows.append(write_material("frame_3", "关键帧", 5))
    index_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    stale = check_workspace_published_materials_lifecycle(experiments_root, index_path)
    assert stale["status"] == "needs_rebuild"
    assert stale["sqlite_count"] == 4
    assert stale["expected_indexable_count"] == 5
    assert {warning["code"] for warning in stale["warnings"]} & {"count_mismatch", "stale_index"}

    rebuilt = check_workspace_published_materials_lifecycle(experiments_root, index_path, auto_rebuild=True)
    assert rebuilt["status"] == "rebuilt"
    assert rebuilt["sqlite_count"] == 5
    assert rebuilt["expected_indexable_count"] == 5
    assert rebuilt["warnings_before_rebuild"]


def test_material_candidate_approval_api_rebuilds_workspace_published_index():
    import shutil
    import backend.main as main
    from labsopguard.material_maintenance import query_workspace_published_materials

    exp_id = "_pytest_candidate_approval_reindexes_workspace"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    delivery_root = main.PROJECT_ROOT / "outputs" / "material_references" / "审批索引刷新测试_20260508"
    for path in (exp_dir, delivery_root):
        if path.exists():
            shutil.rmtree(path)
    main._EXPERIMENTS.pop(exp_id, None)
    try:
        key_action_dir = exp_dir / "key_action_index"
        candidate_root = exp_dir / "_material_review_queue"
        candidate_frame_dir = candidate_root / "关键帧"
        candidate_clip_dir = candidate_root / "关键片段"
        candidate_frame_dir.mkdir(parents=True, exist_ok=True)
        candidate_clip_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "experiment.json").write_text(
            json.dumps(
                {
                    "experiment_id": exp_id,
                    "title": "审批索引刷新测试",
                    "created_at": "2026-05-08T09:30:00+08:00",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        key_action_dir.mkdir(parents=True, exist_ok=True)
        frame = candidate_frame_dir / "手与烧杯操作_20260508.jpg"
        clip = candidate_clip_dir / "手与烧杯操作_20260508.mp4"
        frame.write_bytes(b"jpg")
        clip.write_bytes(b"mp4")
        rows = [
            {
                "schema_version": "material_reference.item.v1",
                "asset_kind": "关键帧",
                "material_type": "关键帧",
                "candidate_id": "candidate_frame",
                "candidate_group_id": "candidate_group_beaker",
                "candidate_status": "pending",
                "review_status": "pending",
                "recommended": True,
                "stored_file": str(frame),
                "stored_filename": frame.name,
                "action_name": "手与烧杯操作",
                "primary_object": "beaker",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "exists": True,
                "yolo_recheck": {"status": "passed", "primary_object": "beaker", "valid_evidence_count": 3},
                "vlm_semantics": {
                    "status": "aligned",
                    "model": "qwen3.6-plus",
                    "description": "戴手套操作烧杯并进行 pouring liquid",
                    "physical_action": "pouring_liquid",
                },
            },
            {
                "schema_version": "material_reference.item.v1",
                "asset_kind": "关键片段",
                "material_type": "关键片段",
                "candidate_id": "candidate_clip",
                "candidate_group_id": "candidate_group_beaker",
                "candidate_status": "pending",
                "review_status": "pending",
                "recommended": True,
                "stored_file": str(clip),
                "stored_filename": clip.name,
                "action_name": "手与烧杯操作",
                "primary_object": "beaker",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "exists": True,
                "yolo_recheck": {"status": "passed", "primary_object": "beaker", "valid_evidence_count": 3},
                "vlm_semantics": {
                    "status": "aligned",
                    "model": "qwen3.6-plus",
                    "description": "戴手套操作烧杯并进行 pouring liquid",
                    "physical_action": "pouring_liquid",
                },
            },
        ]
        (candidate_root / "素材候选索引.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

        client = TestClient(main.app)
        response = client.post(
            f"/api/v1/experiments/{exp_id}/materials/candidates/candidate_group_beaker/approve",
            json={"reviewer": "pytest"},
            headers={"X-Operator-Role": "admin"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["approval"]["approved_count"] == 2
        assert payload["workspace_published_materials_reindex"]["total"] >= 2

        queried = query_workspace_published_materials(main._workspace_published_materials_index_path(), text="pouring liquid", limit=500)
        current = [item for item in queried["items"] if item.get("experiment_id") == exp_id]
        assert len(current) == 2
    finally:
        main._EXPERIMENTS.pop(exp_id, None)
        for path in (exp_dir, delivery_root):
            if path.exists():
                shutil.rmtree(path)
        rebuild_workspace_published_materials_index(
            main.PROJECT_ROOT / "outputs" / "experiments",
            main._workspace_published_materials_index_path(),
        )


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


def test_workspace_published_query_uses_canonical_text_intent(tmp_path: Path):
    root = tmp_path / "experiments"
    exp_dir = root / "exp_taxonomy_search"
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(json.dumps({"experiment_id": "exp_taxonomy_search"}), encoding="utf-8")
    items = [
        {
            "material_id": "mat_balance",
            "event_id": "evt_balance",
            "experiment_id": "exp_taxonomy_search",
            "event_type": "generic hand action",
            "display_name": "generic action A",
            "canonical_action_type": "hand-balance",
            "canonical_object": "balance",
            "sop_phase": "balance-weighing",
            "time_start": 10.0,
            "published_paths": {"preview": str(exp_dir / "balance.jpg")},
        },
        {
            "material_id": "mat_container",
            "event_id": "evt_container",
            "experiment_id": "exp_taxonomy_search",
            "event_type": "generic hand action",
            "display_name": "generic action B",
            "canonical_action_type": "hand-container",
            "canonical_object": "container",
            "sop_phase": "container-handling",
            "time_start": 20.0,
            "published_paths": {"preview": str(exp_dir / "container.jpg")},
        },
    ]
    (exp_dir / "published_materials.json").write_text(json.dumps({"items": items, "total": len(items)}, ensure_ascii=False), encoding="utf-8")
    index_path = tmp_path / "published.sqlite"
    rebuild_workspace_published_materials_index(root, index_path)

    balance = query_workspace_published_materials(index_path, text="天平称量", sort_by="relevance", limit=10)
    assert balance["items"][0]["canonical_action_type"] == "hand-balance"
    container = query_workspace_published_materials(index_path, text="容器承接", sort_by="relevance", limit=10)
    assert container["items"][0]["canonical_action_type"] == "hand-container"
    filtered = query_workspace_published_materials(index_path, canonical_action_type="hand-container", limit=10)
    assert filtered["total"] == 1
    assert filtered["items"][0]["canonical_object"] == "container"


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
