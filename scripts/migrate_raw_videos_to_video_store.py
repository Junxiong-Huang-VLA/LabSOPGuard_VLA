from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_hash_index(index_path: Path, video_root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not index_path.exists():
        return rows
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        digest = str(row.get("sha256") or "")
        raw_path = row.get("absolute_path") or row.get("path")
        if not digest or not raw_path:
            continue
        path = Path(str(raw_path))
        try:
            path = path.resolve()
            path.relative_to(video_root)
        except Exception:
            continue
        if path.exists() and path.is_file():
            rows.setdefault(digest, {**row, "path": str(path), "absolute_path": str(path)})
    return rows


def final_store_path(video_root: Path, digest: str, suffix: str) -> Path:
    clean_suffix = suffix.lower() if suffix else ".mp4"
    return video_root / "by_hash" / digest[:2] / digest[2:4] / digest / f"{digest}{clean_suffix}"


def iter_video_store_files(video_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in video_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if "_staging" in path.parts:
            continue
        files.append(path.resolve())
    return files


def iter_project_raw_upload_videos(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    experiments = project_root / "outputs" / "experiments"
    if experiments.exists():
        for path in experiments.glob("*/raw/*"):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                candidates.append(path.resolve())
    uploads = project_root / "uploads"
    if uploads.exists():
        for path in uploads.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                candidates.append(path.resolve())
    return sorted(set(candidates), key=lambda item: str(item).lower())


def write_video_ref(old_path: Path, store_path: Path, digest: str, size_bytes: int, *, source: str) -> Path:
    ref_path = old_path.with_name(f"{old_path.name}.video_ref.json")
    payload = {
        "schema_version": "lab_video_store.video_ref.v1",
        "storage_mode": "external_video_store",
        "source": source,
        "old_project_path": str(old_path),
        "video_path": str(store_path),
        "absolute_path": str(store_path),
        "sha256": digest,
        "size_bytes": size_bytes,
        "created_at": now_iso(),
    }
    ref_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ref_path


def replace_paths_in_metadata(project_root: Path, replacements: dict[str, str]) -> dict[str, int]:
    if not replacements:
        return {"files_scanned": 0, "files_updated": 0, "replacements": 0}
    text_roots = [project_root / "outputs" / "experiments"]
    escaped = {
        json.dumps(old, ensure_ascii=False)[1:-1]: json.dumps(new, ensure_ascii=False)[1:-1]
        for old, new in replacements.items()
    }
    forward = {old.replace("\\", "/"): new.replace("\\", "/") for old, new in replacements.items()}
    files_scanned = 0
    files_updated = 0
    replacement_count = 0
    for root in text_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            files_scanned += 1
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
            updated = text
            for old, new in replacements.items():
                if old in updated:
                    replacement_count += updated.count(old)
                    updated = updated.replace(old, new)
            for old, new in escaped.items():
                if old in updated:
                    replacement_count += updated.count(old)
                    updated = updated.replace(old, new)
            for old, new in forward.items():
                if old in updated:
                    replacement_count += updated.count(old)
                    updated = updated.replace(old, new)
            if updated != text:
                path.write_text(updated, encoding="utf-8")
                files_updated += 1
    return {
        "files_scanned": files_scanned,
        "files_updated": files_updated,
        "replacements": replacement_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Move historical project raw/upload videos into the external Lab video store.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--video-store-root", default=os.environ.get("LAB_VIDEO_STORE_ROOT", r"D:\LabVideoStore\raw_uploads"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    video_root = Path(args.video_store_root).resolve()
    video_root.mkdir(parents=True, exist_ok=True)
    index_path = video_root / "indexes" / "video_store_hash_index.jsonl"
    run_dir = video_root / "_migration_logs" / f"project_raw_video_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "migration_manifest.jsonl"
    summary_path = run_dir / "migration_summary.json"

    started = time.perf_counter()
    index = load_hash_index(index_path, video_root)
    indexed_existing = 0
    for idx, path in enumerate(iter_video_store_files(video_root), start=1):
        digest = sha256_file(path)
        if digest not in index:
            row = {
                "schema_version": "lab_video_store.hash_index.v1",
                "sha256": digest,
                "path": str(path),
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "original_filename": path.name,
                "indexed_at": now_iso(),
                "index_source": "existing_video_store_scan",
            }
            index[digest] = row
            if not args.dry_run:
                append_jsonl(index_path, row)
            indexed_existing += 1
        print(f"[index] {idx} video-store files scanned; indexed_new={indexed_existing}", flush=True)

    candidates = iter_project_raw_upload_videos(project_root)
    replacements: dict[str, str] = {}
    moved = 0
    deleted_duplicates = 0
    skipped = 0
    freed_bytes = 0
    for idx, old_path in enumerate(candidates, start=1):
        try:
            old_path.relative_to(project_root)
        except ValueError:
            raise RuntimeError(f"Refusing to migrate path outside project root: {old_path}")
        size_bytes = old_path.stat().st_size
        digest = sha256_file(old_path)
        existing_row = index.get(digest)
        existing_path = Path(str(existing_row.get("absolute_path") or existing_row.get("path"))).resolve() if existing_row else None
        action = "skip"
        if existing_path and existing_path.exists() and existing_path != old_path:
            store_path = existing_path
            action = "delete_project_duplicate"
            if not args.dry_run:
                write_video_ref(old_path, store_path, digest, size_bytes, source="project_duplicate_removed")
                old_path.unlink()
            deleted_duplicates += 1
            freed_bytes += size_bytes
        else:
            store_path = final_store_path(video_root, digest, old_path.suffix)
            if store_path.exists() and store_path.resolve() != old_path:
                action = "delete_project_duplicate_existing_hash_path"
                if not args.dry_run:
                    write_video_ref(old_path, store_path, digest, size_bytes, source="project_duplicate_removed")
                    old_path.unlink()
                deleted_duplicates += 1
                freed_bytes += size_bytes
            else:
                action = "move_to_video_store"
                if not args.dry_run:
                    store_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_path), str(store_path))
                    write_video_ref(old_path, store_path, digest, size_bytes, source="project_raw_moved")
                    row = {
                        "schema_version": "lab_video_store.hash_index.v1",
                        "sha256": digest,
                        "path": str(store_path),
                        "absolute_path": str(store_path),
                        "size_bytes": size_bytes,
                        "original_filename": old_path.name,
                        "indexed_at": now_iso(),
                        "index_source": "project_raw_video_migration",
                    }
                    append_jsonl(index_path, row)
                    index[digest] = row
                moved += 1
                freed_bytes += size_bytes
        if action == "skip":
            skipped += 1
        replacements[str(old_path)] = str(store_path)
        append_jsonl(
            manifest_path,
            {
                "old_project_path": str(old_path),
                "video_store_path": str(store_path),
                "sha256": digest,
                "size_bytes": size_bytes,
                "action": action,
                "dry_run": args.dry_run,
                "processed_at": now_iso(),
            },
        )
        print(
            f"[migrate] {idx}/{len(candidates)} {action} size_gb={size_bytes / (1024**3):.3f} freed_gb={freed_bytes / (1024**3):.3f}",
            flush=True,
        )

    metadata = {"files_scanned": 0, "files_updated": 0, "replacements": 0}
    if not args.dry_run:
        metadata = replace_paths_in_metadata(project_root, replacements)
    summary = {
        "schema_version": "lab_video_store.project_raw_video_migration.v1",
        "project_root": str(project_root),
        "video_store_root": str(video_root),
        "hash_index": str(index_path),
        "manifest_path": str(manifest_path),
        "dry_run": args.dry_run,
        "candidate_count": len(candidates),
        "moved_to_video_store": moved,
        "deleted_project_duplicates": deleted_duplicates,
        "skipped": skipped,
        "freed_bytes": freed_bytes,
        "freed_gb": round(freed_bytes / (1024**3), 3),
        "metadata_rewrite": metadata,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "completed_at": now_iso(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
