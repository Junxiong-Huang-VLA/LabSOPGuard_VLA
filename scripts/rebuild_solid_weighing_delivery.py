from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


EXPERIMENT_ID = "2190fe06-3619-45fc-96ef-1bb8afb9bdf9"
EXPERIMENT_TITLE = "固体称量实验"
EXPERIMENT_DATE = "20260504"
EXPERIMENT_LABEL = f"{EXPERIMENT_TITLE}_{EXPERIMENT_DATE}"

KEYFRAME_DIR = "关键帧"
KEYCLIP_DIR = "关键片段"
REPORT_DIR = "专业报告"

ACTION_WINDOWS = [
    {
        "id": "seg_000001",
        "title": "称量准备与天平接触",
        "start_sec": 12.0,
        "end_sec": 18.0,
        "objects": ["gloved_hand", "balance", "sample_bottle"],
        "description": "双视角恢复片段，覆盖操作者接近天平并完成称量前准备动作。",
    },
    {
        "id": "seg_000002",
        "title": "样品瓶与试剂瓶交互",
        "start_sec": 21.0,
        "end_sec": 33.0,
        "objects": ["gloved_hand", "sample_bottle", "reagent_bottle", "balance"],
        "description": "双视角恢复片段，覆盖样品瓶、试剂瓶与天平附近的手-物交互。",
    },
    {
        "id": "seg_000003",
        "title": "称量过程核心操作",
        "start_sec": 35.0,
        "end_sec": 48.0,
        "objects": ["gloved_hand", "balance", "reagent_bottle"],
        "description": "双视角恢复片段，覆盖称量过程中手、天平和容器的核心交互。",
    },
    {
        "id": "seg_000004",
        "title": "收尾与物体移动",
        "start_sec": 52.0,
        "end_sec": 59.0,
        "objects": ["gloved_hand", "balance"],
        "description": "双视角恢复片段，覆盖实验末段的物体移动和收尾动作。",
    },
]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _probe_duration(path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        encoding="utf-8",
    )
    return float(output.strip())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _find_videos(video_dir: Path) -> tuple[Path, Path]:
    files = sorted(video_dir.glob("*.mp4"))
    top = next((p for p in files if "top_view" in p.name.lower()), None)
    bottom = next((p for p in files if "bottom_view" in p.name.lower()), None)
    if top is None or bottom is None:
        raise FileNotFoundError(f"Expected top_view and bottom_view mp4 files in {video_dir}")
    return top, bottom


def _clear_known_outputs(root: Path) -> None:
    for name in (KEYFRAME_DIR, KEYCLIP_DIR, REPORT_DIR):
        target = root / name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "README.md", "素材索引.json", "素材索引.jsonl"):
        target = root / name
        if target.exists():
            target.unlink()


def _extract_keyframe(top_video: Path, bottom_video: Path, timestamp: float, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(top_video),
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(bottom_video),
            "-filter_complex",
            "[0:v]scale=960:540,setpts=PTS-STARTPTS[top];"
            "[1:v]scale=960:540,setpts=PTS-STARTPTS[bottom];"
            "[top][bottom]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(target),
        ]
    )


def _extract_clip(top_video: Path, bottom_video: Path, start: float, end: float, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.5, end - start)
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(top_video),
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(bottom_video),
            "-filter_complex",
            "[0:v]scale=960:540,setpts=PTS-STARTPTS[top];"
            "[1:v]scale=960:540,setpts=PTS-STARTPTS[bottom];"
            "[top][bottom]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )


def _record(
    *,
    role: str,
    asset_kind: str,
    path: Path,
    source_videos: list[Path],
    action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_title": EXPERIMENT_TITLE,
        "experiment_date": EXPERIMENT_DATE,
        "asset_kind": asset_kind,
        "role": role,
        "stored_file": str(path),
        "file_name": path.name,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": _sha256(path) if path.exists() else None,
        "source_videos": [str(p) for p in source_videos],
        "recovery_status": "recovered_from_source_video",
        "review_status": "accepted_recovery",
        "generation_method": "ffmpeg_dual_view_hstack",
        "action": action,
    }


def _copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _make_pdf(report_pdf: Path, keyframes: list[Path], manifest: dict[str, Any]) -> None:
    report_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font_name = "msyh" if font_path.exists() else "helv"

    def add_text(page: fitz.Page, rect: fitz.Rect, text: str, size: float = 11, align: int = 0) -> None:
        kwargs = {"fontsize": size, "align": align}
        if font_path.exists():
            kwargs["fontname"] = font_name
        page.insert_textbox(rect, text, **kwargs)

    page = doc.new_page(width=595, height=842)
    if font_path.exists():
        page.insert_font(fontname=font_name, fontfile=str(font_path))
    add_text(page, fitz.Rect(46, 44, 550, 88), f"{EXPERIMENT_TITLE} 专业分析报告（恢复版）", 20, 1)
    add_text(
        page,
        fitz.Rect(54, 104, 540, 184),
        "本报告由真实双视角实验视频重新生成。原始分析产物已丢失，因此本恢复版保留真实视频来源、"
        "关键时间窗、关键帧、关键片段和素材索引，但不声称等同于被删除前的 YOLO/VLM 完整审核结果。",
        11,
    )
    lines = [
        f"实验编号：{EXPERIMENT_ID}",
        f"实验名称：{EXPERIMENT_TITLE}",
        f"实验日期：{EXPERIMENT_DATE}",
        f"生成时间：{manifest['generated_at']}",
        "视频来源：",
    ]
    for video in manifest["source_videos"]:
        lines.append(f"- {video['view']}: {video['path']} ({video['duration_sec']:.2f}s)")
    add_text(page, fitz.Rect(54, 205, 540, 340), "\n".join(lines), 10)

    y = 360
    add_text(page, fitz.Rect(54, y, 540, y + 22), "恢复关键动作时间窗", 14)
    y += 30
    for action in ACTION_WINDOWS:
        add_text(
            page,
            fitz.Rect(64, y, 530, y + 58),
            f"{action['id']}  {action['title']}  {action['start_sec']:.1f}-{action['end_sec']:.1f}s\n"
            f"对象：{', '.join(action['objects'])}\n{action['description']}",
            9,
        )
        y += 72

    for idx, image_path in enumerate(keyframes, start=1):
        page = doc.new_page(width=595, height=842)
        if font_path.exists():
            page.insert_font(fontname=font_name, fontfile=str(font_path))
        add_text(page, fitz.Rect(46, 38, 550, 68), f"关键帧 {idx}: {image_path.name}", 14, 1)
        with Image.open(image_path) as img:
            width, height = img.size
        max_w, max_h = 500, 610
        scale = min(max_w / width, max_h / height)
        w, h = width * scale, height * scale
        x0 = (595 - w) / 2
        y0 = 92
        page.insert_image(fitz.Rect(x0, y0, x0 + w, y0 + h), filename=str(image_path))
        action = ACTION_WINDOWS[idx - 1]
        add_text(
            page,
            fitz.Rect(54, y0 + h + 24, 540, 790),
            f"对应片段：{action['title']}\n时间窗：{action['start_sec']:.1f}-{action['end_sec']:.1f}s\n"
            f"说明：{action['description']}",
            10,
        )
    doc.save(report_pdf)
    doc.close()


def rebuild(video_dir: Path, output_root: Path) -> dict[str, Any]:
    top_video, bottom_video = _find_videos(video_dir)
    generated_at = datetime.now().isoformat(timespec="seconds")
    experiment_dir = output_root / "experiments" / EXPERIMENT_ID
    simplified_root = output_root / "material_references" / EXPERIMENT_LABEL
    formal_root = experiment_dir / "material_references"
    reports_dir = experiment_dir / "reports"
    raw_dir = experiment_dir / "raw"

    for root in (simplified_root, formal_root):
        _clear_known_outputs(root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(top_video, raw_dir / "top_view.browser_h264.mp4")
    shutil.copy2(bottom_video, raw_dir / "bottom_view.browser_h264.mp4")

    source_videos = [
        {"view": "top", "path": str(top_video), "duration_sec": _probe_duration(top_video), "size_bytes": top_video.stat().st_size},
        {
            "view": "bottom",
            "path": str(bottom_video),
            "duration_sec": _probe_duration(bottom_video),
            "size_bytes": bottom_video.stat().st_size,
        },
    ]
    manifest: dict[str, Any] = {
        "schema_version": "material_references.recovered_delivery.v1",
        "generated_at": generated_at,
        "experiment_id": EXPERIMENT_ID,
        "experiment_title": EXPERIMENT_TITLE,
        "experiment_date": EXPERIMENT_DATE,
        "experiment_label": EXPERIMENT_LABEL,
        "recovery_status": "recovered_from_source_video",
        "source_videos": source_videos,
        "keyframe_folder": str(simplified_root / KEYFRAME_DIR),
        "key_clip_folder": str(simplified_root / KEYCLIP_DIR),
        "report_folder": str(simplified_root / REPORT_DIR),
        "actions": ACTION_WINDOWS,
    }

    records: list[dict[str, Any]] = []
    keyframes: list[Path] = []
    source_paths = [top_video, bottom_video]
    for idx, action in enumerate(ACTION_WINDOWS, start=1):
        midpoint = (float(action["start_sec"]) + float(action["end_sec"])) / 2
        safe_title = action["title"].replace("/", "_")
        keyframe = simplified_root / KEYFRAME_DIR / f"{idx:02d}_{safe_title}_{midpoint:.1f}s_双视角.jpg"
        keyclip = simplified_root / KEYCLIP_DIR / f"{idx:02d}_{safe_title}_{action['start_sec']:.1f}-{action['end_sec']:.1f}s_双视角.mp4"
        _extract_keyframe(top_video, bottom_video, midpoint, keyframe)
        _extract_clip(top_video, bottom_video, float(action["start_sec"]), float(action["end_sec"]), keyclip)
        keyframes.append(keyframe)
        records.append(_record(role="keyframe", asset_kind=KEYFRAME_DIR, path=keyframe, source_videos=source_paths, action=action))
        records.append(_record(role="key_clip", asset_kind=KEYCLIP_DIR, path=keyclip, source_videos=source_paths, action=action))

    report_pdf = simplified_root / REPORT_DIR / "professional_report_recovered.pdf"
    _make_pdf(report_pdf, keyframes, manifest)
    report_html = simplified_root / REPORT_DIR / "professional_report_recovered.html"
    report_json = simplified_root / REPORT_DIR / "professional_report_recovered.json"
    report_manifest = simplified_root / REPORT_DIR / "professional_report_manifest.json"
    report_html.write_text(
        "<!doctype html><meta charset='utf-8'><title>固体称量实验恢复报告</title>"
        f"<h1>{EXPERIMENT_TITLE} 专业分析报告（恢复版）</h1>"
        "<p>本报告由真实双视角视频重新生成，原始完整 YOLO/VLM 审核产物已丢失。</p>"
        + "".join(
            f"<h2>{a['id']} {a['title']}</h2><p>{a['start_sec']:.1f}-{a['end_sec']:.1f}s：{a['description']}</p>"
            for a in ACTION_WINDOWS
        ),
        encoding="utf-8",
    )
    _write_json(report_json, {"manifest": manifest, "records": records})
    _write_json(report_manifest, {"schema_version": "professional_report.recovered.v1", **manifest})
    for role, path in [
        ("professional_report_pdf", report_pdf),
        ("professional_report_html", report_html),
        ("professional_report_json", report_json),
        ("professional_report_manifest", report_manifest),
    ]:
        records.append(_record(role=role, asset_kind=REPORT_DIR, path=path, source_videos=source_paths))

    manifest.update(
        {
            "file_count": len(records),
            "keyframe_count": sum(1 for row in records if row["asset_kind"] == KEYFRAME_DIR),
            "key_clip_count": sum(1 for row in records if row["asset_kind"] == KEYCLIP_DIR),
            "report_count": sum(1 for row in records if row["asset_kind"] == REPORT_DIR),
            "index_json": str(simplified_root / "素材索引.json"),
            "index_jsonl": str(simplified_root / "素材索引.jsonl"),
        }
    )
    _write_json(simplified_root / "manifest.json", manifest)
    _write_json(simplified_root / "素材索引.json", {"schema_version": "material_index.recovered.v1", "records": records})
    _write_jsonl(simplified_root / "素材索引.jsonl", records)
    (simplified_root / "README.md").write_text(
        f"# {EXPERIMENT_LABEL}\n\n"
        "这是一次从真实双视角视频重新生成的恢复版交付包。\n\n"
        f"- {KEYFRAME_DIR}: {manifest['keyframe_count']} 个双视角关键帧\n"
        f"- {KEYCLIP_DIR}: {manifest['key_clip_count']} 个双视角关键片段\n"
        f"- {REPORT_DIR}: {manifest['report_count']} 个报告文件\n\n"
        "注意：原始完整 YOLO/VLM 审核产物已丢失，本交付包不伪造原审核结论；所有素材均来自给定真实视频。\n",
        encoding="utf-8",
    )

    _copy_tree_contents(simplified_root, formal_root)
    _copy_tree_contents(simplified_root / REPORT_DIR, reports_dir)
    _write_json(experiment_dir / "manifest.json", manifest)
    return {"simplified_root": str(simplified_root), "experiment_dir": str(experiment_dir), "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("C:/Users/Xx7/Desktop") / "双视角实验视频",
    )
    parser.add_argument("--output-root", type=Path, default=Path("D:/LabCapability/LabSOPGuard/outputs"))
    args = parser.parse_args()
    result = rebuild(args.video_dir, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
