from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _required_ppe_keys(rules: Dict[str, Any]) -> List[str]:
    ppe_cfg = rules.get("ppe_requirements", {}) if isinstance(rules, dict) else {}
    must_wear = ppe_cfg.get("must_wear", []) if isinstance(ppe_cfg, dict) else []
    key_map = {
        "gloves": "wear_gloves",
        "goggles": "wear_goggles",
        "lab_coat": "wear_lab_coat",
    }
    keys = [key_map[str(x).strip().lower()] for x in must_wear if str(x).strip().lower() in key_map]
    return keys or ["wear_gloves", "wear_goggles"]


def _human_presence(det: Dict[str, Any]) -> bool:
    layer1 = det.get("layer_outputs", {}).get("layer1_realtime_pose", {})
    pose_instances = int(layer1.get("pose_instances", 0)) if isinstance(layer1, dict) else 0
    if pose_instances > 0:
        return True
    labels = {str(o.get("label", "")).strip().lower() for o in det.get("objects", []) if isinstance(o, dict)}
    return bool(labels.intersection({"person", "human", "operator", "worker"}))


def _build_detection_index(detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    idx: Dict[int, Dict[str, Any]] = {}
    for d in detections:
        if not isinstance(d, dict):
            continue
        try:
            fid = int(d.get("frame_id", -1))
        except Exception:
            continue
        idx[fid] = d
    return idx


def analyze(batch_monitor_dir: Path, rules: Dict[str, Any]) -> Dict[str, Any]:
    required = _required_ppe_keys(rules)

    diagnostics: List[Dict[str, Any]] = []
    rule_counter: Counter[str] = Counter()
    missing_ppe_counter: Counter[str] = Counter()
    missing_ppe_samples: Counter[str] = Counter()

    for sample_file in sorted(batch_monitor_dir.glob("*.json")):
        if sample_file.name == "summary.json":
            continue
        data = _load_json(sample_file)
        if not isinstance(data, dict):
            continue
        sample_id = str(sample_file.stem)
        detections = data.get("detections", []) if isinstance(data.get("detections", []), list) else []
        violations = data.get("violations", []) if isinstance(data.get("violations", []), list) else []
        det_index = _build_detection_index(detections)

        for v in violations:
            if not isinstance(v, dict):
                continue
            rule_id = str(v.get("rule_id", "unknown"))
            rule_counter[rule_id] += 1
            frame_id = int(v.get("frame_id", -1))
            det = det_index.get(frame_id, {})
            ppe = det.get("ppe", {}) if isinstance(det.get("ppe", {}), dict) else {}
            labels = sorted(
                {
                    str(o.get("label", "")).strip().lower()
                    for o in det.get("objects", [])
                    if isinstance(o, dict) and str(o.get("label", "")).strip()
                }
            )
            missing = [k for k in required if not bool(ppe.get(k, False))]
            for k in missing:
                missing_ppe_counter[k] += 1
            if rule_id == "missing_ppe":
                missing_ppe_samples[sample_id] += 1

            diagnostics.append(
                {
                    "sample_id": sample_id,
                    "rule_id": rule_id,
                    "severity": v.get("severity"),
                    "message": v.get("message"),
                    "frame_id": frame_id,
                    "timestamp_sec": v.get("timestamp_sec"),
                    "human_presence": _human_presence(det),
                    "ppe_flags": {k: bool(ppe.get(k, False)) for k in required},
                    "missing_ppe_items": missing,
                    "detected_labels": labels,
                }
            )

    recommendations: List[str] = []
    total_v = len(diagnostics)
    if total_v == 0:
        recommendations.append("No violations detected; keep current thresholds and expand validation set.")
    else:
        if rule_counter.get("missing_ppe", 0) == total_v:
            recommendations.append(
                "All current violations are missing_ppe; prioritize PPE classifier quality before adding new SOP rules."
            )

        lab_coat_miss = missing_ppe_counter.get("wear_lab_coat", 0)
        if total_v > 0 and (lab_coat_miss / total_v) >= 0.8:
            recommendations.append(
                "wear_lab_coat is missing in most violations; consider either collecting/labeling lab_coat samples or temporarily removing lab_coat from configs/sop/rules.yaml must_wear for MVP phase."
            )

        if len(missing_ppe_samples) > 0:
            heavy = sorted(missing_ppe_samples.items(), key=lambda kv: (-kv[1], kv[0]))
            top_sid, top_cnt = heavy[0]
            recommendations.append(
                f"Focus manual review on sample '{top_sid}' first (missing_ppe count={top_cnt})."
            )

    return {
        "batch_monitor_dir": str(batch_monitor_dir),
        "total_violations": total_v,
        "rule_distribution": dict(rule_counter),
        "missing_ppe_item_distribution": dict(missing_ppe_counter),
        "diagnostics": diagnostics,
        "recommendations": recommendations,
    }


def render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Violation Diagnostics Report")
    lines.append("")
    lines.append(f"- total_violations: {report.get('total_violations', 0)}")
    lines.append("")

    lines.append("## Rule Distribution")
    rd = report.get("rule_distribution", {})
    if isinstance(rd, dict) and rd:
        for k, v in sorted(rd.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Missing PPE Item Distribution")
    mp = report.get("missing_ppe_item_distribution", {})
    if isinstance(mp, dict) and mp:
        for k, v in sorted(mp.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Recommendations")
    rec = report.get("recommendations", [])
    if isinstance(rec, list) and rec:
        for r in rec:
            lines.append(f"- {r}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Violation Rows")
    rows = report.get("diagnostics", [])
    if isinstance(rows, list) and rows:
        for r in rows:
            sid = r.get("sample_id", "")
            rid = r.get("rule_id", "")
            ts = r.get("timestamp_sec", "")
            miss = r.get("missing_ppe_items", [])
            hp = r.get("human_presence", False)
            lines.append(f"- {sid}: rule={rid}, t={ts}, human_presence={hp}, missing={miss}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostics for monitor/export violations")
    parser.add_argument("--batch-monitor-dir", default="outputs/predictions/batch_monitor")
    parser.add_argument("--rules", default="configs/sop/rules.yaml")
    parser.add_argument("--out-json", default="outputs/reports/violation_diagnostics.json")
    parser.add_argument("--out-md", default="outputs/reports/violation_diagnostics.md")
    args = parser.parse_args()

    rules = _load_json(Path(args.rules)) if str(args.rules).lower().endswith(".json") else None
    if rules is None:
        import yaml

        rules = yaml.safe_load(Path(args.rules).read_text(encoding="utf-8")) or {}

    report = analyze(Path(args.batch_monitor_dir), rules)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(report), encoding="utf-8")

    print(f"diagnostics json: {out_json}")
    print(f"diagnostics md: {out_md}")
    print(f"total violations: {report.get('total_violations', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
