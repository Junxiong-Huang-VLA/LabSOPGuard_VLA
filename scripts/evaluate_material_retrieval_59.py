from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labsopguard.material_maintenance import rebuild_workspace_published_materials_index
from labsopguard.material_retrieval_eval import evaluate_material_retrieval_quality


def _validate_rate(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be a number between 0.0 and 1.0: {value}") from exc
    if not (0.0 <= parsed <= 1.0):
        raise argparse.ArgumentTypeError(f"{label} must be between 0.0 and 1.0: {value}")
    return parsed


def _fixture_index(fixture_json: str) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmp = tempfile.TemporaryDirectory()
    tmp_root = Path(tmp.name)
    experiments_root = tmp_root / "experiments"
    fixture = json.loads(Path(fixture_json).read_text(encoding="utf-8"))
    published = fixture.get("published") if isinstance(fixture, dict) else {}
    items = [item for item in (published.get("items") or []) if isinstance(item, dict)]
    experiment_id = next((str(item.get("experiment_id")) for item in items if item.get("experiment_id")), "material_retrieval_fixture")
    exp_dir = experiments_root / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    payload = {**published, "items": items, "total": len(items)}
    (exp_dir / "published_materials.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    index_path = tmp_root / "published_materials_index.sqlite"
    rebuild_workspace_published_materials_index(experiments_root, index_path)
    return tmp, index_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate 5.9 canonical material retrieval quality.")
    parser.add_argument("--index", default=str(PROJECT_ROOT / "outputs" / "materials" / "published_materials_index.sqlite"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--fixture-json", default="", help="Build a temporary retrieval index from a published-material fixture.")
    parser.add_argument("--strict", action="store_true", help="Fail on insufficient retrieval quality.")
    parser.add_argument("--min-canonical-hit-rate", type=lambda value: _validate_rate(value, "min-canonical-hit-rate"), default=1.0)
    parser.add_argument("--min-top-k-hit-rate", type=lambda value: _validate_rate(value, "min-top-k-hit-rate"), default=0.9)
    parser.add_argument("--min-top1-hit-rate", type=lambda value: _validate_rate(value, "min-top1-hit-rate"), default=0.9)
    args = parser.parse_args()

    tmp: tempfile.TemporaryDirectory[str] | None = None
    index_path = Path(args.index)
    if args.fixture_json:
        tmp, index_path = _fixture_index(args.fixture_json)
    try:
        report = evaluate_material_retrieval_quality(index_path, top_k=args.top_k)
    finally:
        if tmp is not None:
            tmp.cleanup()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if not args.strict:
        return 0

    failures: list[str] = []
    if report["canonical_hit_rate"] < args.min_canonical_hit_rate:
        failures.append(f"canonical_hit_rate {report['canonical_hit_rate']} < {args.min_canonical_hit_rate}")
    if report["top_k_hit_rate"] < args.min_top_k_hit_rate:
        failures.append(f"top_k_hit_rate {report['top_k_hit_rate']} < {args.min_top_k_hit_rate}")
    if report["top1_hit_rate"] < args.min_top1_hit_rate:
        failures.append(f"top1_hit_rate {report['top1_hit_rate']} < {args.min_top1_hit_rate}")
    if failures:
        print("FAIL: retrieval evaluation does not meet thresholds", file=sys.stderr)
        for item in failures:
            print(f" - {item}", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
