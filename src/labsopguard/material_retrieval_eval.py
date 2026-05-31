from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from labsopguard.material_maintenance import query_workspace_published_materials


MATERIAL_RETRIEVAL_EVAL_SCHEMA_VERSION = "material_retrieval_quality_eval.v2"

DEFAULT_MATERIAL_RETRIEVAL_QUERIES: List[Dict[str, str]] = [
    {"query": "天平称量", "expected_canonical_action_type": "hand-balance", "expected_canonical_object": "balance"},
    {"query": "电子天平称样", "expected_canonical_action_type": "hand-balance", "expected_canonical_object": "balance"},
    {"query": "把样品放到天平上", "expected_canonical_action_type": "hand-balance", "expected_canonical_object": "balance"},
    {"query": "balance weighing", "expected_canonical_action_type": "hand-balance", "expected_canonical_object": "balance"},
    {"query": "称量盘附近操作", "expected_canonical_action_type": "hand-balance", "expected_canonical_object": "balance"},
    {"query": "天平读数前的称量动作", "expected_canonical_action_type": "hand-balance", "expected_canonical_object": "balance"},
    {"query": "取试剂瓶", "expected_canonical_action_type": "hand-bottle", "expected_canonical_object": "bottle"},
    {"query": "拿起试剂瓶", "expected_canonical_action_type": "hand-bottle", "expected_canonical_object": "bottle"},
    {"query": "试剂瓶倾倒", "expected_canonical_action_type": "hand-bottle", "expected_canonical_object": "bottle"},
    {"query": "reagent bottle handling", "expected_canonical_action_type": "hand-bottle", "expected_canonical_object": "bottle"},
    {"query": "瓶口靠近容器", "expected_canonical_action_type": "hand-bottle", "expected_canonical_object": "bottle"},
    {"query": "手持试剂瓶加料", "expected_canonical_action_type": "hand-bottle", "expected_canonical_object": "bottle"},
    {"query": "药匙加样", "expected_canonical_action_type": "hand-spatula", "expected_canonical_object": "spatula"},
    {"query": "药匙取样", "expected_canonical_action_type": "hand-spatula", "expected_canonical_object": "spatula"},
    {"query": "spatula solid transfer", "expected_canonical_action_type": "hand-spatula", "expected_canonical_object": "spatula"},
    {"query": "用药匙转移固体", "expected_canonical_action_type": "hand-spatula", "expected_canonical_object": "spatula"},
    {"query": "药匙靠近称量纸", "expected_canonical_action_type": "hand-spatula", "expected_canonical_object": "spatula"},
    {"query": "固体样品刮取", "expected_canonical_action_type": "hand-spatula", "expected_canonical_object": "spatula"},
    {"query": "称量纸", "expected_canonical_action_type": "hand-paper", "expected_canonical_object": "paper"},
    {"query": "放置称量纸", "expected_canonical_action_type": "hand-paper", "expected_canonical_object": "paper"},
    {"query": "weighing paper prep", "expected_canonical_action_type": "hand-paper", "expected_canonical_object": "paper"},
    {"query": "手拿称量纸", "expected_canonical_action_type": "hand-paper", "expected_canonical_object": "paper"},
    {"query": "纸片放置样品", "expected_canonical_action_type": "hand-paper", "expected_canonical_object": "paper"},
    {"query": "称量纸上有固体", "expected_canonical_action_type": "hand-paper", "expected_canonical_object": "paper"},
    {"query": "容器承接", "expected_canonical_action_type": "hand-container", "expected_canonical_object": "container"},
    {"query": "烧杯承接", "expected_canonical_action_type": "hand-container", "expected_canonical_object": "container"},
    {"query": "container receiving sample", "expected_canonical_action_type": "hand-container", "expected_canonical_object": "container"},
    {"query": "手持容器接样", "expected_canonical_action_type": "hand-container", "expected_canonical_object": "container"},
    {"query": "样品进入烧杯", "expected_canonical_action_type": "hand-container", "expected_canonical_object": "container"},
    {"query": "容器承接试剂", "expected_canonical_action_type": "hand-container", "expected_canonical_object": "container"},
]


def _empty_reason(index_path: Path, hits: List[Dict[str, Any]], expected_action: str, expected_object: Optional[str]) -> Optional[str]:
    if hits:
        return None
    if not index_path.exists():
        return "missing_workspace_material_index"
    if expected_object:
        return f"no_result_for_{expected_action}_{expected_object}"
    return f"no_result_for_{expected_action}"


def evaluate_material_retrieval_quality(
    index_path: str | Path,
    *,
    queries: Optional[Iterable[Dict[str, str]]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Evaluate whether canonical material search is finding the expected evidence classes."""

    path = Path(index_path)
    query_rows = list(queries or DEFAULT_MATERIAL_RETRIEVAL_QUERIES)
    limit = max(1, int(top_k))
    results: List[Dict[str, Any]] = []
    top1_hits = 0
    top3_hits = 0
    top_k_hits = 0
    canonical_hits = 0
    canonical_action_hits = 0
    canonical_object_hits = 0
    for row in query_rows:
        query = row["query"]
        expected_action = row["expected_canonical_action_type"]
        expected_object = row.get("expected_canonical_object")
        payload = query_workspace_published_materials(
            path,
            text=query,
            sort_by="relevance",
            sort_order="desc",
            limit=limit,
        )
        hits = [item for item in (payload.get("items") or []) if isinstance(item, dict)]
        top_actions = [str(item.get("canonical_action_type") or "") for item in hits]
        top_objects = [str(item.get("canonical_object") or "") for item in hits]
        top1_action = top_actions[0] if top_actions else None
        top1_object = top_objects[0] if top_objects else None
        top3_actions = top_actions[:3]
        top3_objects = top_objects[:3]
        top_k_actions = top_actions[:limit]
        top_k_objects = top_objects[:limit]

        action_top3_hit = expected_action in top3_actions
        object_top3_hit = not expected_object or expected_object in top3_objects
        action_top_k_hit = expected_action in top_k_actions
        object_top_k_hit = not expected_object or expected_object in top_k_objects
        top_k_hit = action_top_k_hit and object_top_k_hit
        top1_hit = top1_action == expected_action and (not expected_object or top1_object == expected_object)
        top3_hit = action_top3_hit and object_top3_hit
        if action_top_k_hit:
            canonical_action_hits += 1
        if object_top_k_hit:
            canonical_object_hits += 1
        if top_k_hit:
            canonical_hits += 1
            top_k_hits += 1
        if top1_hit:
            top1_hits += 1
        if top3_hit:
            top3_hits += 1
        results.append(
            {
                "query": query,
                "expected_canonical_action_type": expected_action,
                "expected_canonical_object": expected_object,
                "returned": len(hits),
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "top_k_hit": top_k_hit,
                "canonical_action_hit": action_top3_hit,
                "canonical_object_hit": object_top3_hit,
                "canonical_action_top_k_hit": action_top_k_hit,
                "canonical_action_top3_hit": action_top3_hit,
                "canonical_object_top_k_hit": object_top_k_hit,
                "canonical_object_top3_hit": object_top3_hit,
                "top1_canonical_action_type": top1_action,
                "top1_canonical_object": top1_object,
                "top3_canonical_action_types": top3_actions,
                "top3_canonical_objects": top3_objects,
                "top_k_canonical_action_types": top_k_actions,
                "top_k_canonical_objects": top_k_objects,
                "wrong_classes": sorted({action for action in top3_actions if action and action != expected_action}),
                "empty_result_reason": _empty_reason(path, hits, expected_action, expected_object),
                "top_k_action_types": top_actions,
                "top_k_material_ids": [item.get("material_id") or item.get("item_id") for item in hits],
            }
        )

    total = len(query_rows)
    return {
        "schema_version": MATERIAL_RETRIEVAL_EVAL_SCHEMA_VERSION,
        "index_path": str(path),
        "top_k": limit,
        "query_count": total,
        "canonical_hit_count": canonical_hits,
        "top_k_hit_count": top_k_hits,
        "top1_hit_count": top1_hits,
        "top3_hit_count": top3_hits,
        "canonical_hit_rate": round(canonical_hits / total, 4) if total else 0.0,
        "top_k_hit_rate": round(top_k_hits / total, 4) if total else 0.0,
        "top1_hit_rate": round(top1_hits / total, 4) if total else 0.0,
        "top3_hit_rate": round(top3_hits / total, 4) if total else 0.0,
        "canonical_action_hit_count": canonical_action_hits,
        "canonical_object_hit_count": canonical_object_hits,
        "canonical_action_hit_rate": round(canonical_action_hits / total, 4) if total else 0.0,
        "canonical_object_hit_rate": round(canonical_object_hits / total, 4) if total else 0.0,
        "queries": results,
    }
