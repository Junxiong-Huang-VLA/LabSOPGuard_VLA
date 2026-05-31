# retrieval_boost_factors Schema

**Location:** Within each result from `VectorIndex.query(fusion_weights=...)` and `rerank_results()`

## Description

Explainable fusion scoring breakdown attached to each retrieval result after second-pass reranking. Enables transparency into why a result was ranked at a specific position.

## Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `text_similarity` | float | 0.0-1.0 | Base vector similarity score |
| `time_proximity` | float | 0.0-1.0 | Bonus for shorter/focused segments |
| `dual_view_consistency` | float | 0.6-1.0 | 1.0 if both views present, 0.6 otherwise |
| `evidence_strength` | float | 0.0-1.0 | Based on evidence_level and quality |
| `fusion_score` | float | 0.0-1.0 | Weighted combination of all factors |

## Default Fusion Weights

```json
{
  "text_similarity": 0.50,
  "time_proximity": 0.20,
  "dual_view_consistency": 0.15,
  "evidence_strength": 0.15
}
```

## Evidence Strength Mapping

| evidence_level | Base Score |
|---------------|------------|
| `"strong"` | 1.0 |
| `"moderate"` | 0.7 |
| `"weak"` | 0.4 |
| (missing) | 0.3 |

Quality override: `"high"` → max(score, 0.9), `"medium"` → max(score, 0.6)

## Usage

```python
from key_action_indexer import rerank_results

results = index.query("称量样品", top_k=10)
reranked = rerank_results(
    results,
    fusion_weights={"text_similarity": 0.6, "evidence_strength": 0.4},
    dedup_same_segment=True,
    max_results=5,
)
for r in reranked:
    print(r["retrieval_boost_factors"])
```

## Deduplication

When `dedup_same_segment=True`, results with the same `micro_segment_id` or `segment_id` are deduplicated, keeping only the highest `fusion_score` entry.
