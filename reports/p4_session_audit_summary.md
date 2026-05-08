# P4 Session Audit Summary

- Generated: `2026-05-08T08:40:11.674487+00:00`
- Sessions audited: `3`
- Health pass: `3/3`
- QA pass: `1/3`
- Average candidate ratio: `0.183932`
- Total strong micro evidence: `10`
- Total retrieval-only micro evidence: `16`

## Session Metrics

| Session | Segments | Micro | Strong | Retrieval-only | Video events | Candidate ratio | Rollup | QA | Health | Queue | Risks |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | 5 | 8 | 2 | 6 | 81 | 0.185185 | 28 | `pass` | `pass` | 0 | 1 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | 1 | 8 | 6 | 2 | 131 | 0.10687 | 49 | `needs_review` | `pass` | 4 | 2 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | 6 | 10 | 2 | 8 | 77 | 0.25974 | 34 | `needs_review` | `pass` | 4 | 2 |

## Risks

### 2190fe06-3619-45fc-96ef-1bb8afb9bdf9
- `history_under_sampled`: History model has fewer than 6 sessions.

### 3ccd635c-217e-40dd-9922-0e1e397739ce
- `qa_not_pass`: Quality assurance is not passing.
- `history_under_sampled`: History model has fewer than 6 sessions.

### 53ca6efe-a100-4e86-b041-7c98e2fcc662
- `qa_not_pass`: Quality assurance is not passing.
- `history_under_sampled`: History model has fewer than 6 sessions.

## Query Smoke

### 2190fe06-3619-45fc-96ef-1bb8afb9bdf9
- `balance weighing` -> `3` results, top `seg_000004_micro_001`, traceable `True`

### 3ccd635c-217e-40dd-9922-0e1e397739ce
- `balance weighing` -> `3` results, top `seg_000001_micro_009`, traceable `True`

### 53ca6efe-a100-4e86-b041-7c98e2fcc662
- `balance weighing` -> `3` results, top `seg_000004_micro_001`, traceable `True`
