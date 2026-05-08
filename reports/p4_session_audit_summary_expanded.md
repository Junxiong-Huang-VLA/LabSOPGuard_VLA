# P4 Session Audit Summary

- Generated: `2026-05-08T10:26:24.765390+00:00`
- Sessions audited: `3`
- Health pass: `3/3`
- QA pass: `1/3`
- Average candidate ratio: `0.183932`
- Total strong micro evidence: `10`
- Total retrieval-only micro evidence: `16`

## Final Acceptance Table

| Session | QA | Health | Candidate ratio | Strong | Retrieval-only | Segment backfill promoted | Chinese queries | Dual-view traceability | Query semantic diversity |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | `pass` | `pass` | 0.185185 | 2 | 6 | 0 | 4 | 8/8 | 0.5 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | `needs_review` | `pass` | 0.10687 | 6 | 2 | 0 | 4 | 8/8 | 0.5 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | `needs_review` | `pass` | 0.25974 | 2 | 8 | 0 | 4 | 8/8 | 0.5 |

## Session Metrics

| Session | Segments | Micro | Strong | Retrieval-only | Video events | Candidate ratio | Rollup | QA | Health | Queue | Risks |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | 5 | 8 | 2 | 6 | 81 | 0.185185 | 28 | `pass` | `pass` | 0 | 0 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | 1 | 8 | 6 | 2 | 131 | 0.10687 | 49 | `needs_review` | `pass` | 0 | 1 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | 6 | 10 | 2 | 8 | 77 | 0.25974 | 34 | `needs_review` | `pass` | 0 | 1 |

## Risks

### 2190fe06-3619-45fc-96ef-1bb8afb9bdf9
- None

### 3ccd635c-217e-40dd-9922-0e1e397739ce
- `qa_not_pass`: Quality assurance is not passing.

### 53ca6efe-a100-4e86-b041-7c98e2fcc662
- `qa_not_pass`: Quality assurance is not passing.


## Retrieval Acceptance

| Session | Query | Results | Top Result | Traceable | Dual-view Clips | Keyframes |
|---|---|---:|---|---|---|---:|
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | `balance weighing` | 3 | `seg_000001_micro_002` | `True` | `True` | 3 |
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | `pipetting liquid transfer` | 3 | `seg_000002_micro_001` | `True` | `True` | 3 |
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | `sample handling` | 3 | `seg_000002_micro_001` | `True` | `True` | 3 |
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | `recording balance readout` | 3 | `seg_000003_micro_002` | `True` | `True` | 3 |
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | `查找使用天平称量的片段` | 3 | `seg_000001_micro_002` | `True` | `True` | 3 |
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | `查找移液操作片段` | 3 | `seg_000004` | `True` | `True` | 0 |
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | `查找样品处理片段` | 3 | `seg_000002_micro_001` | `True` | `True` | 3 |
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | `查找记录读数片段` | 3 | `seg_000003_micro_002` | `True` | `True` | 3 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | `balance weighing` | 3 | `seg_000001_micro_009` | `True` | `True` | 3 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | `pipetting liquid transfer` | 3 | `seg_000001_micro_010` | `True` | `True` | 3 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | `sample handling` | 3 | `seg_000001_micro_003` | `True` | `True` | 3 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | `recording balance readout` | 3 | `seg_000001_micro_008` | `True` | `True` | 3 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | `查找使用天平称量的片段` | 3 | `seg_000001_micro_009` | `True` | `True` | 3 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | `查找移液操作片段` | 3 | `seg_000001_micro_010` | `True` | `True` | 3 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | `查找样品处理片段` | 3 | `seg_000001_micro_003` | `True` | `True` | 3 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | `查找记录读数片段` | 3 | `seg_000001_micro_008` | `True` | `True` | 3 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | `balance weighing` | 3 | `seg_000004_micro_001` | `True` | `True` | 3 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | `pipetting liquid transfer` | 3 | `seg_000001_micro_003` | `True` | `True` | 3 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | `sample handling` | 3 | `seg_000001_micro_001` | `True` | `True` | 3 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | `recording balance readout` | 3 | `seg_000001_micro_002` | `True` | `True` | 3 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | `查找使用天平称量的片段` | 3 | `seg_000004_micro_001` | `True` | `True` | 3 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | `查找移液操作片段` | 3 | `seg_000001_micro_003` | `True` | `True` | 3 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | `查找样品处理片段` | 3 | `seg_000001_micro_001` | `True` | `True` | 3 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | `查找记录读数片段` | 3 | `seg_000001_micro_002` | `True` | `True` | 3 |

## Query Semantic Diversity

| Session | Queries | Distinct Top Results | Max Top Reuse | Diversity Ratio | Most Reused Top Result |
|---|---:|---:|---:|---:|---|
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | 8 | 4 | 3 | 0.5 | `seg_000002_micro_001` |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | 8 | 4 | 2 | 0.5 | `seg_000001_micro_009` |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | 8 | 4 | 2 | 0.5 | `seg_000004_micro_001` |

## Backfill Evidence Guard

| Session | Retrieval-only | Strong | Segment backfill | Promoted segment backfill |
|---|---:|---:|---:|---:|
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | 6 | 2 | 0 | 0 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | 2 | 6 | 0 | 0 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | 8 | 2 | 3 | 0 |
