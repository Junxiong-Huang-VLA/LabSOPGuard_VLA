# Codex Handoff - LabCapability

Generated: 2026-05-08 16:45 Asia/Shanghai

## Project Guardrails

- Work only on the key-action indexer mainline: physical-evidence extraction from long dual-view experiment videos, multimodal time alignment, descriptions, vector indexing, and retrieval.
- Keep `src/key_action_indexer` independent from the existing LabSOPGuard app.
- Prioritize YOLO-backed key action segments, hand-object interaction evidence, micro-segments, multiview clip alignment, metadata, and retrieval.
- Do not reintroduce PTZ, cloud PTZ, camera orchestration, camera port mapping, MQTT tooling, or unrelated infrastructure. PTZ lives in `D:\PtzTracker`.
- Do not treat rendered YOLO boxes as the deliverable; they must feed segment/micro-segment metadata and retrieval.
- Dry-run mode must remain runnable without real videos or ffmpeg.
- Preserve `pytest -q`, backend Python compilation, frontend `npm run build`, existing key-actions pages, and segment-level retrieval.

## Current State

Three real key-action sessions were refreshed and audited:

- `D:\LabCapability\LabSOPGuard\outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\key_action_index`
- `D:\LabCapability\LabSOPGuard\outputs\experiments\3ccd635c-217e-40dd-9922-0e1e397739ce\key_action_index`
- `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index`

Latest audit report:

- Markdown: `D:\LabCapability\reports\p4_session_audit_summary.md`
- JSON: `D:\LabCapability\reports\p4_session_audit_summary.json`
- Generated: `2026-05-08T08:40:11.674487+00:00`
- Health pass: `3/3`
- QA pass: `1/3`
- Average candidate ratio: `0.183932`
- Max candidate ratio: `0.25974`
- Strong process micro evidence: `10`
- Retrieval-only micro evidence: `16`
- Query smoke `balance weighing`: all three sessions return 3 traceable results.

Session metrics:

| Session | Segments | Micro | Strong | Retrieval-only | Video events | Candidate ratio | Rollup removed | QA | Health | Queue |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | 5 | 8 | 2 | 6 | 81 | 0.185185 | 28 | `pass` | `pass` | 0 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | 1 | 8 | 6 | 2 | 131 | 0.10687 | 49 | `needs_review` | `pass` | 4 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | 6 | 10 | 2 | 8 | 77 | 0.25974 | 34 | `needs_review` | `pass` | 4 |

## Completed In This Thread

Implemented and refreshed:

- Added segment-level micro coverage backfill in `src/key_action_indexer/micro_coverage_backfill.py`.
- Wired backfill into `batch-refresh` before micro quality enrichment.
- Added CLI command `micro-coverage`.
- Updated micro quality enrichment so forced segment-level backfill remains retrieval-only:
  - `process_evidence_role = retrieval_candidate`
  - `strong_process_evidence = false`
  - `retrieval_priority_bucket = segment_level_backfill`
  - warning includes `segment_level_retrieval_backfill`
- Updated `video_understanding` so segment-level backfill rows compress into one `segment_level_retrieval_candidate` event instead of expanding into multiple candidate families.
- Improved candidate rollup:
  - same-micro, same-object, same-action cross-family rollup
  - weak same-micro candidate bundle rollup
  - summary now exposes primary, cross-family, and weak-bundle group counts.
- `batch-refresh` now builds/loads stage scope, so known out-of-scope labels do not block model coverage.
- Built the same 3-session `history_model.json` into all three refreshed sessions.
- Generated missing-step recovery artifacts and human review bundles for the two remaining QA-blocked sessions.

Main changed files:

- `src/key_action_indexer/micro_coverage_backfill.py`
- `src/key_action_indexer/micro_quality_enrichment.py`
- `src/key_action_indexer/video_understanding.py`
- `src/key_action_indexer/batch_refresh.py`
- `src/key_action_indexer/cli.py`
- `tests/test_micro_coverage_backfill.py`
- `tests/test_video_understanding.py`
- `tests/test_batch_refresh.py`

## Recovery Artifacts

Generated reports:

- `D:\LabCapability\reports\p4_missing_step_recovery_3ccd635c-217e-40dd-9922-0e1e397739ce.json`
- `D:\LabCapability\reports\p4_missing_step_recovery_53ca6efe-a100-4e86-b041-7c98e2fcc662.json`
- `D:\LabCapability\reports\p4_review_bundle_3ccd635c-217e-40dd-9922-0e1e397739ce.md`
- `D:\LabCapability\reports\p4_review_bundle_53ca6efe-a100-4e86-b041-7c98e2fcc662.md`

Recovery summary:

| Session | Step | Action | Status | Confidence | Video candidates | Transcript candidates | Asset candidates | Best video | Best asset |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `3ccd...` | `step_001` Weighing | `weighing` | `skipped_or_unobserved` | 0.20 | 8 | 0 | 8 | 0.904 | 0.37 |
| `3ccd...` | `step_004` Recording | `recording` | `not_observed` | 0.10 | 8 | 0 | 8 | 0.5831 | 0.49 |
| `53ca...` | `step_003` Sample Handling | `sample_handling` | `not_observed` | 0.10 | 8 | 0 | 8 | 0.5765 | 0.49 |
| `53ca...` | `step_004` Recording | `recording` | `not_observed` | 0.10 | 8 | 0 | 8 | 0.554 | 0.49 |

Review bundle summaries:

- `3ccd...`: 4 pending items: Weighing review, Pipetting review, Sample Handling suggested approve, Recording review.
- `53ca...`: 4 pending items: Weighing suggested approve, Pipetting suggested approve, Sample Handling review, Recording review.

Important: do not auto-approve these. Use the review bundles and candidate clips/keyframes for human confirmation.

## Remaining Blockers

1. `history_under_sampled` remains on all sessions because only 3 real complete sessions were found. The QA/history reuse check can pass with the current model, but audit still correctly flags fewer than 6 source sessions.
2. `3ccd...` QA remains `needs_review`:
   - `step_reasoning` needs review.
   - `completion` needs review; unresolved steps are currently `step_001` and `step_004`.
   - `human_confirmation` has 4 pending queue items.
3. `53ca...` QA remains `needs_review`:
   - `step_reasoning` needs review.
   - `completion` needs review; unresolved steps are currently `step_003` and `step_004`.
   - `human_confirmation` has 4 pending queue items.
4. `53ca...` segment parent micro coverage is now fixed by retrieval-only backfill. The added backfills must not be promoted into strong process evidence unless real visual confirmation supports that later.

## Recommended Next Steps

1. Human-review the pending confirmation queues for `3ccd...` and `53ca...`.
   - Start from the two `p4_review_bundle_*.md` files.
   - Use the corresponding `p4_missing_step_recovery_*.json` files to inspect candidate clips/keyframes.
   - Apply decisions with `confirmation-batch` only after a human-readable decision file is prepared.
2. Add at least 3 more real, complete same-class sessions to the history model to clear `history_under_sampled`.
3. After human decisions and more sessions, rerun `batch-refresh`, rebuild `history-model`, then rerun `audit-sessions`.
4. Keep candidate ratios below the current level; do not remove rollup or let retrieval-only segment backfill explode into multiple event families.
5. Keep multi-camera/PTZ work out of this repo. The separate frontend previously checked is under `D:\MultiCameraMonitor\multi_camera_monitor\frontend-app`.

## Useful Commands

Batch refresh:

```powershell
python -m key_action_indexer.cli batch-refresh --source "D:\LabCapability\LabSOPGuard\outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\key_action_index" --source "D:\LabCapability\LabSOPGuard\outputs\experiments\3ccd635c-217e-40dd-9922-0e1e397739ce\key_action_index" --source "D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index" --query "balance weighing" --output-summary "D:\LabCapability\reports\p4_batch_refresh_summary.json"
```

Audit:

```powershell
python -m key_action_indexer.cli audit-sessions --source "D:\LabCapability\LabSOPGuard\outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\key_action_index" --source "D:\LabCapability\LabSOPGuard\outputs\experiments\3ccd635c-217e-40dd-9922-0e1e397739ce\key_action_index" --source "D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index" --query "balance weighing" --output-json "D:\LabCapability\reports\p4_session_audit_summary.json" --output-md "D:\LabCapability\reports\p4_session_audit_summary.md"
```

Missing-step recovery:

```powershell
python -m key_action_indexer.cli missing-step-recovery --session-dir "D:\LabCapability\LabSOPGuard\outputs\experiments\3ccd635c-217e-40dd-9922-0e1e397739ce\key_action_index" --output "D:\LabCapability\reports\p4_missing_step_recovery_3ccd635c-217e-40dd-9922-0e1e397739ce.json" --confidence-threshold 0.5 --window-padding-sec 5.0
python -m key_action_indexer.cli missing-step-recovery --session-dir "D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index" --output "D:\LabCapability\reports\p4_missing_step_recovery_53ca6efe-a100-4e86-b041-7c98e2fcc662.json" --confidence-threshold 0.5 --window-padding-sec 5.0
```

Review bundles:

```powershell
python -m key_action_indexer.cli review-bundle --session-dir "D:\LabCapability\LabSOPGuard\outputs\experiments\3ccd635c-217e-40dd-9922-0e1e397739ce\key_action_index" --output "D:\LabCapability\reports\p4_review_bundle_3ccd635c-217e-40dd-9922-0e1e397739ce.md" --format md
python -m key_action_indexer.cli review-bundle --session-dir "D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index" --output "D:\LabCapability\reports\p4_review_bundle_53ca6efe-a100-4e86-b041-7c98e2fcc662.md" --format md
```

Validation:

```powershell
python -m pytest -q
python -m compileall -q src LabSOPGuard\backend tests
npm run build
```

For frontend builds, run `npm run build` from:

- `D:\LabCapability\LabSOPGuard\frontend-app`
- `D:\MultiCameraMonitor\multi_camera_monitor\frontend-app`

## Last Validation

The latest full validation before this handoff passed:

- `python -m pytest -q` -> `194 passed`
- `python -m compileall -q src LabSOPGuard\backend tests` -> passed
- `npm run build` in `D:\LabCapability\LabSOPGuard\frontend-app` -> passed
- `npm run build` in `D:\MultiCameraMonitor\multi_camera_monitor\frontend-app` -> passed

Only report files and this handoff were added after that validation.
