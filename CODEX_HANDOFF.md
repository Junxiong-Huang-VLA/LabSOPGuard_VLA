# Codex Handoff - LabCapability

Generated: 2026-05-08 18:26:35 Asia/Shanghai

## Read This First

This handoff freezes the current P4 key-action review and retrieval state for a new Codex conversation.

The latest user intent was to finish the manual-review closure, retrieval semantic diversity, homogeneous history samples, promotion safety review, and fixed final acceptance table. These are now implemented and validated. Do not restart from older P4 notes unless you need archaeology.

## Project Boundaries

- Root repo `D:\LabCapability` owns `src/key_action_indexer` and the key-action backend/report artifacts.
- Nested repo `D:\LabCapability\LabSOPGuard` owns only the integration page, upload entry, and key-action display surfaces.
- Keep `src/key_action_indexer` independent from the LabSOPGuard application.
- Mainline scope is physical-evidence extraction from long dual-view experiment videos, multimodal time alignment, text descriptions, vector indexing, and query.
- Do not reintroduce PTZ, cloud PTZ, five-camera orchestration, camera port mapping, MQTT tooling, wireless-video SDKs, `/api/v1/cameras`, or multi-monitor recording endpoints.
- PTZ lives separately at `D:\PtzTracker`.
- Multi-camera monitoring lives separately at `D:\MultiCameraMonitor`.
- YOLO bounding boxes are not the final deliverable; they must feed segment/micro-segment metadata and retrieval.
- Dry-run mode must remain runnable without real videos or ffmpeg.
- Preserve `pytest -q`, backend Python compilation, frontend `npm run build`, existing key-actions pages, and existing segment-level retrieval.

## Git State

Root repo:

- Path: `D:\LabCapability`
- Branch: `codex/handoff-review-history-freeze`
- Existing commits already made earlier:
  - `d50dd5f feat: add key action indexer backend baseline`
  - `0931bc8 docs: freeze p4 handoff and audit artifacts`
  - `a9d17b6 fix: expose nested reviewed clip traceability`
- Current working tree still has many uncommitted changes and untracked files. Do not blindly stage everything.

Nested LabSOPGuard repo:

- Path: `D:\LabCapability\LabSOPGuard`
- Branch: `feat/full-closed-loop-backbone`
- Ahead of origin by 2 commits:
  - `81db242 chore: remove ptz and multi-camera surfaces`
  - `e8434e2 feat: wire key action integration surfaces`
- Current nested working tree still has modified integration files:
  - `backend/main.py`
  - `frontend-app/src/api.ts`
  - `frontend-app/src/pages/KeyActionReviewQueue.tsx`

Commit advice:

- Keep root key-action indexer backend/report changes separate from LabSOPGuard UI integration changes.
- Keep promotion/gold benchmark safety changes separate from reviewed index/query semantics if possible.
- Do not mix PTZ/multi-camera cleanup, key-action page fixes, backend indexer capability, and report artifacts in one commit.

## Sessions In Scope

Main refreshed/audited sessions:

- `D:\LabCapability\LabSOPGuard\outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\key_action_index`
- `D:\LabCapability\LabSOPGuard\outputs\experiments\3ccd635c-217e-40dd-9922-0e1e397739ce\key_action_index`
- `D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index`

Additional complete key_action_index sources used for history:

- `D:\LabCapability\LabSOPGuard\outputs\experiments\acceptance_test_20260505\key_action_index`
- `D:\LabCapability\LabSOPGuard\outputs\experiments\solid-weighing-dual-view-20260508-153648\key_action_index`

History source inventory:

- `D:\LabCapability\reports\p4_history_source_inventory.json`
- `D:\LabCapability\reports\p4_history_model_key_action_index_only.json`
- `D:\LabCapability\reports\p4_history_model_key_action_index_only_summary.json`

History result:

- Complete key_action_index sources: `5`
- Legacy sources: `0`
- History process records: `5`
- Sources with process records: `5`
- Transition priors: `true`
- Note: distinct event `session_count` is `4` because `acceptance_test_20260505` reuses the 2190 session id internally, but `key_action_index_session_count` is `5`.

## What Was Completed

### 1. Manual Confirmation Closure

The two QA-blocked sessions had confirmation queues closed from `pending` to explicit, auditable states.

Important safety rule: Codex did not fabricate human approval. Remaining uncertain items were set to `needs_review` through non-human safety-gate files.

Generated/applied files:

- `D:\LabCapability\reports\p4_confirmation_decisions_3ccd635c-217e-40dd-9922-0e1e397739ce.codex_needs_more_review.json`
- `D:\LabCapability\reports\p4_confirmation_decisions_53ca6efe-a100-4e86-b041-7c98e2fcc662.codex_needs_more_review.json`
- `D:\LabCapability\reports\p4_confirmation_remaining_pending_3ccd635c-217e-40dd-9922-0e1e397739ce.codex_needs_more_review.json`
- `D:\LabCapability\reports\p4_confirmation_remaining_pending_53ca6efe-a100-4e86-b041-7c98e2fcc662.codex_needs_more_review.json`
- `D:\LabCapability\reports\p4_confirmation_remaining_pending_summary.json`

Current queue state:

- `3ccd...`: pending `0`, resolved `4`, all unresolved human decisions remain `needs_review`.
- `53ca...`: pending `0`, resolved `4`, all unresolved human decisions remain `needs_review`.

### 2. Review Packet / Decision Normalization

Decision files now use human-friendly allowed values:

- `approve`
- `reject`
- `needs_more_review`

The backend normalizes them internally to:

- `approved`
- `rejected`
- `needs_review`

Key files:

- `src/key_action_indexer/confirmation_loop.py`
- `src/key_action_indexer/review_packet.py`
- `tests/test_confirmation_loop.py`
- `tests/test_review_packet.py`

### 3. Retrieval Semantic Diversity

The previous 3ccd retrieval concentration problem was fixed.

Root causes addressed:

- Reviewed micro-window segment rows were losing meaningful micro `index_text` and retaining review-only text.
- Micro-window reviewed segments now preserve micro ids, micro time windows, micro global time, and rich micro text.
- Chinese aliases and rerank behavior now separate weighing, pipetting, sample handling, and recording/readout.

Current 3ccd query-to-result table:

- Markdown: `D:\LabCapability\reports\p4_query_to_result_table_3ccd635c-217e-40dd-9922-0e1e397739ce.md`
- JSON: `D:\LabCapability\reports\p4_query_to_result_table_3ccd635c-217e-40dd-9922-0e1e397739ce.json`
- Distinct top results: `4/8`
- Top-result distribution:
  - `seg_000001_micro_009`: 2 queries, weighing / Chinese weighing
  - `seg_000001_micro_010`: 2 queries, pipetting / Chinese pipetting
  - `seg_000001_micro_003`: 2 queries, sample handling / Chinese sample handling
  - `seg_000001_micro_008`: 2 queries, recording / Chinese recording

Key files:

- `src/key_action_indexer/reviewed_dataset.py`
- `src/key_action_indexer/semantic_alias.py`
- `src/key_action_indexer/session_audit.py`
- `tests/test_reviewed_dataset.py`
- `tests/test_semantic_alias.py`
- `tests/test_session_audit.py`

### 4. Batch Refresh / Final Audit

Batch refresh completed successfully for all three main sessions.

Main output:

- `D:\LabCapability\reports\p4_batch_refresh_summary_expanded.json`
- `D:\LabCapability\reports\p4_session_audit_summary_expanded.json`
- `D:\LabCapability\reports\p4_session_audit_summary_expanded.md`

Latest final acceptance table:

| Session | QA | Health | Candidate ratio | Strong | Retrieval-only | Segment backfill promoted | Chinese queries | Dual-view traceability | Query semantic diversity |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `2190fe06-3619-45fc-96ef-1bb8afb9bdf9` | `pass` | `pass` | 0.185185 | 2 | 6 | 0 | 4 | 8/8 | 0.5 |
| `3ccd635c-217e-40dd-9922-0e1e397739ce` | `needs_review` | `pass` | 0.10687 | 6 | 2 | 0 | 4 | 8/8 | 0.5 |
| `53ca6efe-a100-4e86-b041-7c98e2fcc662` | `needs_review` | `pass` | 0.25974 | 2 | 8 | 0 | 4 | 8/8 | 0.5 |

Current audit risks:

- `2190...`: none.
- `3ccd...`: `qa_not_pass`.
- `53ca...`: `qa_not_pass`.

`history_under_sampled` is no longer raised because the audit now looks at complete key_action_index source support and process-record/transition support instead of only raw event session count.

### 5. Promotion / Gold Benchmark Safety

Promotion safety was hardened.

Policy now enforced:

- Machine generation may propose candidate bindings.
- Machine generation must not write `human_verified=true`.
- `human_verified=true` requires an explicit human decision file passed to `confirm-gold-query-benchmark --decisions`.
- A promoted release is used as default only when `promotion_requirements.gold_benchmark_binding_mode == human_verified_review_file`.

Current promotion review:

- Markdown: `D:\LabCapability\reports\p4_promotion_safety_review.md`
- JSON: `D:\LabCapability\reports\p4_promotion_safety_review.json`
- Existing 3ccd promoted release: `v003`
- Existing promoted binding mode: `None`
- Active reviewed release after safety gate: `v006`
- Active reviewed index: `D:\LabCapability\LabSOPGuard\outputs\experiments\3ccd635c-217e-40dd-9922-0e1e397739ce\key_action_index\reviewed_releases\v006\reviewed_index`

Key files:

- `src/key_action_indexer/retrieval_eval.py`
- `src/key_action_indexer/cli.py`
- `src/key_action_indexer/reviewed_dataset.py`
- `tests/test_retrieval_eval.py`
- `tests/test_reviewed_dataset.py`

## Current Remaining Gaps

1. `3ccd...` and `53ca...` QA remain `needs_review`.
   - This is expected and correct.
   - The queue is not pending anymore, but true `approve/reject` still requires a real human reviewer.
   - Do not auto-promote these to complete.

2. Recording evidence remains retrieval/needs-review unless a human confirms the paper/readout clip.
   - Do not treat recording retrieval hits as strong process evidence automatically.

3. `acceptance_test_20260505` is a complete key_action_index source but reuses the 2190 session id internally.
   - History source quality is acceptable for now.
   - For stronger future reasoning, add more complete sessions with distinct real session ids.

4. Working tree is dirty.
   - There are many untracked docs/data files from earlier work.
   - Review `git status --short` before staging.
   - Avoid staging unrelated untracked files.

## Useful Commands

Use PowerShell from `D:\LabCapability`.

Batch refresh:

```powershell
python -m key_action_indexer.cli batch-refresh `
  --source "D:\LabCapability\LabSOPGuard\outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\key_action_index" `
  --source "D:\LabCapability\LabSOPGuard\outputs\experiments\3ccd635c-217e-40dd-9922-0e1e397739ce\key_action_index" `
  --source "D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index" `
  --query "balance weighing" `
  --query "pipetting liquid transfer" `
  --query "sample handling" `
  --query "recording balance readout" `
  --query "查找使用天平称量的片段" `
  --query "查找移液操作片段" `
  --query "查找样品处理片段" `
  --query "查找记录读数片段" `
  --output-summary "D:\LabCapability\reports\p4_batch_refresh_summary_expanded.json"
```

Audit:

```powershell
python -m key_action_indexer.cli audit-sessions `
  --source "D:\LabCapability\LabSOPGuard\outputs\experiments\2190fe06-3619-45fc-96ef-1bb8afb9bdf9\key_action_index" `
  --source "D:\LabCapability\LabSOPGuard\outputs\experiments\3ccd635c-217e-40dd-9922-0e1e397739ce\key_action_index" `
  --source "D:\LabCapability\LabSOPGuard\outputs\experiments\53ca6efe-a100-4e86-b041-7c98e2fcc662\key_action_index" `
  --query "balance weighing" `
  --query "pipetting liquid transfer" `
  --query "sample handling" `
  --query "recording balance readout" `
  --query "查找使用天平称量的片段" `
  --query "查找移液操作片段" `
  --query "查找样品处理片段" `
  --query "查找记录读数片段" `
  --output-json "D:\LabCapability\reports\p4_session_audit_summary_expanded.json" `
  --output-md "D:\LabCapability\reports\p4_session_audit_summary_expanded.md"
```

Freeze reviewed dataset:

```powershell
python -m key_action_indexer.cli freeze-reviewed-dataset --session-dir "D:\LabCapability\LabSOPGuard\outputs\experiments\3ccd635c-217e-40dd-9922-0e1e397739ce\key_action_index"
```

Gold benchmark safety:

```powershell
# This must fail without --decisions. That is intentional.
python -m key_action_indexer.cli confirm-gold-query-benchmark --session-dir "<session>" --decisions "<real-human-decision-file.json>"
```

Validation:

```powershell
python -m pytest -q
python -m compileall -q src LabSOPGuard\backend tests
powershell -ExecutionPolicy Bypass -File D:\LabCapability\scripts\check_project_scope.ps1
cd D:\LabCapability\LabSOPGuard\frontend-app
npm run build
```

## Last Validation

All latest validation passed:

- `python -m pytest -q` -> `210 passed`
- `python -m compileall -q src LabSOPGuard\backend tests` -> passed
- `powershell -ExecutionPolicy Bypass -File D:\LabCapability\scripts\check_project_scope.ps1` -> passed, no PTZ/camera/wireless/multi-monitor core code found
- `npm run build` in `D:\LabCapability\LabSOPGuard\frontend-app` -> passed

## Recommended Next Conversation Prompt

Paste this into the new conversation:

```text
Read D:\LabCapability\CODEX_HANDOFF.md and continue from the frozen P4 state.
Preserve the root/LabSOPGuard git boundary. Do not reintroduce PTZ or multi-camera scope.
First inspect git status in both repos, then help split the remaining changes into clean commits.
Do not fabricate human approvals: unresolved 3ccd and 53ca QA items must stay needs_review until real human decision files exist.
```
