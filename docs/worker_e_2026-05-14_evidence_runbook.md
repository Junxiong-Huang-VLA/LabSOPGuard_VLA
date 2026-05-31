# Worker E Evidence Pipeline Runbook - 2026-05-14

## Scope

Worker E owns the documentation/eval handoff lane for the current LabCapability evidence-package pass. This runbook stays within `D:\LabCapability\docs\*` and the optional eval fixture space under `D:\LabCapability\tests\fixtures\*`. It does not modify `src\key_action_indexer`, LabSOPGuard backend/frontend code, PTZ code, camera monitor code, or any multi-camera monitor scope.

Project scope constraints still apply on 2026-05-14:

- LabCapability mainline is long dual-view experiment video evidence extraction, time alignment, physical evidence, vector indexing, retrieval, and evaluation.
- YOLO bounding boxes are candidate evidence only; the durable deliverable is segment/micro-segment metadata, `physical_change_log`, time alignment, retrieval records, and eval outputs.
- Dry-run mode must remain runnable without real video files or ffmpeg.
- PTZ belongs in `D:\PtzTracker`.
- Multi-camera and wireless video monitoring belong in `D:\MultiCameraMonitor`.

## Current Baseline

Baseline source experiment:

- Experiment ID: `db64d9d2-16ce-4de0-8fb3-597066531c31`
- Experiment root: `D:\LabCapability\LabSOPGuard\outputs\experiments\db64d9d2-16ce-4de0-8fb3-597066531c31`
- Key-action index root: `D:\LabCapability\LabSOPGuard\outputs\experiments\db64d9d2-16ce-4de0-8fb3-597066531c31\key_action_index`
- Evidence package root: `D:\LabCapability\LabSOPGuard\outputs\experiments\db64d9d2-16ce-4de0-8fb3-597066531c31\material_references`
- Evidence package manifest: `D:\LabCapability\LabSOPGuard\outputs\experiments\db64d9d2-16ce-4de0-8fb3-597066531c31\material_references\evidence_package_manifest.json`
- Frontend check URL from latest memory signal: `http://127.0.0.1:5173/experiments/db64d9d2-16ce-4de0-8fb3-597066531c31/materials`

Run status from `key_action_index\job_status.json`:

- Status: `completed`
- Message: `Key-action pipeline completed; quality gate passed and delivery package is ready`
- Queued: `2026-05-13T07:46:44.695374+00:00`
- Started: `2026-05-13T07:46:44.715622+00:00`
- Completed: `2026-05-13T08:54:46.825282+00:00`
- Updated: `2026-05-13T09:38:30.883240+00:00`
- Total elapsed: `351.196` seconds
- Core stage total from `pipeline_summary.json`: `155.756` seconds
- Stage split: validation `0.043`, detection `109.265`, micro segmentation `44.575`, vector index `1.873` seconds
- GPU runtime: CUDA available, `NVIDIA GeForce RTX 3060 Laptop GPU`, torch `2.6.0+cu124`, CUDA `12.4`
- VLM assist in this run: disabled; configured model was `qwen3.6-plus`

Evidence counts:

- Segments: `5`
- Micro-segments: `24`
- Raw micro-segments before dedup/merge: `26`
- Total action duration: `99.0` seconds
- YOLO frame rows: `2144`
- Reviewed dataset: `24` reviewed segments, `24` reviewed micro-segments, `985` reviewed evidence rows, `48` reviewed vectors
- Reviewed dataset delivery status: `delivery_ready=true`
- Reviewed release: `v001`

Material candidate/review state:

- Current review manifest at `_material_review_queue\manifest.json`: `71` candidates, `34` recommended, `26` approved, `28` pending, `8` rejected, `9` not selected, `43` processed
- Current `published_materials.json`: `26` accepted items
- Published canonical counts: `hand-paper=11`, `hand-bottle=9`, `hand-spatula=4`, `hand-balance=2`, `hand-container=0`
- Published asset kind counts: `14` keyframes, `12` clips
- Current evidence package files: `key_material_references.jsonl=26`, `physical_change_log.jsonl=26`, `time_alignment.json` present
- Current package `physical_change_log` event type: hand-object interaction evidence inferred from approved material references

Time alignment baseline:

- Package schema: `time_alignment.v1`
- Timezone: `Asia/Shanghai`
- Two video streams are present: `camera_00_6252967e502` and `camera_01_6100c5d8b02`
- Both streams currently have `offset_sec=0.0`, `clock_scale=1.0`, `alignment_confidence=0.7`, and `offset_source=explicit`
- Message alignment defaults: `default_window_before_sec=90.0`, `default_window_after_sec=180.0`, `fallback_to_segment_search=true`

Retrieval baseline from `material_retrieval_evaluation.json`:

- Schema: `material_retrieval_eval.v1`
- Generated: `2026-05-14T05:44:09.004888+00:00`
- Query set: `30` queries
- Top-1 hits: `23 / 30`, accuracy `0.7667`
- Top-3 hits: `23 / 30`, accuracy `0.7667`
- Expected canonical distribution: `hand-balance=6`, `hand-bottle=6`, `hand-container=6`, `hand-paper=6`, `hand-spatula=6`
- Misses: all six `hand-container` queries returned no class; one `hand-spatula` query (`spatula near weighing paper`) returned `hand-paper`

## Real Issues

| ID | Symptom | Evidence | Likely cause | Status | Next action |
| --- | --- | --- | --- | --- | --- |
| RI-001 | `job_status.json` material candidate totals are stale relative to current review/published artifacts. | `job_status.json` reports `approved_total=20`; `_material_review_queue\manifest.json` and `published_materials.json` report `26` approved/accepted on 2026-05-14. | Review/publish step updated material artifacts after the key-action job status snapshot. | Needs code owner decision. | Treat current material manifest/published files as the package baseline; add a later sync step so `job_status.json` reflects publish state. |
| RI-002 | `hand-container` retrieval is absent from the current formal package. | Retrieval eval has six `hand-container` expected queries and zero top-1/top-3 hits; published canonical counts have `hand-container=0`. | Current approved evidence lacks container-specific material references. | Needs data/review. | Add or approve container receiving samples only when YOLO/micro evidence supports it; do not synthesize from text alone. |
| RI-003 | One `hand-spatula` retrieval query is confused with `hand-paper`. | Query `spatula near weighing paper` returned `hand-paper`. | Query terms overlap with weighing paper evidence; reranking lacks enough object/action separation. | Needs eval expansion. | Add category-pair queries and inspect top bundles before changing retrieval scoring. |
| RI-004 | Current evidence-package eval query command can report broad positives for underspecified queries. | Exploratory runs on 2026-05-14 returned `correct` evidence for container/cleanup queries even when specific evidence is weak. | `evidence-package-query` retrieves related material and does not yet enforce negative/insufficient checks by target object. | Needs product/eval decision. | Keep negative queries exploratory until object-targeted insufficient checks are hardened. |
| RI-005 | Root experiment `artifacts\key_material_references.jsonl` and `artifacts\physical_change_log.jsonl` are zero-line files while `material_references\*` has the current package records. | `artifacts\*.jsonl` line counts are `0`; `material_references\key_material_references.jsonl=26` and `physical_change_log.jsonl=26`. | Package publish path is authoritative; root artifacts are not refreshed or are placeholders. | Needs documentation/code owner decision. | Consumers should load `material_references\evidence_package_manifest.json`; later decide whether root artifacts should mirror or be removed from docs. |
| RI-006 | VLM was not part of the current baseline run. | `key_action_vlm_assist.enabled=false`, reason `disabled`, model `qwen3.6-plus`. | Environment flag disabled during run. | Needs rerun if VLM semantics are required. | Follow the VLM rerun procedure below and compare baseline deltas instead of overwriting conclusions. |
| RI-007 | Historical rerun stderr references PTZ/camera proxy registration. | `D:\LabCapability\output\key_action_rerun_3ccd_swapped.stderr.log` logs `/api/v1/cameras` and `/api/v1/ptz-tracker`. | LabSOPGuard backend registers routes at import time. | Out of scope for Worker E. | Do not expand this in LabCapability; only record it as a scope guard warning. |

## T0-T12 Completion Criteria

| Task | Completion criteria |
| --- | --- |
| T0 Baseline capture | Current baseline numbers are recorded with source paths, absolute dates, run ID/status, material counts, retrieval metrics, and known stale/contradictory artifacts. |
| T1 Scope guard | Documentation explicitly preserves `AGENTS.md` boundaries and excludes PTZ, cloud PTZ, five-camera orchestration, `/api/v1/cameras`, wireless SDKs, and multi-monitor recording as LabCapability core work. |
| T2 Evidence package source of truth | Consumers are directed to `material_references\evidence_package_manifest.json`, not to zero-line root artifact placeholders. |
| T3 YOLO candidate layer | Candidate counts, recommended/pending/approved/rejected/not-selected totals, and the policy that YOLO is candidate evidence only are recorded. |
| T4 Audit/package layer | Approved material references, physical change records, reviewed dataset counts, and delivery-ready state are recorded. |
| T5 Time alignment | Dual-view stream IDs, offsets, clock scale, confidence, timezone, and message-window policy are recorded. |
| T6 Eval query set | A schema-compatible query fixture or documented plan exists, with positive scored checks separated from exploratory gap-finding queries. |
| T7 Retrieval baseline | Top-1/top-3 accuracy, query count, canonical distribution, and concrete misses are recorded. |
| T8 VLM rerun procedure | A non-destructive PowerShell procedure exists for rerunning with VLM enabled and collecting logs/outputs for comparison. |
| T9 Review feedback dataset plan | Review decisions are mapped to durable fields: query ID, decision, expected material/action, reviewer, notes, evidence path, and replay target. |
| T10 Validation checklist | Commands are listed for JSON fixture parsing, evidence-package eval, strict package validation, Python compile, pytest, frontend build, and dry-run preservation. |
| T11 Real issue list | Each issue has symptom, evidence, likely cause, status, and one next action. |
| T12 Handoff/task report | Worker E changed files and validation status can be reported without claiming other workers' source-module changes. |

## VLM Rerun Procedure

Use this only after confirming no other worker is actively using the target experiment output directory. Prefer creating a new experiment/run or preserving logs under `D:\LabCapability\output\` instead of overwriting baseline evidence.

PowerShell preflight:

```powershell
Set-Location -LiteralPath 'D:\LabCapability'
git status --short
Get-Content -LiteralPath 'D:\LabCapability\AGENTS.md'
Get-Content -LiteralPath 'D:\LabCapability\LabSOPGuard\outputs\experiments\db64d9d2-16ce-4de0-8fb3-597066531c31\key_action_index\job_status.json' -Raw
```

Environment for GPU/VLM rerun:

```powershell
Set-Location -LiteralPath 'D:\LabCapability'
$env:PYTHONPATH = 'D:\LabCapability\src'
$env:KEY_ACTION_ENABLE_VLM_ASSIST = '1'
$env:KEY_ACTION_VLM_MODEL = 'qwen3.6-plus'
$env:KEY_ACTION_YOLO_DEVICE = '0'
& 'C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe' -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Historical rerun template:

```powershell
Set-Location -LiteralPath 'D:\LabCapability'
$env:KEY_ACTION_ENABLE_VLM_ASSIST = '1'
$env:KEY_ACTION_VLM_MODEL = 'qwen3.6-plus'
$env:KEY_ACTION_YOLO_DEVICE = '0'
& 'C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe' 'D:\LabCapability\output\scripts\rerun_3ccd_swapped_views.py' *> 'D:\LabCapability\output\key_action_vlm_rerun_2026-05-14.log'
```

Important: `D:\LabCapability\output\scripts\rerun_3ccd_swapped_views.py` targets historical experiment `3ccd635c-217e-40dd-9922-0e1e397739ce`. Do not use it as-is for `db64d9d2-16ce-4de0-8fb3-597066531c31`. For the current experiment, clone/adapt the experiment ID and raw video paths in a temporary local rerun helper or launch through the existing LabSOPGuard task path, then record the new run ID and compare against the baseline above.

Post-rerun comparisons:

```powershell
Set-Location -LiteralPath 'D:\LabCapability'
$env:PYTHONPATH = 'D:\LabCapability\src'
& 'C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe' -m key_action_indexer.cli evidence-package-validate `
  --package-root 'D:\LabCapability\LabSOPGuard\outputs\experiments\db64d9d2-16ce-4de0-8fb3-597066531c31\material_references' `
  --strict

& 'C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe' -m key_action_indexer.cli evidence-package-eval `
  --package-root 'D:\LabCapability\LabSOPGuard\outputs\experiments\db64d9d2-16ce-4de0-8fb3-597066531c31\material_references' `
  --queries 'D:\LabCapability\tests\fixtures\evidence_package_eval_queries_2026_05_14.json' `
  --limit 5
```

Record deltas for:

- `key_action_vlm_assist.enabled`, model, reviewed group count, and failure reason.
- Candidate counts and canonical distribution.
- `key_material_references.jsonl` and `physical_change_log.jsonl` line counts.
- Retrieval and evidence-package eval rates.
- New false positives/false negatives in the real issue list.

## Review Feedback Dataset Plan

Goal: make reviewer feedback replayable, auditable, and useful for retrieval/eval without turning it into hidden labels.

Dataset location proposal:

- Draft decisions: `D:\LabCapability\LabSOPGuard\outputs\experiments\<experiment_id>\key_action_index\metadata\gold_query_decisions.json`
- Confirmed benchmark: `D:\LabCapability\LabSOPGuard\outputs\experiments\<experiment_id>\key_action_index\metadata\gold_query_benchmark.json`
- Worker E fixtures: `D:\LabCapability\tests\fixtures\*.json`

Minimum decision fields:

- `query_id`
- `decision`: `approved`, `rejected`, `not_applicable`, or `out_of_scope`
- `reviewer`
- `reviewed_at`
- `expected_canonical_action_type`
- `expected_material_ids` or `expected_micro_segment_ids`
- `evidence_package_root`
- `evidence_paths`
- `note`

Review process:

1. Bootstrap exploratory queries from current known misses: all `hand-container` gaps, the `hand-spatula` vs `hand-paper` confusion, and target-object insufficient checks.
2. Run query/eval output and save top bundles.
3. Reviewer marks each query as approved, rejected, not applicable, or out of scope.
4. Only approved/not-applicable decisions enter hard metrics; rejected exploratory rows stay in the issue backlog.
5. Re-run default retrieval eval and evidence-package eval after every query-set update.

## Evidence-Package-Eval Query Set Plan

The fixture `D:\LabCapability\tests\fixtures\evidence_package_eval_queries_2026_05_14.json` is a schema-compatible starter set for `evidence-package-eval`. It has two lanes:

- Scored positive smoke queries with `expected_intent`, `expected_status`, `expected_label`, and `expected_event_type`.
- Exploratory gap queries without expected fields, so they execute and preserve actual outputs without polluting pass/fail metrics before object-targeted insufficient checks are hardened.

Expansion plan:

1. Grow from this starter set to `30` evidence-package queries to match the current material retrieval eval size.
2. Keep balanced coverage for `hand-balance`, `hand-paper`, `hand-bottle`, `hand-spatula`, `hand-container`, and negative/insufficient targets.
3. Add material ID checks only after the package has stable IDs or a reviewer confirms the top material IDs.
4. Keep `hand-container` as exploratory until approved evidence exists.
5. Add paired ambiguity queries such as `spatula near weighing paper` and `weighing paper without spatula`.
6. Store the eval output under the run's `reports\` directory or a temporary output path; do not write back to the package root.

## Validation Checklist

Use these checks for Worker E changes and handoff validation:

```powershell
Set-Location -LiteralPath 'D:\LabCapability'
python -m json.tool 'D:\LabCapability\tests\fixtures\evidence_package_eval_queries_2026_05_14.json' | Out-Null
```

```powershell
Set-Location -LiteralPath 'D:\LabCapability'
$env:PYTHONPATH = 'D:\LabCapability\src'
& 'C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe' -m key_action_indexer.cli evidence-package-eval `
  --package-root 'D:\LabCapability\LabSOPGuard\outputs\experiments\db64d9d2-16ce-4de0-8fb3-597066531c31\material_references' `
  --queries 'D:\LabCapability\tests\fixtures\evidence_package_eval_queries_2026_05_14.json' `
  --limit 5
```

Broader validation expected before claiming a full project pass:

```powershell
Set-Location -LiteralPath 'D:\LabCapability'
pytest -q
python -m compileall src
```

```powershell
Set-Location -LiteralPath 'D:\LabCapability\LabSOPGuard\frontend-app'
npm run build
```

```powershell
Set-Location -LiteralPath 'D:\LabCapability'
python -m key_action_indexer.cli run --manifest '.runtime\timeline_demo_manifest.json' --dry-run
```

Because the current worktree has many source and test edits from other workers on 2026-05-14, Worker E should report which checks were run and whether failures are attributable to this docs/fixture lane or to ambient code changes.

## Worker E Task Report

Changed files expected for this lane:

- `D:\LabCapability\docs\worker_e_2026-05-14_evidence_runbook.md`
- `D:\LabCapability\tests\fixtures\evidence_package_eval_queries_2026_05_14.json`

Validation performed on 2026-05-14:

- `python -m json.tool D:\LabCapability\tests\fixtures\evidence_package_eval_queries_2026_05_14.json`: passed.
- `evidence-package-eval` with `PYTHONPATH=D:\LabCapability\src`, current package root, and the new fixture: passed; `query_count=8`, `insufficient_count=1`, scored `intent/status/label/event_type` rates all `1.0`.
- `evidence-package-validate --strict` against the current package root: passed; `references=26`, `physical_changes=26`, `errors=0`, `warnings=0`.
- `pytest -q`: passed, `300 passed in 235.16s`.
- `python -m compileall src`: passed.
- `npm run build` in `D:\LabCapability\LabSOPGuard\frontend-app`: passed.
- Dry-run pipeline command was not run from this docs/fixture lane because the documented `.runtime\timeline_demo_manifest.json` path is absent and a dry-run would generate files outside Worker E's write scope.

Completion report template:

```markdown
## Summary
- Worker E documented the 2026-05-14 baseline, real issue list, T0-T12 completion criteria, VLM rerun steps, review feedback plan, eval query set plan/fixture, and validation checklist.

## Evidence Pipeline Impact
- Candidate generation:
- Candidate audit:
- Evidence package:
- physical_change_log:
- time_alignment:
- Retrieval/eval:

## Validation
- Passed:
- Not run:
- Blocked by:

## Real Issues
- [status] symptom - evidence - next action

## Scope Notes
- No PTZ, multi-camera monitor, wireless-video, or complex frontend scope was added.
```
