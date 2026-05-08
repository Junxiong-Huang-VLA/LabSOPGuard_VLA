# Handoff Freeze - 2026-05-08

This freeze preserves the current handoff artifacts and records the repository
boundary for the next key-action development pass.

## Frozen Artifacts

- `D:\LabCapability\CODEX_HANDOFF.md`
- `D:\LabCapability\reports\p4_batch_refresh_summary.json`
- `D:\LabCapability\reports\p4_session_audit_summary.json`
- `D:\LabCapability\reports\p4_session_audit_summary.md`

The files above are the baseline for the P4 session state. Future refreshes
should write new timestamped summaries instead of overwriting the meaning of
this handoff.

## Git Boundaries

- Root project `D:\LabCapability` owns the independent key-action indexer:
  - `src\key_action_indexer`
  - `tests`
  - `examples`
  - `docs`
  - `reports`
- Nested project `D:\LabCapability\LabSOPGuard` owns application integration only:
  - key-action pages
  - upload and experiment-entry surfaces
  - material/key-action display components
  - backend integration routes needed by those pages

## Out-of-Scope Guard

Do not reintroduce PTZ, camera proxy/streaming, wireless-video SDKs, five-camera
orchestration, MQTT tooling, or multi-monitor recording endpoints into
LabCapability. PTZ lives in `D:\PtzTracker`; multi-camera monitoring lives in
`D:\MultiCameraMonitor`.

Before merging key-action work, run:

```powershell
powershell -ExecutionPolicy Bypass -File D:\LabCapability\scripts\check_project_scope.ps1
python -m pytest -q
python -m compileall -q src LabSOPGuard\backend tests
```

For LabSOPGuard frontend integration, also run:

```powershell
npm run build
```

from `D:\LabCapability\LabSOPGuard\frontend-app`.

## Review Packet Outputs

The strengthened recovery review packets for the two QA-blocked sessions are:

- `D:\LabCapability\reports\p4_recovery_review_packet_3ccd635c-217e-40dd-9922-0e1e397739ce.md`
- `D:\LabCapability\reports\p4_recovery_review_packet_53ca6efe-a100-4e86-b041-7c98e2fcc662.md`

Decision templates were generated next to them. They intentionally contain
blank decisions and must be filled by a human reviewer before running
`confirmation-batch`.

## Expanded Refresh Artifacts

The history model has been rebuilt with six same-class sources:

- `2190fe06-3619-45fc-96ef-1bb8afb9bdf9`
- `3ccd635c-217e-40dd-9922-0e1e397739ce`
- `53ca6efe-a100-4e86-b041-7c98e2fcc662`
- `817133dd-7e2f-442c-a0a7-dd3be1957387`
- `cd35d322-1da3-4ae8-897c-7e405f5b5452`
- `d16e3f8d-63ca-4d12-9aaa-6ec3b602a557`

Canonical expanded outputs:

- `D:\LabCapability\reports\p4_history_model_6_source.json`
- `D:\LabCapability\reports\p4_batch_refresh_summary_expanded.json`
- `D:\LabCapability\reports\p4_session_audit_summary_expanded.json`
- `D:\LabCapability\reports\p4_session_audit_summary_expanded.md`

The expanded audit now includes fixed retrieval acceptance and backfill evidence
guard tables. It covers balance weighing, pipetting, sample handling, recording,
Chinese queries, and dual-view clip traceability.

Current expanded audit facts:

- `history_under_sampled` is cleared; all three current sessions read
  `history.session_count = 6`.
- Candidate ratio did not rebound; average remains `0.183932`, max remains
  `0.25974`.
- Segment-level retrieval backfill remains retrieval-only; promoted segment
  backfill count is `0`.
- Health gate is `pass` for all three current sessions. The
  `default_chinese_query_validation.json` artifact on `3ccd...` is explicitly
  marked `bootstrap_pending_human_verification`, so it remains a warning until
  the Chinese fixed-50 benchmark is manually verified.
- `3ccd635c-217e-40dd-9922-0e1e397739ce` and
  `53ca6efe-a100-4e86-b041-7c98e2fcc662` remain QA `needs_review` until a
  human reviewer applies the pending confirmation decisions.
