# Key Action Annotation Guidelines

## Scope

This guideline covers physical-evidence extraction for long dual-view experiment videos in `src/key_action_indexer`. It focuses on YOLO-backed key action segments, hand-object interaction evidence, micro-segments, multiview clip alignment, metadata, retrieval, and evaluation. Bounding boxes are supporting evidence, not the final deliverable.

## Required Evaluation Coverage

Each evaluation set manifest should include:

- Real dual-view video: both `third_person` and `first_person` paths must point to real files.
- Dialogue: transcript JSONL with utterance IDs, local start/end seconds, and text.
- SOP: ordered procedure steps with `step_id`, `name`, `expected_action`, and optional entry/completion conditions.
- Human labels: parent segment labels and micro-segment labels.
- Expected output: target process timeline, retrieval expectations, and report-facing metrics.
- Time alignment anchors: hand-picked events visible across views or between video/dialogue/SOP.

## Parent Segment Labels

Use JSONL, one segment per line. Required fields:

- `segment_id`
- `session_id`
- `global_start_time`
- `global_end_time`
- `action_type`
- `primary_objects`
- `views`: include the views where the action is visible.
- `evidence_notes`: short human-readable notes.

Boundaries should cover the full meaningful action, not just the bounding-box contact. Prefer evidence-backed start/end points: first visible preparation, first contact, release, readout, or operator narration.

## Micro-Segment Labels

Use JSONL, one micro-segment per line. Required fields:

- `micro_segment_id`
- `parent_segment_id`
- `session_id`
- `global_start_time`
- `global_end_time`
- `interaction_type`
- `primary_object`
- `secondary_objects`
- `contact_phase`: `contact`, `peak`, `release`, or `full_micro_segment`.
- `evidence_views`
- `expected_keyframes`: contact, peak, and release frame references when available.

Micro-segments should capture hand-object interaction evidence and object state changes. Do not create trusted sample-adding labels without pipette, pipette tip, tube, liquid-transfer, or dialogue evidence.

## Time Alignment Anchors

Use JSONL, one anchor per line. Required fields:

- `anchor_id`
- `source`
- `expected_global_time`
- `predicted_global_time` when evaluating predictions inline, or provide a separate prediction file keyed by `anchor_id`.
- `required`: defaults to `true`.
- `evidence`: short note explaining why the anchor is reliable.

Good anchors include sharp events such as contact with a balance, pipette tip entering a tube, a display readout, a spoken timestamped instruction, or a SOP step transition visible in both views.

The time alignment evaluator reports `mae_sec`, `max_residual_sec`, `anchor_coverage_rate`, `drift_error_sec`, and `drift_error_per_min`.

## Version And Audit Trail

Historical process records and database write packages must preserve:

- `version`
- `source_session_id`
- `audit_trail`

Every writeback should append a new version instead of overwriting older process records. Audit entries should include timestamp, actor, action, source session, and relevant paths or notes.

## Expected Output

Expected output JSON should include at least:

- `source_session_id`
- expected parent segment count and micro-segment count.
- expected process step statuses.
- expected retrieval queries with acceptable result IDs or object/action constraints.
- expected time alignment metric thresholds.

Keep expected output strict enough to catch regressions but broad enough to allow equivalent evidence-backed segment boundaries.
