# YOLO Material Candidate Pipeline Contract

This contract protects the LabCapability key-material pipeline from regressions while the detector, semantic rules, VLM assist, and frontend review flow continue to evolve.

## Pipeline

Run material candidates through one path:

1. Load YOLO frame rows from `cv_outputs/yolo_frame_rows.jsonl`.
2. On real GPU runs, run `enable_micro_refine_rescan` inside detected key-action windows at the configured dense sample rate.
3. Normalize YOLO boxes to the render frame size before drawing.
4. Preserve `time_sec`, `frame_width`, `frame_height`, and `source_view` in micro-level YOLO evidence.
5. Filter implausible physical detections such as blue workbench-as-hand or workbench-as-paper.
6. Select hand-object interactions from YOLO evidence.
7. Apply semantic rules such as balance-weighing, but keep semantic context separate from annotation targets.
8. Build tracklets for the manipulated object and hands.
9. Render only the active hand-object instance for each candidate.
10. Backfill uncovered parent-segment interactions from YOLO rows even when the parent already has another micro-segment; mark these as `coverage_backfill` and keep them review-gated.
11. Reconnect stale event `micro_segment_id` values by `parent_segment_id + primary_object` before judging YOLO evidence.
12. Put all candidates behind the frontend review gate.
13. Mark only high-quality candidates as default recommendations.

## Non-Negotiable Rules

- `manipulated_object` is the annotation target.
- `instrument_context` is semantic context only. It must not be rendered as a target box.
- A candidate titled `hand + paper` must draw only the paper currently operated by the hand, not every visible paper object.
- A candidate titled `hand + reagent_bottle` must draw only the bottle currently operated by the hand, not every visible bottle.
- Tracklet rendering must match the current frame's hand-object interaction anchor before drawing.
- YOLO boxes from a source resolution such as `1920x1080` must be scaled before rendering onto clips such as `960x540`.
- Low-quality or weak-hand-evidence candidates may remain in the review queue, but must not be default recommended.
- Sparse-but-plausible hand-object contact is a `human_review` candidate, not `auto_ready`; weak edge/context detections remain `low_quality`.
- Label-level pseudo-tracks are not physical object movement. Object movement needs measured displacement plus stable identity/active evidence.
- Frontend/API ordering must show recommended or review candidates before `low_quality` diagnostics.
- GPU real runs must not silently fall back to CPU; dense micro refine is part of the default real-video quality path.
- Qwen/VLM may refine semantics only when the answer references YOLO evidence; it cannot invent unsupported object changes.

## One Command

Use this command to rebuild candidates through the guarded path:

```powershell
python -m key_action_indexer.cli material-candidates --session-dir "D:\LabCapability\LabSOPGuard\outputs\experiments\<experiment-id>\key_action_index" --rebuild-source --no-enable-vlm
```

The command writes the review queue under:

```text
<experiment>\_material_review_queue
```

## Regression Guard

Before accepting changes to this area, run:

```powershell
python -m pytest tests/test_material_candidate_pipeline_contract.py tests/test_material_references.py tests/test_yolo_detector.py tests/test_tracklet_annotations.py -q
```

For a broad check:

```powershell
python -m pytest -q
```

The dedicated contract test file covers:

- annotation target excludes instrument context;
- active interaction instance only;
- source-to-render bbox scaling;
- low-quality candidates never become default recommendations.
- parent segments with one valid micro still receive uncovered YOLO coverage backfill;
- stale event micro IDs reconnect to current micro evidence by parent segment and object;
- label-level pseudo-track movement is suppressed instead of surfaced as a physical action.
