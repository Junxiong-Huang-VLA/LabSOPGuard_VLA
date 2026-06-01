# Advanced Evidence Input Protocol

LabEmbodied accepts external model outputs as optional JSONL files under:

`LabSOPGuard/outputs/experiments/<experiment_id>/key_action_index/metadata/`

The frontend and review queue consume the fused evidence schema. Model vendors can change as long as these adapter files stay stable.

## Canonical Files

- `object_tracks.jsonl`: object trajectories, track IDs, motion metrics, bbox points.
- `panel_ocr.jsonl`: equipment panel OCR, readouts, button/knob/switch states.
- `liquid_state.jsonl`: liquid flow, meniscus, liquid level, mask, volume, stream direction.
- `container_state.jsonl`: open/closed, cap/lid, color, liquid level, container state changes.

Legacy aliases are still accepted, including `equipment_panel_states.jsonl`, `liquid_segmentation.jsonl`, and `container_state_events.jsonl`.

Validate the adapter inputs with:

```powershell
python -m key_action_indexer.cli validate-evidence-adapters --session-dir <key_action_index>
```

The validator writes `metadata/evidence_adapter_validation.json` with row counts, coverage windows, views, warning/error counts, schema issues, session-id mismatches, and out-of-session time windows. The quality gate consumes this report before a key-action session can be marked complete.

## Common Fields

Every row should include as many of these as possible:

- `observation_id`
- `session_id`
- `segment_id`
- `micro_segment_id`
- `start_sec`, `end_sec`
- `global_start_time`, `global_end_time`
- `view`
- `object_label`
- `confidence`
- `confirmation_level`: `candidate`, `confirmed`, or `measured`
- `evidence_reasons`
- `limitations`
- `asset_refs`

The adapter normalizes these rows into `model_observation_events.jsonl`, then the pipeline fuses them into `advanced_vision_evidence.jsonl` and video-understanding events.
