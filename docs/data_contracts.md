# Data Contracts

All JSON/JSONL artifacts are versioned and should remain append-friendly. Downstream consumers must be able to rely on the fields below even when a source is missing or dry-run data is synthetic.

## `video_source`

Written to `metadata/video_sources.jsonl`.

Required fields: `schema_version`, `session_id`, `view_id`, `path`, `absolute_path`, `start_time`, `fps`, `offset_sec`, `exists`, `availability_status`, `is_primary`, `role`, `time_basis`.

## `input_event`

Used for `metadata/user_text_events.jsonl`, `metadata/ai_reply_events.jsonl`, and `metadata/upload_events.jsonl`.

Required fields: `schema_version`, `event_id`, `session_id`, `event_type`, `modality`, `source`, `source_path`, `source_row_index`, `global_time`, `session_time_sec`, `duration_sec`, `anchor_strategy`, `anchor_confidence`, `text`, `links`, `payload`.

Upload rows also include `upload_type`, `file_path`, `sha256`, `hash_status`, `uploaded_at`, `parsed_text`, and `thumbnail_path`.

AI rows also include `reply_type`, with values such as `ai_suggestion`, `ai_conclusion`, `ai_correction`, or `ai_reply`.

## `timeline_event`

Written to `metadata/unified_multimodal_timeline.jsonl`.

Required fields: `timeline_event_id`, `session_id`, `event_type`, `modality`, `source`, `global_time`, `session_time_sec`, `duration_sec`, `anchor_confidence`, `anchor_strategy`, `payload`, `links`, `text`.

## `key_action_segment`

Written to `metadata/key_action_segments.jsonl`.

Required fields: `session_id`, `segment_id`, `global_start_time`, `global_end_time`, `duration_sec`, `third_person`, `first_person`, `cv_detection`, `text_description`, `dialogue_context`, `index`, `interaction_keyframes`, `interaction_events`, `yolo_interactions`, `micro_segments`, `evidence`.

## `micro_segment`

Written to `metadata/micro_segments.jsonl`.

Required fields: `micro_segment_id`, `parent_segment_id`, `session_id`, `display_order`, `display_id`, `start_sec`, `end_sec`, `duration_sec`, `global_start_time`, `global_end_time`, `first_person`, `third_person`, `interaction`, `keyframes`, `dialogue_context`, `text_description`, `index`, `quality`, `evidence`.

## `model_observation_event`

Written to `metadata/model_observation_events.jsonl`.

Required fields: `observation_id`, `session_id`, `source_file`, `source_type`, `observation_type`, `event_type`, `confirmation_level`, `confidence`, `evidence_reasons`, `limitations`, `metrics`, `asset_refs`, `payload`.

## `video_understanding_event`

Written to `metadata/video_understanding.jsonl`.

Required fields: `video_event_id`, `session_id`, `segment_id`, `micro_segment_id`, `event_type`, `global_start_time`, `global_end_time`, `primary_object`, `action_type`, `confidence`, `confidence_reasons`, `anomaly_flags`, `asset_refs`, `payload`.

## `experiment_context`

Written to `metadata/experiment_context.json`.

Required fields: `session_id`, `purpose`, `procedure_candidates`, `materials`, `parameters`, `source_counts`, `text_evidence`, `upload_evidence`, `ai_evidence`, `transcript_evidence`, `database_evidence`, `video_evidence`, `fused_context`, `confidence`, `gaps`.

## `experiment_step`

Contained in `metadata/experiment_process.json`.

Required fields: `step_id`, `name`, `expected_action`, `status`, `observed`, `inferred`, `completed`, `confidence`, `confidence_reasons`, `evidence_refs`, `requires_human_confirmation`, `conflict_flags`, `reasoning`.

## `evidence_ref`

Reusable reference object embedded across artifacts.

Required fields: `evidence_id` or `id`, `source_type`, `source_id`, `path`, `global_time`, `session_time_sec`, `asset_type`, `evidence_level`, `confidence`, `text`.
