export type ExperimentStatus = 'created' | 'uploaded' | 'queued' | 'running' | 'completed' | 'failed' | string

export interface ExperimentTaskStatus {
  task_id: string | null
  experiment_id: string
  status: ExperimentStatus
  current_stage: string
  progress: number
  video_path?: string | null
  error_type?: string | null
  error_message?: string | null
  started_at?: string | null
  completed_at?: string | null
  output_paths: Record<string, string>
}

export interface SegmentPreviewCanonicalFields {
  third_view_realtime_preview?: string | null
  third_view_realtime_preview_url?: string | null
  third_view_realtime_preview_path?: string | null
  third_view_preview_url?: string | null
  third_preview_url?: string | null
  first_view_realtime_preview?: string | null
  first_view_realtime_preview_url?: string | null
  first_view_realtime_preview_path?: string | null
  first_view_preview_url?: string | null
  first_preview_url?: string | null
  side_by_side_realtime_preview?: string | null
  side_by_side_realtime_preview_url?: string | null
  side_by_side_realtime_preview_path?: string | null
  side_by_side_preview_url?: string | null
  fast_preview?: string | null
  fast_preview_url?: string | null
  fast_preview_path?: string | null
  raw_window_start_global_timestamp_us?: string | number | null
  raw_window_end_global_timestamp_us?: string | number | null
  actual_experiment_start_global_timestamp_us?: string | number | null
  actual_experiment_end_global_timestamp_us?: string | number | null
  focus_preview_start_global_timestamp_us?: string | number | null
  focus_preview_end_global_timestamp_us?: string | number | null
  actual_experiment_start?: string | number | boolean | null
  actual_experiment_duration_s?: number | null
}

export interface Experiment {
  experiment_id: string
  title: string
  description: string
  status: ExperimentStatus
  created_at: string
  updated_at?: string | null
  started_at?: string | null
  completed_at?: string | null
  analyzed_at?: string | null
  archived_at?: string | null
  video_asset_id?: string | null
  analysis_job_id?: string | null
  video_paths?: string[]
  output_paths?: Record<string, string>
  total_steps: number
  inferred_steps: number
  avg_confidence: number | null
  evidence_count: number
  processing_stage: string
  processing_error?: string | null
  models_used: string[]
  protocol_text?: string | null
  task?: ExperimentTaskStatus
  key_action_summary?: {
    status?: string
    progress?: number
    message?: string | null
    segment_count?: number
    micro_segment_count?: number
    interaction_count?: number
    raw_yolo_interaction_count?: number
    vector_count?: number
    source?: string | null
  }
}

export interface StepRecord {
  step_id: string
  experiment_id: string
  step_index: number
  step_name: string
  step_description: string
  status: 'confirmed' | 'candidate' | 'inferred' | 'skipped' | 'needs_review' | string
  start_time_sec: number
  end_time_sec?: number
  duration_sec?: number
  confidence: number
  step_confidence?: 'high' | 'medium' | 'low' | string
  completed_by_inference?: boolean
  inference_method?: string
  inference_model?: string
  evidence_refs: EvidenceRef[]
  parameters?: StepParameter[]
  notes?: string
  evidence_notes?: string
  provenance?: StepProvenance
  created_at?: string
  updated_at?: string
  [key: string]: unknown
}

export interface StepProvenance {
  source?: string
  is_inferred?: boolean
  confidence?: number
  inference_method?: string
  model_used?: string
  timestamp?: string
  [key: string]: unknown
}

export interface EvidenceRef {
  evidence_id: string
  evidence_type: string
  source: string
  frame_id?: number
  timestamp_sec?: number
  confidence: number
  description?: string
  provenance?: StepProvenance
  [key: string]: unknown
}

export interface StepParameter {
  name: string
  value: unknown
  unit?: string
  source?: string
}

export interface UpdateStepRequest {
  step_name?: string
  step_description?: string
  status?: StepRecord['status']
  start_time_sec?: number
  end_time_sec?: number
  confidence?: number
  notes?: string
}

export interface StructuredExperimentResult {
  analysis?: unknown
  [key: string]: unknown
}

export interface BenchmarkMetricSummary {
  sample_count?: number
  p50_sec?: number | null
  p90_sec?: number | null
  best_sec?: number | null
  latest_sec?: number | null
  mean_sec?: number | null
  max_sec?: number | null
}

export interface TimingStageSummary {
  stage: string
  stage_label_zh?: string
  duration_sec?: number | null
  inputs?: number | null
  outputs?: number | null
  errors?: number | null
  workers?: number | null
}

export interface DisplayTimingStageSummary {
  stage: string
  label_zh: string
  duration_sec?: number | null
  available?: boolean
  missing_reason?: string | null
  missing_reason_zh?: string | null
  history_sample_count?: number
  p50_sec?: number | null
  p90_sec?: number | null
}

export interface SamplingTransparency {
  schema_version?: string
  available?: boolean
  coarse_sampled_frame_count?: number | null
  fine_sampled_frame_count?: number | null
  coarse_wall_sec?: number | null
  fine_wall_sec?: number | null
  gpu_device?: string | null
  gpu_devices?: string[]
  requested_gpu_devices?: string[]
  batch_size?: number | null
  batch_sizes?: number[]
  source?: string
}

export interface TimingHistorySummary {
  schema_version?: string
  stage_order?: string[]
  history_path?: string
  history_file_exists?: boolean
  history_file_rows?: number
  upload_e2e_rows?: number
  current_included?: boolean
  sample_count?: number
  stages?: Record<string, {
    sample_count?: number
    p50_sec?: number | null
    p90_sec?: number | null
    latest_sec?: number | null
  }>
}

export interface UploadE2ETargetStatus {
  metric?: string
  target_p50_sec?: number | null
  target_p90_sec?: number | null
  actual_p50_sec?: number | null
  actual_p90_sec?: number | null
  p50_ok?: boolean
  p90_ok?: boolean
  passed?: boolean
}

export interface UploadE2ERunSummary {
  experiment_id?: string
  title?: string
  status?: string
  client_end_to_end_sec?: number | null
  client_create_sec?: number | null
  client_upload_http_sec?: number | null
  client_raw_upload_sec?: number | null
  client_run_request_sec?: number | null
  client_queue_start_sec?: number | null
  client_analysis_poll_wait_sec?: number | null
  client_result_fetch_sec?: number | null
  client_sub_experiments_fetch_sec?: number | null
  client_materials_fetch_sec?: number | null
  client_analysis_wait_sec?: number | null
  upload_mode?: string | null
  segment_count?: number | null
  segment_video_count?: number | null
  dual_view_segment_count?: number | null
  published_material_count?: number | null
  ready_for_demo?: boolean
  backend_timing?: {
    server_end_to_end_sec?: number | null
    upload_save_sec?: number | null
    queue_wait_sec?: number | null
    algorithm_elapsed_sec?: number | null
    core_analysis_sec?: number | null
    stage_rows?: TimingStageSummary[]
  }
}

export interface UploadE2EBenchmarkSummary {
  schema_version?: string
  status?: string
  source_path?: string
  generated_at?: string | null
  repeat_count?: number
  current_repeat_count?: number
  successful_ready_count?: number
  current_successful_ready_count?: number
  history_count?: number
  history_ready_count?: number
  metric_scope?: string
  metric_scope_zh?: string
  metric_labels_zh?: Record<string, string>
  metrics?: Record<string, BenchmarkMetricSummary>
  target_status?: UploadE2ETargetStatus
  pipeline_contract?: {
    schema_version?: string
    quality_gates?: Record<string, unknown>
    steps?: Array<Record<string, unknown>>
  }
  latest_run?: UploadE2ERunSummary | null
  current_runs?: UploadE2ERunSummary[]
  run_history?: UploadE2ERunSummary[]
  runs?: UploadE2ERunSummary[]
}

export interface TimeAxisHealth {
  schema_version?: string
  status?: string
  time_axis_unreliable?: boolean
  can_publish_formal_materials?: boolean
  can_write_video_memory?: boolean
  source_path?: string | null
  reasons?: string[]
  raw?: Record<string, unknown>
}

export interface ActionAlignmentHealth {
  status?: string
  reason?: string | null
  reasons?: string[]
  dual_view_action_event_count?: number
  unmatched_view_evidence_count?: number
  aligned_episode_count?: number
  formal_results_allowed?: boolean
  decision?: string | null
  source_path?: string | null
  [key: string]: unknown
}

export interface FormalPublishHealth {
  schema_version?: string
  status?: string
  reason?: string | null
  reasons?: string[]
  can_publish_formal_segments?: boolean
  can_publish_formal_materials?: boolean
  can_write_video_memory?: boolean
  time_axis?: TimeAxisHealth
  action_alignment?: ActionAlignmentHealth
  official_segment_count?: number
  raw_official_segment_count?: number
  candidate_segment_count?: number
  dual_view_action_event_count?: number
  unmatched_view_evidence_count?: number
  user_message_zh?: string
  [key: string]: unknown
}

export interface AnalysisOverview {
  schema_version: string
  experiment: {
    experiment_id: string
    experiment_name: string
    description?: string
  }
  run: {
    run_id: string
    result_version: string
    status: ExperimentStatus
    stage: string
    progress: number
    message?: string
    elapsed_sec?: number | null
    core_analysis_sec?: number | null
    timing?: {
      source?: string
      elapsed_sec?: number | null
      core_analysis_sec?: number | null
      algorithm_elapsed_sec?: number | null
      server_end_to_end_sec?: number | null
      upload_save_sec?: number | null
      queue_wait_sec?: number | null
      stage_count?: number | null
      performance_benchmark?: {
        schema_version?: string
        sample_count?: number
        metric_labels_zh?: Record<string, string>
        current?: Record<string, number | null | undefined>
        metrics?: Record<string, BenchmarkMetricSummary>
      }
      upload_e2e_benchmark?: UploadE2EBenchmarkSummary
      stages?: TimingStageSummary[]
      stage_rows?: TimingStageSummary[]
      stage_labels_zh?: Record<string, string>
      display_stages?: DisplayTimingStageSummary[]
      display_stage_order?: string[]
      display_stage_count?: number
      display_stage_labels_zh?: Record<string, string>
      timing_buckets?: Record<string, number | null>
      timing_bucket_labels_zh?: Record<string, string>
      timing_history_summary?: TimingHistorySummary
      timing_history_path?: string
      sampling_transparency?: SamplingTransparency
      coarse_sampled_frame_count?: number | null
      fine_sampled_frame_count?: number | null
      coarse_wall_sec?: number | null
      fine_wall_sec?: number | null
      gpu_device?: string | null
      batch_size?: number | null
      request_received_at?: string | null
      upload_completed_at?: string | null
      queued_at?: string | null
      started_at?: string | null
      completed_at?: string | null
    }
    updated_at?: string | null
    trace_id?: string
  }
  time_axis?: TimeAxisHealth
  formal_publish_health?: FormalPublishHealth | null
  action_alignment_health?: ActionAlignmentHealth | null
  readiness: {
    summary_ready: boolean
    steps_ready: boolean
    alerts_ready: boolean
    artifacts_ready: boolean
    annotated_video_ready: boolean
    writeback_ready: boolean
  }
  summary: {
    frame_count: number
    detection_count: number
    alert_count: number
    official_step_count: number
    candidate_step_count: number
    confirmed_step_count: number
    inferred_step_count: number
    avg_confidence: number | null
    model_name: string
  }
  steps: {
    official: StepRecord[]
    candidate: StepRecord[]
    inferred: StepRecord[]
  }
  scene_summary: {
    description?: string
    activities: string[]
    objects: string[]
    visible_lab_objects?: string[]
    uncertain_objects?: string[]
    background_objects?: string[]
    step_indicators: string[]
    ppe_assessment: Record<string, unknown>
    alerts: Array<Record<string, unknown>> | string[]
    detections: Array<Record<string, unknown>>
    evidence_source?: string
    evidence_note?: string
    raw?: unknown
  }
  alerts: Array<{
    alert_id: string
    rule_name: string
    rule_id?: string
    severity: string
    source_frame?: number
    timestamp_sec?: number
    camera_id?: string | null
    event_id?: string | null
    evidence_refs: Array<Record<string, unknown>>
    rule_basis: string
    related_objects: string[]
    confidence?: number
    message: string
  }>
  markers: {
    steps: Array<{ id: string; label: string; timestamp_sec: number; kind: string }>
    alerts: Array<{ id: string; label: string; timestamp_sec: number; kind: string; severity?: string }>
    evidence: Array<{ id: string; timestamp_sec: number; kind: string }>
  }
  artifacts: Record<string, {
    name: string
    ready: boolean
    kind?: string | null
    size_bytes: number
    updated_at?: string | null
    url?: string | null
    time_start_sec?: number | null
    true_start_sec?: number | null
    time_end_sec?: number | null
    duration_sec?: number | null
    focus_source?: string | null
    focus_anchor?: Record<string, unknown> | null
  }>
  debug: Record<string, unknown>
}

export interface VideoMemoryTaskStatus {
  task_id: string
  label: string
  status: string
  required_symbols?: string[]
  data_ready?: boolean
}

export interface VideoMemoryEvidenceView {
  view_id: string
  source_video_id?: string | null
  local_start_sec?: number | null
  local_end_sec?: number | null
  clip_uri?: string | null
  keyframe_uri?: string | null
  clip_url?: string | null
  keyframe_url?: string | null
  frame_ids?: string[]
  yolo_refs?: Array<Record<string, unknown>>
  evidence_role?: string | null
  confidence?: number | null
}

export interface VideoMemoryEvidenceItem {
  schema_version: string
  evidence_id: string
  cluster_id: string
  session_id: string
  experiment_id: string
  experiment_title?: string
  created_at: string
  observed_at: string
  time_range: {
    global_start_time?: string
    global_end_time?: string
    start_sec?: number
    end_sec?: number
  }
  action: {
    type?: string
    primary_object?: string
    secondary_objects?: string[]
    description?: string
    interaction_type?: string
  }
  physical_evidence?: Record<string, unknown>
  views: VideoMemoryEvidenceView[]
  vlm?: {
    cache_key?: string
    model?: string
    prompt_version?: string
    summary?: string
    confidence?: number
    status?: string
  }
  retrieval?: {
    index_text?: string
    embedding_id?: string | null
    score?: number
    index_level?: string
  }
  trace?: Record<string, unknown>
  retention?: Record<string, unknown>
  ui?: {
    title?: string
    clip_url?: string | null
    keyframe_url?: string | null
    view_count?: number
  }
}

export interface VideoMemoryCluster {
  schema_version?: string
  cluster_id: string
  lifecycle_state: string
  score: number
  score_reasons?: string[]
  evidence_item_ids?: string[]
  last_observed_at?: string
  view_ids?: string[]
  evidence_count?: number
  archive_reason?: string
}

export interface VideoMemoryExperimentSummary {
  experiment_id: string
  title?: string
  status?: string
  created_at?: string
  latest_observed_at?: string | null
  source_level?: string
  counts?: Record<string, number | string | null | undefined>
  alignment?: Record<string, unknown>
  paths?: Record<string, string | null>
}

export interface VideoMemoryOverview {
  schema_version: string
  contract_version: string
  generated_at: string
  scope: {
    retention_days: number
    max_items: number
    source?: string
  }
  core: {
    available: boolean
    module: string
    status: string
    missing_symbols?: string[]
    implemented_symbols?: string[]
    error?: string
  }
  counts: Record<string, number | string | null | undefined>
  coverage: {
    is_partial: boolean
    retention_days: number
    start_time?: string
    end_time?: string
    latest_observed_at?: string | null
    truncation_reason?: string | null
    skipped_items?: number
    window_days_available?: number
    window_days_expected?: number
    window_completeness?: string
    window_completeness_ratio?: string
  }
  tasks: VideoMemoryTaskStatus[]
  snapshot?: Record<string, unknown>
  experiments: VideoMemoryExperimentSummary[]
  evidence_items: VideoMemoryEvidenceItem[]
  clusters: VideoMemoryCluster[]
  limitations?: string[]
  endpoints?: Record<string, string>
}

export interface VideoMemoryQueryAnswer {
  answer_id?: string
  schema_version?: string
  query?: string
  generated_at?: string
  snapshot_id?: string | null
  answer_summary?: string
  answer?: {
    status?: string
    text?: string
  }
  claims?: Array<Record<string, unknown>>
  unresolved_questions?: string[]
  confidence?: number
  human_confirmation_status?: string
  partial_window?: boolean
  is_full_30_day_memory?: boolean
  partial_window_notice?: string
  partial_window_notice_zh?: string
  window_days_available?: number
  window_days_expected?: number
  window_completeness?: string
  window_scope?: {
    window_days_available?: number
    window_days_expected?: number
    window_completeness?: string
    window_completeness_ratio?: string
    partial_window_reason?: string
    partial_window_notice?: string
    partial_window_notice_zh?: string
    [key: string]: unknown
  }
  evidence_trace?: {
    query?: string
    retrieved_evidence?: Array<Record<string, unknown>>
    claims?: Array<Record<string, unknown>>
    limitations?: string[]
    [key: string]: unknown
  }
  [key: string]: unknown
}

export interface CreateExperimentRequest {
  title: string
  description?: string
  context_text?: string
  protocol_text?: string
}

export interface MaterialSearchItem {
  item_id: string
  experiment_id: string
  timestamp_sec: number
  local_timestamp_sec?: number
  view?: string | null
  camera_view?: string | null
  camera_id?: string | null
  stream_id?: string | null
  frame_path?: string | null
  clip_id?: string | null
  clip_file_path?: string | null
  clip_exists?: number
  object_labels?: string[]
  actions?: string[]
  event_types?: string[]
  payload?: Record<string, unknown>
  preview_url?: string | null
  clip_url?: string | null
  report_url?: string | null
  material_url?: string | null
  published_paths?: Record<string, string>
  material_group_id?: string
  physical_action_material_id?: string
  asset_items?: MaterialSearchItem[]
  assets?: Record<string, Record<string, MaterialSearchItem | undefined>>
  asset_kind?: string
  material_type?: string
  event_id?: string
  display_title?: string | null
  display_name?: string
  event_type?: string
  semantic_action?: string | null
  manipulated_object?: string | null
  instrument_context?: string | null
  corrected_primary_object?: string | null
  primary_object?: string | null
  raw_primary_object?: string | null
  yolo_primary_object?: string | null
  canonical_action_type?: string
  canonical_object?: string
  sop_phase?: string
  interaction_family?: string
  window_id?: string | null
  segment_id?: string | null
  experiment_window_id?: string | null
  source_window_sync_index?: string | null
  window_preview?: string | null
  sample_grid?: string | null
  review_status?: string
  official_status?: string
  keyframe_quality_score?: number | null
  cli_ready_folder?: string | null
  evidence_bundle_id?: string | null
  orphan_material?: boolean
  first_keyframe?: string | null
  third_keyframe?: string | null
  first_keyclip?: string | null
  third_keyclip?: string | null
  side_by_side?: string | null
  side_by_side_keyclip?: string | null
  best_score?: number
  best_score_breakdown?: Record<string, unknown>
  best_score_formula?: string
  best_reason?: string
  public_item_id?: string
  visual_review_status?: string
  visual_review_note?: string | null
  preferred_best?: boolean
  time_start?: number
  time_end?: number
  evidence_level?: string
  pipeline_status?: string | null
  pipeline_stage?: string | null
  vlm_status?: string | null
  vlm_semantics?: ({ status?: string | null } & Record<string, unknown>) | null
  yolo_annotation_rendered?: boolean | null
  rerender_error?: string | null
  box_filter?: string | null
  secondary_objects?: string[] | null
  review_reason_codes?: string[] | null
  missing_reason?: string | null
  quality_reasons?: string[] | null
  warnings?: string[] | null
  [key: string]: unknown
}

export interface MaterialSearchResponse {
  total: number
  returned?: number
  items: MaterialSearchItem[]
}

export interface WorkspacePublishedMaterialsResponse extends MaterialSearchResponse {
  schema_version?: string
  storage?: string
  index_path?: string
  next_cursor?: string | null
  index_lifecycle?: WorkspacePublishedHealthResponse
  sort?: Record<string, unknown>
  permission_filter?: Record<string, unknown>
}

export interface WorkspacePublishedHealthResponse {
  schema_version?: string
  status?: string
  index_path?: string
  index_exists?: boolean
  index_mtime?: string | null
  latest_source_mtime?: string | null
  sqlite_count?: number | null
  expected_indexable_count?: number
  formal_jsonl_material_count?: number
  formal_report_count?: number
  experiment_count?: number
  warnings?: Array<Record<string, unknown>>
  warnings_before_rebuild?: Array<Record<string, unknown>>
  rebuild?: Record<string, unknown>
  experiments?: Array<Record<string, unknown>>
  [key: string]: unknown
}

export interface MaterialDiagnosticsEvidenceItem {
  candidate_id?: string | null
  candidate_group_id?: string | null
  asset_kind?: string | null
  display_name?: string | null
  canonical_action_type?: string | null
  canonical_object?: string | null
  sop_phase?: string | null
  review_status?: string | null
  approved_by?: string | null
  approved_at?: string | null
  yolo_recheck_status?: string | null
  yolo_valid_evidence_count?: number | null
  vlm_status?: string | null
  vlm_model?: string | null
  vlm_description?: string | null
  source_file?: string | null
  source_candidate_file?: string | null
  stored_file?: string | null
  material_url?: string | null
  material_exists?: boolean
  url_accessible?: boolean
  [key: string]: unknown
}

export interface MaterialDiagnosticsResponse {
  schema_version?: string
  experiment_id: string
  published_total?: number
  formal_material_total?: number
  best_material_total?: number
  candidate_total?: number
  candidate_pending_total?: number
  candidate_approved_total?: number
  candidate_rejected_total?: number
  candidate_deferred_total?: number
  candidate_not_selected_total?: number
  last_formal_sync_at?: string | null
  approved_unsynced_count?: number
  approved_unsynced_candidates?: MaterialDiagnosticsEvidenceItem[]
  file_access_status?: string
  missing_file_count?: number
  formal_material_reference_count?: number
  url_accessible_count?: number
  missing_clip_count?: number
  missing_preview_count?: number
  warnings_count?: number
  evidence_items?: MaterialDiagnosticsEvidenceItem[]
  taxonomy_calibration?: Record<string, unknown>
  [key: string]: unknown
}

export interface MaterialQualityReviewRecord {
  status: 'unreviewed' | 'confirmed_best' | 'needs_replacement' | 'unusable' | string
  preferred_item_id?: string | null
  reviewed_at?: string | null
  reviewed_by?: string | null
  note?: string | null
  reason?: string | null
}

export interface MaterialQualityReviewResponse {
  schema_version?: string
  experiment_id: string
  updated_at?: string | null
  reviews: Record<string, MaterialQualityReviewRecord>
}

export interface MaterialRetrievalEvaluationQuery {
  query: string
  expected_canonical_action_type: string
  top1_canonical_action_type?: string | null
  top3_canonical_action_types?: string[]
  top1_hit: boolean
  top3_hit: boolean
  wrong_classes?: string[]
  empty_result_reason?: string | null
  results?: MaterialSearchItem[]
}

export interface MaterialRetrievalEvaluationResponse {
  schema_version?: string
  experiment_id?: string | null
  generated_at?: string
  query_count: number
  top1_hit_count: number
  top3_hit_count: number
  top1_accuracy: number
  top3_accuracy: number
  queries: MaterialRetrievalEvaluationQuery[]
  index_lifecycle?: Record<string, unknown>
}

export interface VideoPlaybackCalibrationResponse {
  schema_version?: string
  experiment_id: string
  offset_adjust_sec: number
  updated_at?: string | null
  updated_by?: string | null
  note?: string | null
  history?: Array<Record<string, unknown>>
}

export interface MaterialCandidateBatchLogEntry {
  schema_version?: string
  batch_id: string
  experiment_id?: string
  action: 'approve' | 'false_positive' | string
  created_at?: string
  created_by?: string
  reason_code?: string | null
  reason?: string | null
  notes?: string | null
  candidate_group_ids?: string[]
  candidate_group_count?: number
  affected_candidate_ids?: string[]
  affected_candidate_count?: number
  expected_formal_additions?: number
  expected_formal_additions_by_action?: Record<string, number>
  before_status_counts?: Record<string, number>
  after_status_counts?: Record<string, number>
  undone?: boolean
  undone_at?: string | null
  undone_by?: string | null
  undo_notes?: string | null
  [key: string]: unknown
}

export interface MaterialCandidateBatchLogResponse {
  schema_version?: string
  experiment_id: string
  total: number
  items: MaterialCandidateBatchLogEntry[]
  log_path?: string
}

export interface MaterialCandidateBatchReviewResponse {
  experiment_id: string
  batch: MaterialCandidateBatchLogEntry
  candidates?: MaterialCandidatesResponse
  published_materials?: { total?: number; items?: MaterialSearchItem[] }
  diagnostics?: MaterialDiagnosticsResponse
  workspace_published_materials_reindex?: Record<string, unknown>
}

export interface MaterialTaxonomyRegressionResponse {
  schema_version?: string
  generated_at?: string
  requested_limit?: number
  experiment_count: number
  experiments: Array<Record<string, unknown>>
  aggregate_per_action?: Record<string, Record<string, number>>
}

export interface MaterialCandidateFile {
  item_id?: string
  experiment_id?: string
  timestamp_sec?: number
  local_timestamp_sec?: number
  camera_id?: string | null
  stream_id?: string | null
  frame_path?: string | null
  clip_id?: string | null
  clip_file_path?: string | null
  object_labels?: string[]
  actions?: string[]
  event_types?: string[]
  payload?: Record<string, unknown>
  preview_url?: string | null
  clip_url?: string | null
  report_url?: string | null
  material_url?: string | null
  published_paths?: Record<string, string>
  event_id?: string
  display_title?: string | null
  display_name?: string
  event_type?: string
  semantic_action?: string | null
  manipulated_object?: string | null
  instrument_context?: string | null
  corrected_primary_object?: string | null
  primary_object?: string | null
  raw_primary_object?: string | null
  yolo_primary_object?: string | null
  canonical_action_type?: string
  canonical_object?: string
  sop_phase?: string
  interaction_family?: string
  time_start?: number
  time_end?: number
  evidence_level?: string
  review_status?: string
  candidate_id?: string
  candidate_group_id?: string
  asset_kind?: string
  material_type?: string
  candidate_status?: string
  disposition?: string
  approval_reason_code?: string
  approval_reason?: string
  rejection_reason_code?: string
  rejection_reason?: string
  recommended?: boolean
  recommendation_reason?: string
  quality_score?: number
  quality_bucket?: string | null
  quality_tier?: string | null
  quality_grade?: string | null
  quality_label?: string | null
  yolo_annotation_rendered?: boolean | null
  rerender_error?: string | null
  box_filter?: string | null
  secondary_objects?: string[] | null
  url?: string | null
  exists?: boolean
  [key: string]: unknown
}

export interface MaterialCandidateGroup {
  candidate_group_id: string
  status?: string
  review_status?: string
  recommended?: boolean
  recommended_count?: number
  quality_score?: number
  pipeline_status?: string
  pipeline_stage?: string
  pipeline_flow?: string[]
  review_gate_policy?: string
  yolo_recheck?: ({ status?: string | null } & Record<string, unknown>) | null
  vlm_semantics?: ({ status?: string | null } & Record<string, unknown>) | null
  yolo_annotation_rendered?: boolean | null
  rerender_error?: string | null
  box_filter?: string | null
  secondary_objects?: string[] | null
  view?: string | null
  camera_view?: string | null
  display_title?: string | null
  semantic_action?: string | null
  manipulated_object?: string | null
  instrument_context?: string | null
  corrected_primary_object?: string | null
  primary_object?: string | null
  raw_primary_object?: string | null
  yolo_primary_object?: string | null
  action_name?: string | null
  canonical_action_type?: string | null
  canonical_object?: string | null
  sop_phase?: string | null
  interaction_family?: string | null
  approval_reason_code?: string | null
  approval_reason?: string | null
  rejection_reason_code?: string | null
  rejection_reason?: string | null
  review_notes?: string | null
  quality_bucket?: string | null
  quality_tier?: string | null
  quality_grade?: string | null
  quality_label?: string | null
  micro_segment_id?: string | null
  parent_segment_id?: string | null
  keyframes: MaterialCandidateFile[]
  clips: MaterialCandidateFile[]
  files: MaterialCandidateFile[]
}

export interface MaterialCandidatesResponse {
  schema_version?: string
  experiment_id: string
  total: number
  file_total?: number
  pending_total?: number
  approved_total?: number
  rejected_total?: number
  deferred_total?: number
  not_selected_total?: number
  items: MaterialCandidateGroup[]
  manifest?: Record<string, unknown>
  candidate_index?: string
}

export interface MaterialTimelineResponse {
  experiment_id: string
  total: number
  items: MaterialSearchItem[]
}

export interface MaterialHealthResponse {
  experiment_id: string
  material_count?: number
  pending_total?: number
  clip_count?: number
  status?: string
  [key: string]: unknown
}

export interface ClipBackfillResponse {
  clip_id?: string
  status: string
  [key: string]: unknown
}

export interface AsrUploadResponse {
  experiment_id: string
  provider: string
  model: string
  text: string
  segment_count: number
  context_count: number
  segments: Array<Record<string, unknown>>
}

export interface AsrJobResponse {
  job_id: string
  status: string
  [key: string]: unknown
}

export interface KeyActionStatus {
  experiment_id: string
  status: 'not_started' | 'queued' | 'running' | 'completed' | 'failed' | string
  progress: number
  message?: string
  error?: string
  output_dir?: string
  started_at?: string
  completed_at?: string
  updated_at?: string
  summary?: Record<string, unknown>
}

export interface KeyActionClipRef {
  video_path: string
  clip_path: string
  clip_url?: string | null
  annotated_clip_path?: string | null
  annotated_clip_url?: string | null
  yolo_label_counts?: Record<string, number>
  yolo_detection_count?: number
  local_start_sec: number
  local_end_sec: number
}

export interface KeyActionDetectorConfig {
  sample_fps?: number | null
  start_threshold?: number | null
  end_threshold?: number | null
  start_min_duration_sec?: number | null
  end_min_duration_sec?: number | null
  merge_gap_sec?: number | null
  min_segment_duration_sec?: number | null
  buffer_sec?: number | null
  motion_normalization?: string | null
  roi_mode?: string | null
  [key: string]: unknown
}

export interface KeyActionInteractionAssetPack {
  clip_path?: string | null
  clip_url?: string | null
  preview_path?: string | null
  preview_url?: string | null
  keyframe_paths?: string[]
  keyframe_urls?: Array<string | null>
  quality_score?: number | null
  quality_grade?: string | null
  [key: string]: unknown
}

export interface KeyActionInteractionEvent {
  event_id: string
  event_type?: string
  interaction?: string | null
  view?: string | null
  local_time_sec?: number | null
  global_time?: string | null
  hand_label?: string | null
  object_label?: string | null
  object_name?: string | null
  keyframe_path?: string | null
  display_name?: string | null
  stable_name?: string | null
  actor_name?: string | null
  start_time_sec?: number
  end_time_sec?: number
  duration_sec?: number | null
  confidence?: number | null
  overlap_sec?: number | null
  segment_id?: string | null
  involved_objects?: string[]
  related_detection_classes?: string[]
  evidence_grade?: string | null
  review_status?: string | null
  asset_pack?: KeyActionInteractionAssetPack | null
  clip_url?: string | null
  preview_url?: string | null
  keyframe_urls?: Array<string | null>
  [key: string]: unknown
}

export interface KeyActionInteractionKeyframe {
  event_id: string
  event_type?: string
  interaction?: string | null
  view?: string | null
  local_time_sec?: number | null
  global_time?: string | null
  labels?: string[]
  display_name?: string | null
  path: string
  url?: string | null
  timestamp_sec?: number | null
  frame_idx?: number | string | null
  score?: number | null
  index?: number
  segment_id?: string | null
  clip_url?: string | null
  preview_url?: string | null
  [key: string]: unknown
}

export interface KeyActionYoloFrameScan {
  available: boolean
  detection_frame_count?: number
  sampled_frame_count?: number
  tracklet_count?: number
  physical_event_count?: number
  asset_pack_count?: number
  event_types?: string[]
  error?: string
  [key: string]: unknown
}

export interface KeyActionSegment {
  session_id: string
  segment_id: string
  global_start_time: string
  global_end_time: string
  duration_sec: number
  third_person: KeyActionClipRef
  first_person?: KeyActionClipRef | null
  cv_detection: {
    avg_motion_score: number
    avg_active_score: number
    start_reason: string
    end_reason: string
  }
  text_description: {
    action_type: string
    summary: string
    tools: string[]
    objects: string[]
    numbers: string[]
  }
  dialogue_context: string[]
  yolo_labels?: string[]
  yolo_label_counts?: Record<string, number>
  visual_keywords?: string[]
  index: {
    embedding_id: string
    index_text: string
    vector_store: string
  }
  yolo_detections?: Array<{
    segment_id: string
    view: string
    phase: string
    image_path: string
    annotated_image_path: string
    annotated_image_url?: string | null
    model_path: string
    detections: Array<{ label: string; class_id: number; confidence: number; bbox: number[] }>
  }>
  yolo_annotated_clips?: Array<Record<string, unknown>>
  interaction_events?: KeyActionInteractionEvent[]
  interaction_keyframes?: KeyActionInteractionKeyframe[]
  micro_segments?: KeyActionMicroSegment[]
}

export interface KeyActionMicroSegment {
  micro_segment_id: string
  display_id?: string
  parent_segment_id: string
  session_id?: string
  global_start_time: string
  global_end_time: string
  start_sec?: number
  end_sec?: number
  duration_sec: number
  first_person?: (KeyActionClipRef & { annotated_clip_url?: string | null }) | null
  third_person?: KeyActionClipRef | null
  primary_object?: string | null
  primary_object_family?: string | null
  primary_object_arbitration?: string | null
  interaction_type?: string | null
  max_interaction_score?: number | null
  confidence?: string | null
  peak_keyframe?: string | null
  first_person_clip?: string | null
  third_person_clip?: string | null
  interaction?: {
    interaction_type: string
    primary_object: string
    primary_object_family?: string | null
    secondary_objects?: string[]
    detected_objects?: string[]
    avg_interaction_score?: number
    max_interaction_score?: number
  }
  keyframes?: {
    contact_frame?: string | null
    peak_frame?: string | null
    release_frame?: string | null
    contact_frame_url?: string | null
    peak_frame_url?: string | null
    release_frame_url?: string | null
    urls?: string[]
  }
  quality?: {
    confidence?: string
    warnings?: string[]
  }
  evidence_level?: string | null
  evidence_reasons?: string[]
  limitations?: string[]
  chinese_aliases?: string[]
  text_description?: {
    action_type?: string
    summary?: string
    index_text?: string
  }
  index?: {
    index_level?: string
    embedding_id?: string
  }
}

export interface KeyActionResults {
  experiment_id: string
  status: KeyActionStatus
  output_dir: string
  summary: Record<string, unknown>
  formal_report?: { path?: string | null; url?: string | null; available?: boolean; [key: string]: unknown } | null
  formal_report_path?: string | null
  formal_report_url?: string | null
  gt_coverage_status?: Record<string, unknown>
  metric_mode?: string | null
  query_validation_summary?: Record<string, unknown>
  family_merge_candidate_count?: number
  detected_segments: Array<Record<string, unknown>>
  segments: KeyActionSegment[]
  vector_metadata: Array<Record<string, unknown>>
  yolo_summary?: Record<string, unknown> | null
  yolo_clip_summary?: Record<string, unknown> | null
  yolo_detections?: Array<Record<string, unknown>>
  yolo_annotated_clips?: Array<Record<string, unknown>>
  yolo_frame_scan?: KeyActionYoloFrameScan | null
  interaction_summary?: Record<string, unknown> | null
  interaction_events?: KeyActionInteractionEvent[]
  interaction_keyframes?: KeyActionInteractionKeyframe[]
  micro_segments?: KeyActionMicroSegment[]
  micro_vector_metadata?: Array<Record<string, unknown>>
  micro_segment_config?: Record<string, unknown>
  micro_class_thresholds?: Record<string, Record<string, number>>
  micro_quality_stats?: Record<string, unknown>
  micro_merge_stats?: Record<string, unknown>
  micro_evaluation?: Record<string, unknown>
  object_family_merge_analysis?: Record<string, unknown>
  query_validation?: Record<string, unknown>
  detector_config?: KeyActionDetectorConfig | null
  detection_config?: KeyActionDetectorConfig | null
  debug: {
    roi_preview?: string | null
    frame_scores?: string | null
    frame_score_plot?: string | null
    segments_contact_sheet?: string | null
    report?: string | null
    formal_report?: Record<string, unknown> | string | null
    formal_report_path?: string | null
    formal_report_url?: string | null
    detector_config?: KeyActionDetectorConfig | null
  }
}

export interface KeyActionReviewItem {
  item_id: string
  item_type: 'qa_warning' | 'evidence_semantic' | 'segment' | 'micro_segment' | string
  source_id?: string
  title?: string
  summary?: string
  severity?: 'error' | 'warning' | 'info' | string
  review_status?: 'pending' | 'approved' | 'rejected' | 'needs_review' | string
  reviewer?: string | null
  review_note?: string | null
  reviewed_at?: string | null
  segment_id?: string | null
  micro_segment_id?: string | null
  confidence?: number | null
  start_sec?: number | null
  end_sec?: number | null
  duration_sec?: number | null
  adjusted_start_sec?: number | null
  adjusted_end_sec?: number | null
  reasons?: string[]
  preview_urls?: string[]
  clip_urls?: string[]
  preview_paths?: string[]
  clip_paths?: string[]
  boundary?: Record<string, unknown>
  payload?: Record<string, unknown>
  [key: string]: unknown
}

export interface KeyActionQualityPayload {
  experiment_id?: string
  schema_version?: string
  status?: string
  health_score?: number
  core_metrics?: {
    segment_count?: number
    micro_segment_count?: number
    longest_segment_sec?: number | null
    longest_segment_ratio?: number | null
    total_action_coverage_ratio?: number | null
    vector_count?: number
    unreviewed_count?: number
    review_decision_counts?: Record<string, number>
  }
  health?: Record<string, unknown>
  boundary_policy?: Record<string, unknown>
  coverage_check?: Record<string, unknown>
  quality_gate?: Record<string, unknown>
  boundary_refinement_candidates?: Array<Record<string, unknown>>
  long_segment_split_candidates?: Array<Record<string, unknown>>
  recommendations?: Array<Record<string, unknown>>
  [key: string]: unknown
}

export interface KeyActionReviewQueue {
  schema_version?: string
  experiment_id: string
  generated_at?: string
  summary: {
    total: number
    pending: number
    approved: number
    rejected: number
    needs_review: number
    quality_score?: number
    segment_count?: number
    micro_segment_count?: number
    long_segment_candidate_count?: number
    boundary_refinement_candidate_count?: number
  }
  quality?: KeyActionQualityPayload
  items: KeyActionReviewItem[]
}

export interface KeyActionEvidenceAdapters {
  schema_version?: string
  experiment_id: string
  metadata_dir?: string
  protocol_doc?: string
  input_contracts?: Record<string, string>
  accepted_aliases?: Record<string, string[]>
  counts?: Record<string, number>
  summary?: Record<string, number>
  validation?: Record<string, unknown>
  adapters?: Record<string, {
    adapter?: string
    canonical_file?: string
    present?: boolean
    row_count?: number
    valid_row_count?: number
    error_count?: number
    warning_count?: number
    semantic_issue_count?: number
    status?: string
    coverage?: Record<string, unknown>
    views?: string[]
    issues?: Array<Record<string, unknown>>
    [key: string]: unknown
  }>
  ready?: boolean
}

export interface KeyActionRetrievalEvaluation {
  experiment_id: string
  evaluation: Record<string, unknown>
}
