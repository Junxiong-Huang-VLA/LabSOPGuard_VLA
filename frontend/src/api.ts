import axios from 'axios'
import type {
  AnalysisOverview,
  AsrJobResponse,
  AsrUploadResponse,
  ActionAlignmentHealth,
  ClipBackfillResponse,
  CreateExperimentRequest,
  DisplayTimingStageSummary,
  SegmentPreviewCanonicalFields,
  Experiment,
  ExperimentTaskStatus,
  FormalPublishHealth,
  KeyActionResults,
  KeyActionReviewQueue,
  KeyActionEvidenceAdapters,
  KeyActionRetrievalEvaluation,
  KeyActionStatus,
  MaterialCandidateBatchLogResponse,
  MaterialCandidateBatchReviewResponse,
  MaterialCandidatesResponse,
  MaterialDiagnosticsResponse,
  MaterialHealthResponse,
  MaterialSearchItem,
  MaterialQualityReviewResponse,
  MaterialRetrievalEvaluationResponse,
  MaterialSearchResponse,
  MaterialTaxonomyRegressionResponse,
  MaterialTimelineResponse,
  SamplingTransparency,
  StepRecord,
  TimingHistorySummary,
  UpdateStepRequest,
  UploadE2EBenchmarkSummary,
  VideoPlaybackCalibrationResponse,
  VideoMemoryOverview,
  VideoMemoryQueryAnswer,
  WorkspacePublishedHealthResponse,
  WorkspacePublishedMaterialsResponse,
} from './types'

type ApiCacheOptions = { ttlMs?: number; force?: boolean }
type PublishedMaterialAlignmentGate = {
  status?: string
  reason?: string
  hidden_item_count?: number
}
type PublishedMaterialsResponse = {
  schema_version?: string
  experiment_id: string
  total: number
  returned?: number
  items: MaterialSearchItem[]
  all_items?: MaterialSearchItem[]
  grouped_items?: unknown[]
  alignment_gate?: PublishedMaterialAlignmentGate
}
export type TimeAxisHealthPayload = {
  schema_version?: string
  status?: string
  time_axis_unreliable?: boolean
  can_publish_formal_materials?: boolean
  can_write_video_memory?: boolean
  source_path?: string | null
  reasons?: string[]
  raw?: Record<string, unknown>
}
export type ExperimentSubExperimentSegment = SegmentPreviewCanonicalFields & {
  segment_id: string
  index: number
  start_sec: number
  end_sec: number
  duration_sec: number
  experiment_window_duration_s?: number | null
  preview_duration_s?: number | null
  preview_mode?: string | null
  playback_speed_ratio?: number | null
  preview_playback_speed_ratio?: number | null
  window_preview_output_fps?: number | null
  time_basis?: string | null
  third_person_local_start_sec?: number | null
  third_person_local_end_sec?: number | null
  first_person_local_start_sec?: number | null
  first_person_local_end_sec?: number | null
  activity_segment_count: number
  boundary_before?: Record<string, unknown> | null
  display_name?: string
  scene_summary?: string
  naming_confidence?: number
  naming_source?: string
  preview_ready?: boolean
  preview_video_url?: string | null
  preview_poster_url?: string | null
  window_preview_url?: string | null
  sample_grid_url?: string | null
  window_sync_index?: string | null
  source_window_sync_index?: string | null
  window_reference_camera_audit?: string | null
  window_alignment_quality?: string | null
  window_self_validation_status?: string | null
  first_person_video_url?: string | null
  third_person_video_url?: string | null
  first_person_annotated_video_url?: string | null
  third_person_annotated_video_url?: string | null
  first_person_poster_url?: string | null
  third_person_poster_url?: string | null
  first_person_preview_time_sec?: number | null
  third_person_preview_time_sec?: number | null
  annotated_ready?: boolean
  micro_segment_count?: number
  yolo_interaction_count?: number
  reason_code?: string | null
  decision_path?: string | null
  view_alignment_status?: string | null
  view_alignment_warning?: string | null
  formal_action_status?: 'confirmed' | 'candidate' | string | null
  formal_dual_view_action?: boolean
  formal_action_source?: string | null
  formal_action_reason?: string | null
  dual_view_action_event_ids?: string[]
  view_support?: Record<string, {
    row_count?: number
    active_row_count?: number
    interaction_count?: number
    has_physical_action?: boolean
    labels?: Record<string, number>
  }>
  top_objects?: string[]
  key_materials?: Array<{
    micro_segment_id?: string | null
    title?: string | null
    primary_object?: string | null
    interaction_type?: string | null
    start_sec?: number | null
    end_sec?: number | null
    views?: Record<string, {
      clip_url?: string | null
      keyframe_url?: string | null
      local_start_sec?: number | null
      local_end_sec?: number | null
      material_type?: string | null
    }>
  }>
  micro_segments?: Array<{
    micro_segment_id?: string | null
    display_id?: string | null
    primary_object?: string | null
    interaction_type?: string | null
    confidence?: string | number | null
    duration_sec?: number | null
    global_start_time?: string | null
    global_end_time?: string | null
    max_interaction_score?: number | null
  }>
}
export type ExperimentSubExperiment = {
  sub_id: string
  segment_index: number
  path?: string
  has_materials: boolean
  first_person_video_url?: string | null
  third_person_video_url?: string | null
  first_person_poster_url?: string | null
  third_person_poster_url?: string | null
}
export type ExperimentSubExperimentsResponse = {
  total: number
  official_total?: number
  candidate_total?: number
  segments: ExperimentSubExperimentSegment[]
  sub_experiments: ExperimentSubExperiment[]
  video_duration_sec?: number | null
  preview_ready?: boolean
  preview_video_url?: string | null
  segment_preview_ready_count?: number
  preview_error?: string | null
  message?: string
  source?: string | null
  candidate_debug_total?: number
  dual_view_action_event_count?: number
  formal_action_policy?: string | null
  time_axis_unreliable?: boolean
  time_axis_health?: TimeAxisHealthPayload | null
  formal_publish_health?: FormalPublishHealth | null
  action_alignment_health?: ActionAlignmentHealth | null
  unmatched_view_evidence_count?: number
}
export type ExperimentTimingResponse = {
  status?: string
  stages: Record<string, number>
  stage_rows?: Array<{
    stage: string
    stage_label_zh?: string
    duration_sec?: number | null
    inputs?: number | null
    outputs?: number | null
    errors?: number | null
    workers?: number | null
  }>
  stage_labels_zh?: Record<string, string>
  total_sec: number
  core_analysis_sec?: number | null
  algorithm_elapsed_sec?: number | null
  server_end_to_end_sec?: number | null
  upload_save_sec?: number | null
  queue_wait_sec?: number | null
  performance_benchmark?: AnalysisOverview['run']['timing'] extends infer T
    ? T extends { performance_benchmark?: infer B }
      ? B
      : never
    : never
  upload_e2e_benchmark?: UploadE2EBenchmarkSummary
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
}
export type SegmentUnderstanding = {
  segment_id: string
  index?: number | null
  display_name?: string | null
  start_sec?: number | null
  end_sec?: number | null
  understanding_text: string
  source: 'qwen' | 'template' | string
  source_row_count: number
}
export type SegmentUnderstandingResponse = {
  experiment_id: string
  schema_version?: string
  total: number
  segments: SegmentUnderstanding[]
}
export type ProcessingQueueStatusResponse = {
  max_concurrent: number
  processing_count: number
  queue_size: number
}
export type StepMaterialQueryResponse = {
  experiment_id: string
  step_text: string
  message_video_time_sec?: number | null
  search_window?: { start_sec: number; end_sec: number } | null
  judgement: Record<string, unknown>
  candidates: Array<Record<string, unknown>>
}
export type StepMaterialFeishuPushResponse = {
  experiment_id: string
  query: StepMaterialQueryResponse
  feishu: Record<string, unknown>
}
type KeyActionUploadInputs = {
  firstPersonVideo?: File | null
  thirdPersonVideo?: File | null
  topVideo?: File | null
  bottomVideo?: File | null
  sessionStartTime?: string | null
}

type KeyActionRawUploadView = 'first_person' | 'third_person'

const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' },
})

function apiErrorMessage(error: unknown) {
  const record = error && typeof error === 'object' ? error as { response?: { data?: { detail?: unknown } }; message?: string } : {}
  const detail = record.response?.data?.detail
  if (detail && typeof detail === 'object') {
    const detailRecord = detail as Record<string, unknown>
    for (const key of ['message_zh', 'error_message', 'message', 'error']) {
      const value = detailRecord[key]
      if (typeof value === 'string' && value.trim()) return value.trim()
    }
  }
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  return record.message || '请求失败'
}

api.interceptors.response.use(
  response => response,
  error => {
    const nextError = new Error(apiErrorMessage(error))
    ;(nextError as Error & { response?: unknown }).response = error?.response
    return Promise.reject(nextError)
  },
)

const DEFAULT_GET_TTL_MS = 45_000
const apiGetCache = new Map<string, { expiresAt: number; value?: unknown; promise?: Promise<unknown> }>()

function stableParams(params?: Record<string, unknown>) {
  if (!params) return ''
  return JSON.stringify(Object.entries(params).filter(([, value]) => value !== undefined && value !== null && value !== '').sort(([a], [b]) => a.localeCompare(b)))
}

function getCacheKey(url: string, params?: Record<string, unknown>) {
  return `${url}?${stableParams(params)}`
}

function readCachedGet<T>(url: string, params?: Record<string, unknown>): T | undefined {
  const cached = apiGetCache.get(getCacheKey(url, params))
  if (cached?.value === undefined || cached.expiresAt <= Date.now()) return undefined
  return cached.value as T
}

async function cachedGet<T>(url: string, params?: Record<string, unknown>, options: ApiCacheOptions = {}): Promise<T> {
  const ttlMs = options.ttlMs ?? DEFAULT_GET_TTL_MS
  const key = getCacheKey(url, params)
  const now = Date.now()
  const cached = apiGetCache.get(key)
  if (!options.force && cached?.value !== undefined && cached.expiresAt > now) return cached.value as T
  if (!options.force && cached?.promise) return cached.promise as Promise<T>
  const requestParams = options.force ? { ...(params || {}), _cache_bust: Date.now() } : params
  const promise = api.get<T>(url, {
    params: requestParams,
    headers: options.force ? { 'Cache-Control': 'no-cache' } : undefined,
  }).then(({ data }) => {
    apiGetCache.set(key, { value: data, expiresAt: Date.now() + ttlMs })
    return data
  }).catch(error => {
    const latest = apiGetCache.get(key)
    if (latest?.promise === promise) apiGetCache.delete(key)
    throw error
  })
  apiGetCache.set(key, { value: cached?.value, expiresAt: cached?.expiresAt ?? 0, promise })
  return promise
}

function prefetchGet<T>(url: string, params?: Record<string, unknown>, options?: ApiCacheOptions) {
  void cachedGet<T>(url, params, options).catch(() => undefined)
}

function workspaceOperatorHeaders() {
  const role = String(import.meta.env.VITE_OPERATOR_ROLE || 'admin')
  const allowedExperiments = String(import.meta.env.VITE_ALLOWED_EXPERIMENTS || '')
  return {
    'X-Operator-Role': role,
    ...(allowedExperiments ? { 'X-Allowed-Experiments': allowedExperiments } : {}),
  }
}

export function invalidateApiCache(match?: string) {
  if (!match) {
    apiGetCache.clear()
    return
  }
  for (const key of Array.from(apiGetCache.keys())) {
    if (key.includes(match)) apiGetCache.delete(key)
  }
}

export function invalidateExperimentCache(id: string) {
  invalidateApiCache(`/experiments/${id}`)
  invalidateApiCache('/experiments?')
}

export function prefetchExperimentsList() {
  prefetchGet<{ total: number; experiments: Experiment[] }>('/experiments', { limit: 50 }, { ttlMs: 20_000 })
}

export function prefetchExperimentRoute(id: string, route: 'workspace' | 'materials' | 'materialTimeline' | 'steps' | 'json' | 'report' | 'keyActions' | 'reviewQueue') {
  if (!id) return
  if (route === 'workspace' || route === 'report' || route === 'steps') {
    prefetchGet<AnalysisOverview>(`/experiments/${id}/analysis-overview`, undefined, { ttlMs: 15_000 })
  }
  if (route === 'materials' || route === 'report') {
    prefetchGet<PublishedMaterialsResponse>(`/experiments/${id}/materials/published`, route === 'report' ? { limit: 8 } : undefined, { ttlMs: 60_000 })
  }
  if (route === 'materialTimeline') {
    prefetchGet<MaterialTimelineResponse>(`/experiments/${id}/materials/timeline`, { limit: 1000 }, { ttlMs: 60_000 })
  }
  if (route === 'keyActions') {
    prefetchGet<KeyActionStatus>(`/experiments/${id}/key-actions/status`, undefined, { ttlMs: 15_000 })
    prefetchGet<KeyActionResults>(`/experiments/${id}/key-actions/results`, undefined, { ttlMs: 15_000 })
  }
  if (route === 'reviewQueue') {
    prefetchGet<KeyActionReviewQueue>(`/experiments/${id}/key-actions/review-queue`, undefined, { ttlMs: 15_000 })
  }
}

export const experimentApi = {
  list: (params?: { limit?: number; offset?: number }, options?: ApiCacheOptions) => cachedGet<{ total: number; experiments: Experiment[] }>('/experiments', params, { ttlMs: 20_000, ...options }),

  create: async (request: CreateExperimentRequest) => {
    const { data } = await api.post<{ experiment_id: string; experiment: Experiment }>('/experiments', request)
    invalidateApiCache('/experiments')
    return data
  },

  get: (id: string, options?: ApiCacheOptions) => cachedGet<Experiment>(`/experiments/${id}`, undefined, { ttlMs: 20_000, ...options }),

  delete: async (id: string) => {
    const { data } = await api.delete(`/experiments/${id}`)
    invalidateExperimentCache(id)
    return data
  },

  archive: async (id: string) => {
    const { data } = await api.post<Experiment>(`/experiments/${id}/archive`)
    invalidateExperimentCache(id)
    return data
  },

  unarchive: async (id: string) => {
    const { data } = await api.post<Experiment>(`/experiments/${id}/unarchive`)
    invalidateExperimentCache(id)
    return data
  },

  getStatus: async (id: string) => {
    const { data } = await api.get<ExperimentTaskStatus>(`/experiments/${id}/status`)
    return data
  },

  getAnalysisOverview: (id: string, options?: ApiCacheOptions) => cachedGet<AnalysisOverview>(`/experiments/${id}/analysis-overview`, undefined, { ttlMs: 15_000, ...options }),
  peekAnalysisOverview: (id: string) => readCachedGet<AnalysisOverview>(`/experiments/${id}/analysis-overview`),
  getVideoMemoryOverview: (params?: { retention_days?: number; max_items?: number; experiment_id?: string }, options?: ApiCacheOptions) => cachedGet<VideoMemoryOverview>('/video-memory/overview', params, { ttlMs: 15_000, ...options }),
  getExperimentVideoMemoryOverview: (id: string, params?: { retention_days?: number; max_items?: number }, options?: ApiCacheOptions) => cachedGet<VideoMemoryOverview>(`/experiments/${id}/video-memory/overview`, params, { ttlMs: 15_000, ...options }),
  getLatestVideoMemorySnapshot: (options?: ApiCacheOptions) => cachedGet<Record<string, unknown>>('/video-memory/snapshots/latest', undefined, { ttlMs: 15_000, ...options }),
  buildVideoMemorySnapshot: async (request: { window_end_date?: string; window_days?: number; job_type?: 'backfill' | 'incremental' | 'rebuild' | 'feedback_update'; force_material_sync?: boolean } = {}) => {
    const { data } = await api.post<Record<string, unknown>>('/video-memory/build', request)
    invalidateApiCache('/video-memory/snapshots/latest')
    invalidateApiCache('/video-memory/overview')
    return data
  },
  queryVideoMemory: async (request: { query: string; snapshot_id?: string; filters?: Record<string, unknown>; limit?: number }) => {
    const { data } = await api.post<VideoMemoryQueryAnswer>('/video-memory/query', request)
    return data
  },
  feedbackVideoMemory: async (request: { target_type: string; target_id: string; feedback_type: string; context_fields?: Record<string, unknown>; note?: string; user_id?: string; rebuild_snapshot?: boolean }) => {
    const { data } = await api.post<Record<string, unknown>>('/video-memory/feedback', request)
    invalidateApiCache('/video-memory/snapshots/latest')
    invalidateApiCache('/video-memory/overview')
    return data
  },
  getLatestUploadE2EBenchmark: (options?: ApiCacheOptions) => cachedGet<UploadE2EBenchmarkSummary>('/benchmarks/upload-e2e/latest', undefined, { ttlMs: 30_000, ...options }),
  getJson: (id: string, options?: ApiCacheOptions) => cachedGet<Record<string, unknown>>(`/experiments/${id}/analysis-overview`, undefined, { ttlMs: 15_000, ...options }),
  getSubExperiments: (id: string, options?: ApiCacheOptions) => cachedGet<ExperimentSubExperimentsResponse>(`/experiments/${id}/sub-experiments`, undefined, { ttlMs: 15_000, ...options }),
  getSegmentUnderstanding: (id: string, options?: ApiCacheOptions) => cachedGet<SegmentUnderstandingResponse>(`/experiments/${id}/segment-understanding`, undefined, { ttlMs: 30_000, ...options }),
  getTiming: (id: string, options?: ApiCacheOptions) => cachedGet<ExperimentTimingResponse>(`/experiments/${id}/timing`, undefined, { ttlMs: 5_000, ...options }),
  getProcessingQueueStatus: (options?: ApiCacheOptions) => cachedGet<ProcessingQueueStatusResponse>('/processing/queue-status', undefined, { ttlMs: 3_000, ...options }),
  queryStepMaterials: async (id: string, request: { step_text: string; message_sent_at?: string | null; window_before_sec?: number; window_after_sec?: number; limit?: number }) => {
    const { data } = await api.post<StepMaterialQueryResponse>(`/experiments/${id}/materials/step-query`, request)
    return data
  },
  pushStepMaterialsToFeishu: async (id: string, request: { step_text: string; message_sent_at?: string | null; window_before_sec?: number; window_after_sec?: number; limit?: number; include_evidence_image?: boolean; dry_run?: boolean; public_base_url?: string | null }) => {
    const { data } = await api.post<StepMaterialFeishuPushResponse>(`/experiments/${id}/materials/step-query/feishu`, request)
    return data
  },
  getFeishuStatus: async () => {
    const { data } = await api.get<Record<string, unknown>>('/feishu/status')
    return data
  },

  uploadVideo: async (id: string, file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    const { data } = await api.post(`/experiments/${id}/upload/video`, formData, { headers: { 'Content-Type': 'multipart/form-data' } })
    invalidateExperimentCache(id)
    return data
  },

  uploadKeyActionRawVideo: async (id: string, view: KeyActionRawUploadView, file: File, onProgress?: (loaded: number, total: number) => void) => {
    const { data } = await api.put(`/experiments/${id}/key-actions/upload-raw/${view}`, file, {
      headers: {
        'Content-Type': file.type || 'application/octet-stream',
        'X-Filename': encodeURIComponent(file.name || `${view}.mp4`),
      },
      onUploadProgress: event => {
        if (!onProgress) return
        onProgress(event.loaded || 0, event.total || file.size || 0)
      },
    })
    invalidateExperimentCache(id)
    return data
  },

  uploadKeyActionAnalysisProxy: async (id: string, view: KeyActionRawUploadView, file: File, onProgress?: (loaded: number, total: number) => void) => {
    const { data } = await api.put(`/experiments/${id}/key-actions/upload-analysis-proxy/${view}`, file, {
      headers: {
        'Content-Type': file.type || 'video/mp4',
        'X-Filename': encodeURIComponent(file.name || `${view}_analysis_proxy.mp4`),
      },
      onUploadProgress: event => {
        if (!onProgress) return
        onProgress(event.loaded || 0, event.total || file.size || 0)
      },
    })
    invalidateExperimentCache(id)
    return data
  },

  runKeyActions: async (id: string, request?: { session_start_time?: string | null; force?: boolean }) => {
    const { data } = await api.post(`/experiments/${id}/key-actions/run`, request || {})
    invalidateExperimentCache(id)
    return data
  },

  uploadAndAnalyze: async (id: string, files: KeyActionUploadInputs, onProgress?: (percent: number) => void) => {
    const rawUploadEligible = Boolean(files.firstPersonVideo && files.thirdPersonVideo && !files.topVideo && !files.bottomVideo)
    if (rawUploadEligible && files.firstPersonVideo && files.thirdPersonVideo) {
      const totals = {
        first_person: files.firstPersonVideo.size || 0,
        third_person: files.thirdPersonVideo.size || 0,
      }
      const loaded = { first_person: 0, third_person: 0 }
      const reportProgress = (view: KeyActionRawUploadView, nextLoaded: number, nextTotal: number) => {
        loaded[view] = Math.max(loaded[view], nextLoaded)
        totals[view] = Math.max(totals[view], nextTotal)
        const loadedBytes = loaded.first_person + loaded.third_person
        const totalBytes = Math.max(1, totals.first_person + totals.third_person)
        onProgress?.(Math.min(100, Math.round((loadedBytes / totalBytes) * 100)))
      }
      await Promise.all([
        experimentApi.uploadKeyActionRawVideo(id, 'first_person', files.firstPersonVideo, (nextLoaded, nextTotal) => reportProgress('first_person', nextLoaded, nextTotal)),
        experimentApi.uploadKeyActionRawVideo(id, 'third_person', files.thirdPersonVideo, (nextLoaded, nextTotal) => reportProgress('third_person', nextLoaded, nextTotal)),
      ])
      onProgress?.(100)
      const data = await experimentApi.runKeyActions(id, {
        session_start_time: files.sessionStartTime || null,
        force: true,
      })
      invalidateExperimentCache(id)
      return data
    }

    const formData = new FormData()
    if (files.firstPersonVideo) formData.append('first_person_video', files.firstPersonVideo)
    if (files.thirdPersonVideo) formData.append('third_person_video', files.thirdPersonVideo)
    if (files.topVideo) formData.append('top_video', files.topVideo)
    if (files.bottomVideo) formData.append('bottom_video', files.bottomVideo)
    if (files.sessionStartTime) formData.append('session_start_time', files.sessionStartTime)
    const { data } = await api.post(`/experiments/${id}/key-actions/upload-and-run`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: event => {
        if (!onProgress || !event.total) return
        onProgress(Math.min(100, Math.round((event.loaded / event.total) * 100)))
      },
    })
    invalidateExperimentCache(id)
    return data
  },

  uploadContext: async (id: string, contextText: string) => {
    const formData = new FormData()
    formData.append('context_text', contextText)
    const { data } = await api.post(`/experiments/${id}/upload/context`, formData)
    return data
  },

  uploadProtocol: async (id: string, protocolText: string) => {
    const formData = new FormData()
    formData.append('protocol_text', protocolText)
    const { data } = await api.post(`/experiments/${id}/upload/protocol`, formData)
    return data
  },

  uploadAsr: async (id: string, audio: File, options?: { language?: string; prompt?: string }) => {
    const formData = new FormData()
    formData.append('audio', audio)
    if (options?.language) formData.append('language', options.language)
    if (options?.prompt) formData.append('prompt', options.prompt)
    const { data } = await api.post<AsrUploadResponse>(`/experiments/${id}/upload/asr`, formData, { headers: { 'Content-Type': 'multipart/form-data' } })
    return data
  },

  submitAsrJob: async (id: string, audio: File) => {
    const formData = new FormData()
    formData.append('audio', audio)
    const { data } = await api.post<AsrJobResponse>(`/experiments/${id}/upload/asr/jobs`, formData, { headers: { 'Content-Type': 'multipart/form-data' } })
    return data
  },

  process: async (id: string, videoPath?: string) => {
    const { data } = await api.post(`/experiments/${id}/process`, { video_path: videoPath ?? null })
    invalidateExperimentCache(id)
    return data
  },

  getSteps: (id: string, options?: ApiCacheOptions) => cachedGet<StepRecord[]>(`/experiments/${id}/steps`, undefined, { ttlMs: 45_000, ...options }),
  getStep: async (id: string, stepId: string) => {
    const { data } = await api.get<StepRecord>(`/experiments/${id}/steps/${stepId}`)
    return data
  },
  updateStep: async (id: string, stepId: string, request: UpdateStepRequest) => {
    const { data } = await api.patch<{ step?: StepRecord } | StepRecord>(`/experiments/${id}/steps/${stepId}`, request)
    invalidateExperimentCache(id)
    return 'step' in data && data.step ? data.step : data
  },

  searchMaterials: async (id: string, params: { objects?: string; actions?: string; camera_id?: string; stream_id?: string; has_clip?: boolean; clip_exists?: boolean; text?: string; embedding_text?: string; limit?: number }) => {
    const { data } = await api.get<MaterialSearchResponse>(`/experiments/${id}/materials/search`, { params })
    return data
  },

  getMaterialHealth: async (id: string) => {
    const { data } = await api.get<MaterialHealthResponse>(`/experiments/${id}/materials/health`)
    return data
  },

  getMaterialDiagnostics: async (id: string) => {
    const { data } = await api.get<MaterialDiagnosticsResponse>(`/experiments/${id}/materials/diagnostics`)
    return data
  },

  reindexMaterials: async (id: string, force = true) => {
    const { data } = await api.post(`/experiments/${id}/materials/reindex`, null, { params: { force } })
    return data
  },

  getMaterialTimeline: (id: string, params?: { limit?: number }, options?: ApiCacheOptions) => cachedGet<MaterialTimelineResponse>(`/experiments/${id}/materials/timeline`, params, { ttlMs: 60_000, ...options }),
  peekMaterialTimeline: (id: string, params?: { limit?: number }) => readCachedGet<MaterialTimelineResponse>(`/experiments/${id}/materials/timeline`, params),
  getPublishedMaterials: (id: string, params?: { limit?: number }, options?: ApiCacheOptions) => cachedGet<PublishedMaterialsResponse>(`/experiments/${id}/materials/published`, params, { ttlMs: 0, force: true, ...options }),
  peekPublishedMaterials: (id: string, params?: { limit?: number }) => readCachedGet<PublishedMaterialsResponse>(`/experiments/${id}/materials/published`, params),
  getMaterialCandidates: (id: string, options?: ApiCacheOptions) => cachedGet<MaterialCandidatesResponse>(`/experiments/${id}/materials/candidates`, undefined, { ttlMs: 30_000, ...options }),
  getMaterialQualityReview: (id: string, options?: ApiCacheOptions) => cachedGet<MaterialQualityReviewResponse>(`/experiments/${id}/materials/quality-review`, undefined, { ttlMs: 30_000, ...options }),

  getVideoCalibration: (id: string, options?: ApiCacheOptions) => cachedGet<VideoPlaybackCalibrationResponse>(`/experiments/${id}/video-calibration`, undefined, { ttlMs: 30_000, ...options }),

  updateVideoCalibration: async (id: string, request: { offset_adjust_sec: number; reviewer?: string; note?: string }) => {
    const { data } = await api.put<VideoPlaybackCalibrationResponse>(`/experiments/${id}/video-calibration`, request)
    invalidateExperimentCache(id)
    return data
  },

  resetVideoCalibration: async (id: string, request?: { reviewer?: string; note?: string }) => {
    const { data } = await api.post<VideoPlaybackCalibrationResponse>(`/experiments/${id}/video-calibration/reset`, request || {})
    invalidateExperimentCache(id)
    return data
  },

  updateMaterialQualityReview: async (id: string, canonicalActionType: string, request: { status: string; preferred_item_id?: string | null; reviewer?: string; note?: string; reason?: string }) => {
    const { data } = await api.patch<{ quality_review: MaterialQualityReviewResponse; published_materials?: PublishedMaterialsResponse; diagnostics?: MaterialDiagnosticsResponse }>(`/experiments/${id}/materials/quality-review/${encodeURIComponent(canonicalActionType)}`, request)
    invalidateExperimentCache(id)
    return data
  },

  getMaterialRetrievalEvaluation: (id: string, options?: ApiCacheOptions) => cachedGet<MaterialRetrievalEvaluationResponse>(`/experiments/${id}/materials/retrieval-evaluation`, undefined, { ttlMs: 20_000, ...options }),

  getMaterialTaxonomyRegression: (limit = 3, options?: ApiCacheOptions) => cachedGet<MaterialTaxonomyRegressionResponse>('/materials/taxonomy-regression', { limit }, { ttlMs: 60_000, ...options }),

  approveMaterialCandidate: async (id: string, candidateGroupId: string, request?: { reviewer?: string; notes?: string; reason_code?: string; reason?: string; candidate_ids?: string[]; selected_keyframe_ids?: string[]; selected_clip_ids?: string[] }) => {
    const { data } = await api.post(`/experiments/${id}/materials/candidates/${candidateGroupId}/approve`, request || {})
    invalidateExperimentCache(id)
    return data
  },

  confirmMaterialCandidate: async (id: string, candidateGroupId: string, request?: { reviewer?: string; notes?: string; candidate_ids?: string[] }) => {
    const { data } = await api.post(`/experiments/${id}/materials/candidates/${candidateGroupId}/decision`, {
      decision: 'confirmed',
      ...(request || {}),
    })
    invalidateExperimentCache(id)
    return data
  },

  renameMaterialCandidate: async (id: string, candidateGroupId: string, request: { display_title: string; reviewer?: string; notes?: string; candidate_ids?: string[] }) => {
    const { data } = await api.patch(`/experiments/${id}/materials/candidates/${candidateGroupId}/rename`, request)
    invalidateExperimentCache(id)
    return data
  },

  markMaterialEvidenceMismatch: async (id: string, candidateGroupId: string, request?: { reviewer?: string; notes?: string; reason_code?: string; reason?: string; candidate_ids?: string[] }) => {
    const { data } = await api.post(`/experiments/${id}/materials/candidates/${candidateGroupId}/decision`, {
      decision: 'evidence_mismatch',
      reason_code: 'evidence_mismatch',
      ...(request || {}),
    })
    invalidateExperimentCache(id)
    return data
  },

  getMaterialCandidateBatches: (id: string, options?: ApiCacheOptions) => cachedGet<MaterialCandidateBatchLogResponse>(`/experiments/${id}/materials/candidates/batches`, undefined, { ttlMs: 30_000, ...options }),

  reviewMaterialCandidatesBatch: async (id: string, request: { action: 'approve' | 'false_positive'; groups: Array<{ candidate_group_id: string; candidate_ids?: string[] }>; reviewer?: string; reason_code?: string; reason?: string; notes?: string }) => {
    const { data } = await api.post<MaterialCandidateBatchReviewResponse>(`/experiments/${id}/materials/candidates/batch`, request)
    invalidateExperimentCache(id)
    return data
  },

  undoMaterialCandidateBatch: async (id: string, batchId: string, request?: { reviewer?: string; notes?: string; candidate_ids?: string[] }) => {
    const { data } = await api.post<MaterialCandidateBatchReviewResponse>(`/experiments/${id}/materials/candidates/batches/${encodeURIComponent(batchId)}/undo`, request || {})
    invalidateExperimentCache(id)
    return data
  },

  decideMaterialCandidate: async (id: string, candidateGroupId: string, request: { decision: string; reviewer?: string; reason_code?: string; reason?: string; notes?: string; candidate_ids?: string[] }) => {
    const { data } = await api.post(`/experiments/${id}/materials/candidates/${candidateGroupId}/decision`, request)
    invalidateExperimentCache(id)
    return data
  },

  restoreMaterialCandidate: async (id: string, candidateGroupId: string, request?: { reviewer?: string; notes?: string; candidate_ids?: string[] }) => {
    const { data } = await api.post(`/experiments/${id}/materials/candidates/${candidateGroupId}/restore`, request || {})
    invalidateExperimentCache(id)
    return data
  },

  backfillClip: async (id: string, request: { start_time_sec: number; end_time_sec: number; camera_id?: string; clip_id?: string }) => {
    const { data } = await api.post<ClipBackfillResponse>(`/experiments/${id}/materials/backfill-clip`, request)
    invalidateExperimentCache(id)
    return data
  },

  getKeyActionStatus: async (id: string) => {
    const { data } = await api.get<KeyActionStatus>(`/experiments/${id}/key-actions/status`)
    return data
  },

  getKeyActionResults: async (id: string) => {
    const { data } = await api.get<KeyActionResults>(`/experiments/${id}/key-actions/results`)
    return data
  },

  getKeyActionFocusOverlayStatus: async (id: string) => {
    const { data } = await api.get<Record<string, unknown>>(`/experiments/${id}/key-actions/focus-overlay/status`)
    return data
  },

  renderKeyActionFocusOverlay: async (id: string, request?: { source?: 'rows' | 'model'; views?: string[]; force?: boolean; max_hold_sec?: number }) => {
    const { data } = await api.post<Record<string, unknown>>(`/experiments/${id}/key-actions/focus-overlay/render`, request || { source: 'rows' })
    invalidateExperimentCache(id)
    return data
  },

  getKeyActionReviewQueue: (id: string, options?: ApiCacheOptions) => cachedGet<KeyActionReviewQueue>(`/experiments/${id}/key-actions/review-queue`, undefined, { ttlMs: 15_000, ...options }),

  decideKeyActionReviewItem: async (id: string, itemId: string, request: { decision: string; reviewer?: string; note?: string; boundary_start_sec?: number | null; boundary_end_sec?: number | null }) => {
    const { data } = await api.post<{ queue: KeyActionReviewQueue }>(`/experiments/${id}/key-actions/review/items/${encodeURIComponent(itemId)}/decision`, request)
    invalidateExperimentCache(id)
    return data
  },

  bulkDecideKeyActionReviewItems: async (id: string, request: { item_ids?: string[]; decision: string; reviewer?: string; note?: string }) => {
    const { data } = await api.post<{ queue: KeyActionReviewQueue }>(`/experiments/${id}/key-actions/review/bulk`, request)
    invalidateExperimentCache(id)
    return data
  },

  exportKeyActionReview: async (id: string) => {
    const { data } = await api.get<Record<string, unknown>>(`/experiments/${id}/key-actions/review/export`)
    return data
  },

  freezeKeyActionReviewedDataset: async (id: string) => {
    const { data } = await api.post<{ queue: KeyActionReviewQueue; manifest: Record<string, unknown>; release?: Record<string, unknown> | null; reviewed_export: Record<string, unknown> }>(`/experiments/${id}/key-actions/review/freeze`)
    invalidateExperimentCache(id)
    return data
  },

  rollbackKeyActionReviewedRelease: async (id: string, version?: string) => {
    const { data } = await api.post<{ queue: KeyActionReviewQueue; rollback: Record<string, unknown>; reviewed_export: Record<string, unknown> }>(`/experiments/${id}/key-actions/review/rollback`, { version })
    invalidateExperimentCache(id)
    return data
  },

  promoteKeyActionReviewedRelease: async (id: string, request?: { version?: string; reviewer?: string; note?: string; query_count?: number }) => {
    const { data } = await api.post<{ queue: KeyActionReviewQueue; promotion: Record<string, unknown>; reviewed_export: Record<string, unknown> }>(`/experiments/${id}/key-actions/review/promote`, request || {})
    invalidateExperimentCache(id)
    return data
  },

  getKeyActionEvidenceAdapters: (id: string, options?: ApiCacheOptions) => cachedGet<KeyActionEvidenceAdapters>(`/experiments/${id}/key-actions/evidence/adapters`, undefined, { ttlMs: 30_000, ...options }),

  evaluateKeyActionRetrieval: async (id: string, queryCount = 50) => {
    const { data } = await api.post<KeyActionRetrievalEvaluation>(`/experiments/${id}/key-actions/retrieval/evaluate`, { query_count: queryCount })
    invalidateExperimentCache(id)
    return data
  },

  queryKeyActions: async (id: string, query: string, topK = 5, options?: { indexLevel?: 'segment' | 'micro_segment' | 'all'; primaryObject?: string; interactionType?: string }) => {
    const { data } = await api.post<{ experiment_id: string; query: string; results: Array<Record<string, unknown>> }>(`/experiments/${id}/key-actions/query`, {
      query,
      top_k: topK,
      index_level: options?.indexLevel ?? 'all',
      primary_object: options?.primaryObject ?? null,
      interaction_type: options?.interactionType ?? null,
    })
    return data
  },
}

export const workspaceMaterialApi = {
  getPublishedMaterials: async (params?: { text?: string; event_type?: string; canonical_action_type?: string; canonical_object?: string; sop_phase?: string; actor_name?: string; limit?: number; cursor?: string; sort_by?: string; sort_order?: string }) => {
    const { data } = await api.get<WorkspacePublishedMaterialsResponse>('/materials/published', {
      params,
      headers: workspaceOperatorHeaders(),
    })
    return data
  },

  getPublishedHealth: async (autoRebuild = true) => {
    const { data } = await api.get<WorkspacePublishedHealthResponse>('/materials/published/health', {
      params: { auto_rebuild: autoRebuild },
      headers: workspaceOperatorHeaders(),
    })
    return data
  },
}
