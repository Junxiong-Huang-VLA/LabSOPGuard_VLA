import axios from 'axios'
import type {
  AnalysisOverview,
  AsrJobResponse,
  AsrUploadResponse,
  ClipBackfillResponse,
  CreateExperimentRequest,
  Experiment,
  ExperimentTaskStatus,
  KeyActionResults,
  KeyActionReviewQueue,
  KeyActionEvidenceAdapters,
  KeyActionRetrievalEvaluation,
  KeyActionStatus,
  MaterialCandidatesResponse,
  MaterialDiagnosticsResponse,
  MaterialHealthResponse,
  MaterialSearchResponse,
  MaterialTimelineResponse,
  StepRecord,
  UpdateStepRequest,
  WorkspacePublishedHealthResponse,
  WorkspacePublishedMaterialsResponse,
} from './types'

type ApiCacheOptions = { ttlMs?: number; force?: boolean }
type PublishedMaterialsResponse = { schema_version?: string; experiment_id: string; total: number; returned?: number; items: unknown[] }
type KeyActionUploadInputs = {
  firstPersonVideo?: File | null
  thirdPersonVideo?: File | null
  topVideo?: File | null
  bottomVideo?: File | null
  sessionStartTime?: string | null
}

const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' },
})

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
  const promise = api.get<T>(url, { params }).then(({ data }) => {
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
  getJson: (id: string, options?: ApiCacheOptions) => cachedGet<Record<string, unknown>>(`/experiments/${id}/analysis-overview`, undefined, { ttlMs: 15_000, ...options }),

  uploadVideo: async (id: string, file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    const { data } = await api.post(`/experiments/${id}/upload/video`, formData, { headers: { 'Content-Type': 'multipart/form-data' } })
    invalidateExperimentCache(id)
    return data
  },

  uploadAndAnalyze: async (id: string, files: KeyActionUploadInputs, onProgress?: (percent: number) => void) => {
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
  getPublishedMaterials: (id: string, params?: { limit?: number }, options?: ApiCacheOptions) => cachedGet<PublishedMaterialsResponse>(`/experiments/${id}/materials/published`, params, { ttlMs: 60_000, ...options }),
  peekPublishedMaterials: (id: string, params?: { limit?: number }) => readCachedGet<PublishedMaterialsResponse>(`/experiments/${id}/materials/published`, params),
  getMaterialCandidates: (id: string, options?: ApiCacheOptions) => cachedGet<MaterialCandidatesResponse>(`/experiments/${id}/materials/candidates`, undefined, { ttlMs: 30_000, ...options }),

  approveMaterialCandidate: async (id: string, candidateGroupId: string, request?: { reviewer?: string; notes?: string; candidate_ids?: string[]; selected_keyframe_ids?: string[]; selected_clip_ids?: string[] }) => {
    const { data } = await api.post(`/experiments/${id}/materials/candidates/${candidateGroupId}/approve`, request || {})
    invalidateExperimentCache(id)
    return data
  },

  decideMaterialCandidate: async (id: string, candidateGroupId: string, request: { decision: string; reviewer?: string; reason_code?: string; reason?: string; notes?: string; candidate_ids?: string[] }) => {
    const { data } = await api.post(`/experiments/${id}/materials/candidates/${candidateGroupId}/decision`, request)
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
  getPublishedMaterials: async (params?: { text?: string; event_type?: string; actor_name?: string; limit?: number; cursor?: string; sort_by?: string; sort_order?: string }) => {
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
