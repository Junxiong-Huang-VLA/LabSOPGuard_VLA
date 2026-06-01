import { ChangeEvent, useEffect, useMemo, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { ArrowLeft, CheckCircle2, FileText, RefreshCw, Search } from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import {
  EmptyEvidence,
  EvidenceBadge,
  EvidenceCard,
  PageHero,
  primaryButtonClass,
  secondaryButtonClass,
  toneForStatus,
} from '../components/EvidenceUI'
import ExperimentPageShell from '../components/ExperimentSideNav'
import { cleanDisplayText } from '../displayText'
import { experimentFileUrl } from '../mediaUrl'
import {
  type DemoCuratedMaterial,
  getDemoCuratedMaterials,
  getDemoMaterialCount,
  isDemoExperiment,
} from '../demo/weighingPipettingDemo'
import type {
  AnalysisOverview,
  MaterialCandidateFile,
  MaterialCandidateGroup,
  MaterialDiagnosticsResponse,
  MaterialSearchItem,
} from '../types'

const REVIEW_MODE_VALUES = new Set(['1', 'true', 'yes'])
const KEYFRAME_UNDERSTANDING_TITLE = '关键帧理解'
const KEYCLIP_UNDERSTANDING_TITLE = '关键片段理解'
const KEYFRAME_UNDERSTANDING_MISSING = '关键帧理解待生成'
const KEYCLIP_UNDERSTANDING_MISSING = '关键片段理解待生成'
const UNDERSTANDING_FACTS_LABEL = '可见事实'
const UNDERSTANDING_ACTION_LABEL = '动作理解'
const UNDERSTANDING_UNCERTAINTIES_LABEL = '不确定性'
const UNDERSTANDING_EVIDENCE_LABEL = '证据引用'
const UNDERSTANDING_EMPTY_VALUE_TEXT = '暂无'

type MaterialUnderstanding = {
  visibleFacts: string[]
  actionInterpretation: string[]
  uncertainties: string[]
  evidenceRefs: string[]
}

type AlignmentGate = {
  status?: string
  reason?: string
  hidden_item_count?: number
}

type PublishedMaterialsResponse = {
  experiment_id: string
  total: number
  returned?: number
  items: MaterialSearchItem[]
  all_items?: MaterialSearchItem[]
  alignment_gate?: AlignmentGate
  grouped_items?: unknown[]
}

function finiteNumber(value?: unknown) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function safeText(value?: unknown, ...fallbacks: unknown[]) {
  for (const candidate of [value, ...fallbacks]) {
    if (candidate == null) continue
    const text = String(candidate).trim()
    if (text) return text
  }
  return ''
}

function formatHHMMSS(value?: number | null) {
  const seconds = finiteNumber(value)
  if (seconds == null) return '-'
  const integerSeconds = Math.max(0, Math.floor(seconds))
  const h = Math.floor(integerSeconds / 3600)
  const m = Math.floor((integerSeconds % 3600) / 60)
  const s = integerSeconds % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function formatRangeText(start?: unknown, end?: unknown) {
  const startSec = finiteNumber(start)
  const endSec = finiteNumber(end)
  if (startSec == null) return '-'
  return `${formatHHMMSS(startSec)} 至 ${endSec == null ? '-' : formatHHMMSS(endSec)}`
}

function materialTimestampText(item: MaterialSearchItem) {
  return `时间戳：${formatRangeText(item.time_start, item.time_end)}`
}

function materialSourceSegmentText(item: MaterialSearchItem) {
  const raw = safeText(item.experiment_window_id, item.segment_id, item.parent_segment_id)
  if (!raw) return '未分配'
  const match = raw.match(/(?:episode|formal_window|seg)_(\d+)/i) || raw.match(/(\d+)$/)
  if (!match) return raw
  return `实验片段 ${Number(match[1])}`
}

const MATERIAL_LABEL_ZH: Record<string, string> = {
  'hand-paper': '手部与称量纸操作',
  'hand-bottle': '手部与试剂瓶操作',
  'hand-balance': '天平设备面板操作',
  'hand-spatula': '手部与药匙操作',
  'hand-container': '手部与容器操作',
  hand_paper_contact: '手部与称量纸接触',
  hand_bottle_contact: '手部与试剂瓶接触',
  hand_balance_contact: '天平设备面板操作',
  hand_spatula_contact: '手部与药匙接触',
  hand_container_contact: '手部与容器接触',
  weighing_paper_operation: '手部与称量纸操作',
  balance_operation: '天平设备面板操作',
  pipetting_operation: '移液操作',
  bottle_operation: '试剂瓶操作',
  liquid_transfer: '液体转移操作',
  liquid_transfer_candidate: '液体转移候选操作',
  object_movement_detected: '物体移动',
  object_movement_candidate: '物体移动候选',
  container_state_change: '容器状态变化',
  hand_object_contact: '手物接触',
  paper: '称量纸',
  reagent_bottle: '试剂瓶',
  balance: '天平',
  spatula: '药匙',
  container: '容器',
  beaker: '烧杯',
  pipette: '移液器',
  tube: '试管',
  sample_bottle: '样品瓶',
}

function zhMaterialLabel(value?: unknown) {
  const text = safeText(value)
  if (!text) return ''
  return MATERIAL_LABEL_ZH[text] || MATERIAL_LABEL_ZH[text.toLowerCase()] || cleanDisplayText(text, text)
}

function materialTitle(item: MaterialSearchItem) {
  const display = cleanDisplayText(safeText(item.display_title, item.display_name), '')
  if (display) return display
  const action = zhMaterialLabel(safeText(item.canonical_action_type, item.semantic_action, item.interaction_family))
  const object = zhMaterialLabel(safeText(item.canonical_object, item.primary_object, item.raw_primary_object, item.yolo_primary_object))
  return cleanDisplayText([action, object].filter(Boolean).join(' / '), safeText(item.item_id))
}

function materialAssetUrl(item: MaterialSearchItem, experimentId: string | undefined, keys: string[]) {
  const record = item as Record<string, unknown>
  for (const key of keys) {
    const raw = safeText(record[key], '')
    if (raw) return experimentFileUrl(raw, experimentId)
  }
  return ''
}

function materialConfidence(item: MaterialSearchItem) {
  const raw = finiteNumber(item.quality_score || item.keyframe_quality_score || item.best_score)
  if (raw == null) return '-'
  return raw.toFixed(2)
}

void materialConfidence

function statusLabelZh(value?: unknown) {
  const raw = safeText(value, 'pending').toLowerCase()
  return {
    accepted: '已接收',
    approved: '已批准',
    confirmed: '已确认',
    confirmed_best: '已确认最佳',
    needs_review: '待确认',
    pending: '待确认',
    unreviewed: '未审核',
    rejected: '已拒绝',
    blocked: '已阻断',
    completed: '已完成',
    analyzed: '已分析',
    partial_failed: '部分失败',
    failed: '失败',
    running: '运行中',
    analyzing: '分析中',
    queued: '排队中',
  }[raw] || cleanDisplayText(value, '待确认')
}

void statusLabelZh

function collectEvidenceRows(item: MaterialSearchItem) {
  const rows: Array<{ label: string; value: string }> = []
  const extraRecord = item as Record<string, unknown>

  const append = (label: string, value: unknown) => {
    const text = cleanDisplayText(value, '')
    if (!text) return
    rows.push({ label, value: text })
  }

  append('YOLO标注渲染', item.yolo_annotation_rendered)
  append('语义动作', item.semantic_action)
  append('动作类型', item.canonical_action_type)
  append('相关对象', item.canonical_object)
  append('物理动作类型', item.physical_action_type)
  append('仪器上下文', item.instrument_context)
  append('证据包ID', item.evidence_bundle_id)
  append('窗口同步索引', item.source_window_sync_index)
  append('review_route', item.review_route)

  if (item.vlm_semantics && typeof item.vlm_semantics === 'object') {
    append('VLM状态', (item.vlm_semantics as Record<string, unknown>).status)
  }

  append('VLM推理', (item as Record<string, unknown>).vlm_semantics)

  for (const [key, value] of Object.entries(extraRecord)) {
    if (typeof key !== 'string') continue
    if (key.startsWith('__')) continue
    if (key.startsWith('view_action_review_group_')) {
      append('候选视图分组', value)
      continue
    }
    if ([
      'yolo_annotation_rendered',
      'semantic_action',
      'canonical_action_type',
      'canonical_object',
      'physical_action_type',
      'instrument_context',
      'evidence_bundle_id',
      'source_window_sync_index',
      'review_route',
      'vlm_semantics',
      'time_start',
      'time_end',
      'item_id',
      'display_title',
      'display_name',
      'event_id',
      'experiment_id',
      'experiment_window_id',
      'segment_id',
      'window_id',
      'material_id',
      'view',
      'camera_view',
      'event_type',
      'review_status',
      'orphan_material',
      'first_keyframe',
      'third_keyframe',
      'first_keyclip',
      'third_keyclip',
      'side_by_side_keyclip',
      'first_frame',
      'third_frame',
      'first_clip',
      'third_clip',
      'timestamp_sec',
      'local_timestamp_sec',
      'quality_score',
      'best_score',
      'keyframe_quality_score',
    ].includes(key)) {
      continue
    }
    append(key, value)
  }

  return rows
}

function toTextList(value: unknown) {
  const rows: string[] = []
  toTextValues(value, rows)
  return rows.filter(Boolean)
}

function toTextValues(value: unknown, out: string[], depth = 0) {
  if (value == null || depth > 6) return
  if (typeof value === 'string') {
    const clean = cleanDisplayText(value, '')
    if (clean) out.push(clean)
    return
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    out.push(String(value))
    return
  }
  if (Array.isArray(value)) {
    value.forEach(item => toTextValues(item, out, depth + 1))
    return
  }
  if (typeof value === 'object') {
    for (const item of Object.values(value as Record<string, unknown>)) {
      toTextValues(item, out, depth + 1)
    }
  }
}

function toEvidenceList(value: unknown) {
  if (value == null) return []
  const rows: string[] = []
  toTextValues(value, rows)
  return rows.filter(Boolean)
}

function parseMaterialUnderstanding(raw: unknown): MaterialUnderstanding {
  if (!raw || typeof raw !== 'object') {
    return {
      visibleFacts: toTextList(raw),
      actionInterpretation: [],
      uncertainties: [],
      evidenceRefs: [],
    }
  }

  const source = raw as Record<string, unknown>
  return {
    visibleFacts: toTextList(source.visible_facts ?? source.observed_facts ?? source.facts),
    actionInterpretation: toTextList(
      source.action_interpretation
      ?? source.inferred_steps
      ?? source.interpretation
      ?? source.summary
      ?? source.key_actions_summary
      ?? source.visual_scene_summary,
    ),
    uncertainties: toTextList(source.uncertainties ?? source.unresolved_questions ?? source.uncertainty ?? source.unknown_points),
    evidenceRefs: toEvidenceList(source.evidence_refs ?? source.evidence_references ?? source.refs ?? source.evidence),
  }
}

function resolveMaterialUnderstanding(item: MaterialSearchItem, kind: 'keyframe' | 'keyclip') {
  const record = item as Record<string, unknown>
  const payload = item.payload && typeof item.payload === 'object' ? item.payload as Record<string, unknown> : {}
  const candidates = [
    `${kind}_understanding`,
    `${kind}_vlm_understanding`,
    `vlm_${kind}_understanding`,
    `${kind}_understanding_payload`,
    `${kind}_vlm_payload`,
  ]

  for (const key of candidates) {
    const raw = record[key] ?? payload[key]
    if (raw == null) continue
    const parsed = parseMaterialUnderstanding(raw)
    if (parsed.visibleFacts.length || parsed.actionInterpretation.length || parsed.uncertainties.length || parsed.evidenceRefs.length) {
      return parsed
    }
  }

  if (record[`${kind}_understanding_text`]) {
    return {
      visibleFacts: [],
      actionInterpretation: toTextList(record[`${kind}_understanding_text`]),
      uncertainties: [],
      evidenceRefs: [],
    }
  }

  return null
}

function MaterialUnderstandingPanel({
  title,
  missingText,
  understanding,
  dataSmoke,
}: {
  title: string
  missingText: string
  understanding: MaterialUnderstanding | null
  dataSmoke?: string
}) {
  const normalized = understanding || {
    visibleFacts: [],
    actionInterpretation: [],
    uncertainties: [],
    evidenceRefs: [],
  }
  const noContent = !normalized.visibleFacts.length && !normalized.actionInterpretation.length && !normalized.uncertainties.length && !normalized.evidenceRefs.length

  return (
    <section
      className="rounded-md border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] p-3 text-xs text-[color:var(--ui-text-muted)]"
      data-smoke={dataSmoke}
    >
      <h4 className="mb-2 font-semibold text-[color:var(--ui-text)]">{title}</h4>
      {understanding == null || noContent ? (
        <p>{missingText}</p>
      ) : (
        <div className="space-y-2">
          <div>
            <div className="text-[color:var(--ui-text)]">{UNDERSTANDING_FACTS_LABEL}</div>
            {normalized.visibleFacts.length ? (
              <ul className="ml-4 list-disc">
                {normalized.visibleFacts.map((item, index) => (
                  <li key={`${title}-facts-${index}`}>{item}</li>
                ))}
              </ul>
            ) : (
              <div>{UNDERSTANDING_EMPTY_VALUE_TEXT}</div>
            )}
          </div>
          <div>
            <div className="text-[color:var(--ui-text)]">{UNDERSTANDING_ACTION_LABEL}</div>
            {normalized.actionInterpretation.length ? (
              <ul className="ml-4 list-disc">
                {normalized.actionInterpretation.map((item, index) => (
                  <li key={`${title}-action-${index}`}>{item}</li>
                ))}
              </ul>
            ) : (
              <div>{UNDERSTANDING_EMPTY_VALUE_TEXT}</div>
            )}
          </div>
          <div>
            <div className="text-[color:var(--ui-text)]">{UNDERSTANDING_UNCERTAINTIES_LABEL}</div>
            {normalized.uncertainties.length ? (
              <ul className="ml-4 list-disc">
                {normalized.uncertainties.map((item, index) => (
                  <li key={`${title}-uncertainties-${index}`}>{item}</li>
                ))}
              </ul>
            ) : (
              <div>{UNDERSTANDING_EMPTY_VALUE_TEXT}</div>
            )}
          </div>
          <div>
            <div className="text-[color:var(--ui-text)]">{UNDERSTANDING_EVIDENCE_LABEL}</div>
            {normalized.evidenceRefs.length ? (
              <ul className="ml-4 list-disc">
                {normalized.evidenceRefs.map((item, index) => (
                  <li key={`${title}-evidence-${index}`}>{item}</li>
                ))}
              </ul>
            ) : (
              <div>{UNDERSTANDING_EMPTY_VALUE_TEXT}</div>
            )}
          </div>
        </div>
      )}
    </section>
  )
}
function DemoCuratedMaterialCard({ material }: { material: DemoCuratedMaterial }) {
  return (
    <EvidenceCard className="overflow-hidden" data-smoke="key-material-pair">
      <div className="border-b border-[color:var(--ui-border)] p-3">
        <div className="mb-1 flex flex-wrap items-center gap-2">
          <EvidenceBadge tone="emerald">{material.actionLabel}</EvidenceBadge>
          <EvidenceBadge tone="slate">双视角已对齐</EvidenceBadge>
        </div>
        <div className="text-sm font-semibold text-[color:var(--ui-text)]">
          {material.actionLabel} · {material.objectLabel}
        </div>
        <div className="mt-1 text-xs text-[color:var(--ui-text-muted)]">
          来源：{material.segmentName} · 时间戳 {material.timeRangeText}
        </div>
      </div>

      <div className="grid gap-3 p-3 md:grid-cols-2">
        <section>
          <div className="text-xs font-bold text-[color:var(--ui-text-muted)]">第一人称关键片段</div>
          <div className="mt-1 aspect-video overflow-hidden rounded-md border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)]">
            <video
              src={experimentFileUrl(material.firstVideoUrl, undefined)}
              poster={experimentFileUrl(material.firstKeyframeUrl, undefined)}
              controls
              preload="none"
              playsInline
              className="h-full w-full object-cover"
            />
          </div>
        </section>
        <section>
          <div className="text-xs font-bold text-[color:var(--ui-text-muted)]">第三人称关键片段</div>
          <div className="mt-1 aspect-video overflow-hidden rounded-md border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)]">
            <video
              src={experimentFileUrl(material.thirdVideoUrl, undefined)}
              poster={experimentFileUrl(material.thirdKeyframeUrl, undefined)}
              controls
              preload="none"
              playsInline
              className="h-full w-full object-cover"
            />
          </div>
        </section>
      </div>
    </EvidenceCard>
  )
}

function DemoMaterialLibrary({ experimentName }: { experimentName: string }) {
  const materials = getDemoCuratedMaterials()
  const totalMaterialCount = getDemoMaterialCount()
  return (
    <div className="space-y-6">
      <PageHero
        eyebrow={<Link to="/experiments" className="hover:text-[color:var(--ui-text)]">实验列表</Link>}
        title={experimentName}
        description="精选高置信度双视角交互动作，已标注所属实验片段。"
        actions={
          <Link to="/experiments" className={secondaryButtonClass()}>
            <ArrowLeft className="h-4 w-4" />
            返回实验列表
          </Link>
        }
      />
      <section data-smoke="formal-material-library" data-count={materials.length}>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-base font-semibold text-[color:var(--ui-text)]">关键素材</h3>
          <span className="text-xs font-semibold text-[color:var(--ui-text-muted)]">共 {totalMaterialCount} 个关键素材</span>
        </div>
        <div className="grid gap-4 xl:grid-cols-2">
          {materials.map(material => (
            <DemoCuratedMaterialCard key={material.id} material={material} />
          ))}
        </div>
      </section>
    </div>
  )
}

export default function MaterialSearch() {
  const { id } = useParams<{ id: string }>()
  const location = useLocation()
  const searchParams = new URLSearchParams(location.search)
  const reviewMode = REVIEW_MODE_VALUES.has(String(searchParams.get('review') || '').toLowerCase())

  const [overview, setOverview] = useState<AnalysisOverview | null>(null)
  const [, setWindows] = useState<Array<Record<string, unknown>>>([])
  const [materials, setMaterials] = useState<MaterialSearchItem[]>([])
  const [allMaterials, setAllMaterials] = useState<MaterialSearchItem[]>([])
  const [useAllMaterials, setUseAllMaterials] = useState(false)
  const [, setAlignmentGate] = useState<AlignmentGate | null>(null)
  const [diagnostics, setDiagnostics] = useState<MaterialDiagnosticsResponse | null>(null)
  const [candidates, setCandidates] = useState<MaterialCandidateGroup[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  void setShowAdvanced
  const [evidenceOpenById, setEvidenceOpenById] = useState<Record<string, boolean>>({})
  const [busyAction, setBusyAction] = useState<Record<string, string>>({})
  const [renaming, setRenaming] = useState<Record<string, boolean>>({})

  async function load(force = false) {
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const [nextOverview, nextWindows, nextPublished, nextCandidates, nextDiagnostics] = await Promise.all([
        experimentApi.getAnalysisOverview(id, { force }),
        experimentApi.getSubExperiments(id, { force }).catch(() => null),
        experimentApi.getPublishedMaterials(id, { limit: 500 }, { force }).catch(() => ({
          experiment_id: id,
          total: 0,
          items: [],
        }) as PublishedMaterialsResponse),
        experimentApi.getMaterialCandidates(id, { force }).catch(() => ({
          experiment_id: id,
          total: 0,
          items: [],
        }) as unknown as { experiment_id: string; total: number; items: MaterialCandidateGroup[] }),
        experimentApi.getMaterialDiagnostics(id).catch(() => null),
      ])

      setOverview(nextOverview)

      setWindows(
        Array.isArray((nextWindows as { segments?: unknown[] } | null)?.segments)
          ? ((nextWindows as unknown as { segments: Record<string, unknown>[] }).segments)
          : [],
      )

      const published = nextPublished as PublishedMaterialsResponse
      const hasAllItems = Object.prototype.hasOwnProperty.call(published, 'all_items')
      const sourceItems = Array.isArray((published as { items?: unknown[] }).items)
        ? (published as { items: MaterialSearchItem[] }).items
        : []
      const allItems = hasAllItems && Array.isArray((published as { all_items?: unknown[] }).all_items)
        ? (published as { all_items: MaterialSearchItem[] }).all_items
        : []

      setMaterials(sourceItems.filter(item => item.review_status !== 'rejected'))
      setAllMaterials(hasAllItems ? allItems : [])
      setUseAllMaterials(hasAllItems)
      setAlignmentGate(published.alignment_gate ? {
        status: published.alignment_gate.status,
        reason: published.alignment_gate.reason,
        hidden_item_count: published.alignment_gate.hidden_item_count,
      } : null)

      setDiagnostics(nextDiagnostics as MaterialDiagnosticsResponse | null)

      setCandidates(
        Array.isArray((nextCandidates as { items?: unknown[] }).items)
          ? (nextCandidates as { items: MaterialCandidateGroup[] }).items
          : [],
      )
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Failed to load material page')
      setWindows([])
      setMaterials([])
      setAllMaterials([])
      setUseAllMaterials(false)
      setAlignmentGate(null)
      setDiagnostics(null)
      setCandidates([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load(true)
    void prefetchExperimentRoute(id || '', 'materials')
  }, [id, location.search])

  const experimentId = id || ''
  const formalBase = useAllMaterials ? allMaterials : materials

  const formalMaterials = useMemo(() => {
    return formalBase
  }, [formalBase])

  const filteredFormalMaterials = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    if (!keyword) return formalMaterials
    return formalMaterials.filter(item => {
      const source = [
        item.display_title,
        item.display_name,
        item.item_id,
        item.canonical_action_type,
        item.canonical_object,
        item.event_type,
        item.experiment_window_id,
      ]
        .filter(Boolean)
        .map(itemValue => String(itemValue).toLowerCase())
        .join(' ')
      return source.includes(keyword)
    })
  }, [formalMaterials, query])

  const keyframeCount = filteredFormalMaterials.filter(item => Boolean(item.first_keyframe || item.third_keyframe)).length
  const keyclipCount = filteredFormalMaterials.filter(item => Boolean(item.first_keyclip || item.third_keyclip || item.side_by_side_keyclip)).length
  const pendingReviewCount = filteredFormalMaterials.filter(item => safeText(item.review_status).toLowerCase().includes('needs_review')).length
  const confirmedCount = filteredFormalMaterials.filter(item => safeText(item.review_status).toLowerCase() === 'confirmed').length

  const filteredCandidates = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    if (!keyword) return candidates
    return candidates.filter(item => {
      const source = [
        item.display_title,
        item.candidate_group_id,
        item.parent_segment_id,
        item.interaction_family,
        item.canonical_action_type,
        item.action_name,
      ]
        .filter(Boolean)
        .map(itemValue => String(itemValue).toLowerCase())
        .join(' ')
      return source.includes(keyword)
    })
  }, [candidates, query])

  async function performCandidateAction(groupId: string, action: 'confirm' | 'upgrade' | 'reject' | 'rename', payload?: { display_title?: string }) {
    if (!id) return
    setBusyAction(previous => ({ ...previous, [groupId]: action }))
    try {
      if (action === 'confirm') {
        await experimentApi.confirmMaterialCandidate(id, groupId)
      } else if (action === 'upgrade') {
        await experimentApi.approveMaterialCandidate(id, groupId)
      } else if (action === 'reject') {
        await experimentApi.decideMaterialCandidate(id, groupId, {
          decision: 'false_positive',
          reason_code: 'rejected',
          reason: '鐢ㄦ埛纭璇ョ礌鏉愪笉浣滀负鍏抽敭绱犳潗',
        })
      } else if (action === 'rename' && payload?.display_title) {
        await experimentApi.renameMaterialCandidate(id, groupId, { display_title: payload.display_title })
      }
      await load(true)
    } finally {
      setBusyAction(previous => {
        const next = { ...previous }
        delete next[groupId]
        return next
      })
      setRenaming(previous => {
        const next = { ...previous }
        delete next[groupId]
        return next
      })
    }
  }

  function onRenameCandidate(candidate: MaterialCandidateGroup) {
    const groupId = candidate.candidate_group_id || candidate.parent_segment_id || 'candidate-material'
    const current = safeText(candidate.display_title, candidate.interaction_family, candidate.canonical_action_type, candidate.action_name)
    const next = window.prompt('请输入新的素材名称', current)
    if (!next?.trim()) return
    setRenaming(previous => ({ ...previous, [groupId]: true }))
    void performCandidateAction(groupId, 'rename', { display_title: next.trim() })
  }

  function resolveCandidateActionLabel(candidate: MaterialCandidateGroup) {
    const label = safeText(
      candidate.display_title,
      candidate.canonical_action_type,
      candidate.action_name,
      candidate.semantic_action,
      candidate.interaction_family,
      '待确认素材',
    )
    return zhMaterialLabel(label) || '待确认素材'
  }

  function resolveCandidateWindow(candidate: MaterialCandidateGroup) {
    return safeText(candidate.parent_segment_id, candidate.micro_segment_id, candidate.candidate_group_id, '未关联实验片段')
  }

  function resolveCandidateTime(candidate: MaterialCandidateGroup) {
    const allFiles: Array<MaterialCandidateFile | undefined> = [...candidate.keyframes, ...candidate.clips, ...candidate.files]
    const media = allFiles.find(item => Boolean(item))
    if (!media) return '-'
    return formatRangeText(media.timestamp_sec, media.time_start)
  }

  function resolveCandidateAsset(candidate: MaterialCandidateGroup) {
    const candidateFiles: Array<MaterialCandidateFile | undefined> = [...candidate.keyframes, ...candidate.clips, ...candidate.files]
    const file = candidateFiles.find(Boolean) as MaterialCandidateFile | undefined
    if (!file) return ''
    return experimentFileUrl(
      safeText(file.clip_url) || safeText(file.clip_file_path) || safeText(file.preview_url) || safeText(file.frame_path) || safeText(file.url),
      experimentId,
    )
  }

  if (isDemoExperiment(experimentId)) {
    return (
      <ExperimentPageShell experimentId={experimentId}>
      <DemoMaterialLibrary
        experimentName={overview ? cleanDisplayText(overview.experiment.experiment_name, '称量移液实验') : '称量移液实验'}
      />
      </ExperimentPageShell>
    )
  }

  return (
    <ExperimentPageShell experimentId={experimentId}>
    <div className="space-y-6">
      <PageHero
        eyebrow={<Link to="/experiments" className="hover:text-[color:var(--ui-text)]">实验列表</Link>}
        title={overview ? cleanDisplayText(overview.experiment.experiment_name, `实验 ${experimentId}`) : '实验关键素材'}
        description=""
        actions={
          <>
            <Link to={`/experiments/${experimentId}/workspace`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'workspace')} className={secondaryButtonClass()}>
              <ArrowLeft className="h-4 w-4" />
              返回工作台
            </Link>
            <button type="button" onClick={() => void load(true)} className={secondaryButtonClass()}>
              <RefreshCw className="h-4 w-4" />
              刷新
            </button>
          </>
        }
      />

      {loading && <EmptyEvidence title="正在加载" description="正在读取关键素材..." />}
      {error && <EmptyEvidence title="加载失败" description={error} />}

      {!loading && !error && (
        <>
          <section className="grid gap-3">
            <div className="mb-2">
              <h2 className="text-lg font-semibold text-[color:var(--ui-text)]">关键素材</h2>
            </div>

              <section data-smoke="formal-material-library" data-count={filteredFormalMaterials.length}>
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="text-base font-semibold text-[color:var(--ui-text)]">素材列表</h3>
                </div>

                <label className="mb-3 inline-flex items-center gap-2 rounded-md border border-[color:var(--ui-border)] px-3 py-1.5 text-xs text-[color:var(--ui-text-muted)]">
                  <Search className="h-3.5 w-3.5" />
                  <input
                    value={query}
                    onChange={(event: ChangeEvent<HTMLInputElement>) => setQuery(event.target.value)}
                    placeholder="按实验片段、动作或物体筛选"
                    className="h-7 border-0 bg-transparent p-0 text-xs text-[color:var(--ui-text)] placeholder:text-[color:var(--ui-text-muted)] outline-none"
                  />
                </label>

                {filteredFormalMaterials.length === 0 ? (
                  <EmptyEvidence title="暂无关键素材" description="完成分析或确认候选素材后，关键帧和关键片段会显示在这里。" />
                ) : (
                  <div className="grid gap-4 xl:grid-cols-2">
                    {filteredFormalMaterials.map(material => {
                      const uid = safeText(material.item_id, material.material_id, material.event_id, material.public_item_id, `formal-${material.experiment_window_id || 'material'}`)
                      const title = materialTitle(material)
                      const thirdKeyframe = materialAssetUrl(material, experimentId, ['third_keyframe'])
                      const firstKeyframe = materialAssetUrl(material, experimentId, ['first_keyframe'])
                      const thirdKeyclip = materialAssetUrl(material, experimentId, ['third_keyclip', 'third_clip'])
                      const firstKeyclip = materialAssetUrl(material, experimentId, ['first_keyclip', 'first_clip'])
                      const keyframeUnderstanding = resolveMaterialUnderstanding(material, 'keyframe')
                      const keyclipUnderstanding = resolveMaterialUnderstanding(material, 'keyclip')
                      const evidenceRows = collectEvidenceRows(material)
                      const showReviewDetails = !!evidenceOpenById[uid]

                      return (
                        <EvidenceCard key={uid} className="overflow-hidden" data-smoke="key-material-pair">
                          <div className="border-b border-[color:var(--ui-border)] p-3">
                            <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
                              <div>
                                <div className="text-sm font-semibold text-[color:var(--ui-text)]">{title}</div>
                                <div className="mt-1 text-xs text-[color:var(--ui-text-muted)]">{materialTimestampText(material)}</div>
                              </div>
                            </div>
                            <div className="text-xs text-[color:var(--ui-text-muted)]">
                              来源：{materialSourceSegmentText(material)}
                            </div>
                          </div>

                          <div className="grid gap-3 p-3 md:grid-cols-2">
                            <section>
                              <div className="text-xs font-bold text-[color:var(--ui-text-muted)]">第一人称关键帧</div>
                              <div className="mt-1 h-44 overflow-hidden rounded-md border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] sm:h-52">
                                {firstKeyframe
                                  ? <img src={firstKeyframe} alt="第一人称关键帧" className="h-full w-full object-cover" />
                                  : <div className="flex h-full items-center justify-center text-xs text-[color:var(--ui-text-muted)]">第一人称关键帧待生成</div>}
                              </div>
                            </section>
                            <section>
                              <div className="text-xs font-bold text-[color:var(--ui-text-muted)]">第三人称关键帧</div>
                              <div className="mt-1 h-44 overflow-hidden rounded-md border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] sm:h-52">
                                {thirdKeyframe
                                  ? <img src={thirdKeyframe} alt="第三人称关键帧" className="h-full w-full object-cover" />
                                  : <div className="flex h-full items-center justify-center text-xs text-[color:var(--ui-text-muted)]">第三人称关键帧待生成</div>}
                              </div>
                            </section>
                          </div>
                          <div className="grid gap-3 p-3 md:grid-cols-2">
                            <section>
                              <div className="text-xs font-bold text-[color:var(--ui-text-muted)]">第一人称关键片段</div>
                              <div className="mt-1 h-44 overflow-hidden rounded-md border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] sm:h-52">
                                {firstKeyclip
                                  ? <video src={firstKeyclip} controls className="h-full w-full object-cover" />
                                  : <div className="flex h-full items-center justify-center text-xs text-[color:var(--ui-text-muted)]">第一人称关键片段待生成</div>}
                              </div>
                            </section>
                            <section>
                              <div className="text-xs font-bold text-[color:var(--ui-text-muted)]">第三人称关键片段</div>
                              <div className="mt-1 h-44 overflow-hidden rounded-md border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] sm:h-52">
                                {thirdKeyclip
                                  ? <video src={thirdKeyclip} controls className="h-full w-full object-cover" />
                                  : <div className="flex h-full items-center justify-center text-xs text-[color:var(--ui-text-muted)]">第三人称关键片段待生成</div>}
                              </div>
                            </section>
                          </div>

                          <div className="grid gap-3 p-3 md:grid-cols-2">
                            <MaterialUnderstandingPanel
                              dataSmoke="material-keyframe-understanding"
                              title={KEYFRAME_UNDERSTANDING_TITLE}
                              missingText={KEYFRAME_UNDERSTANDING_MISSING}
                              understanding={keyframeUnderstanding}
                            />
                            <MaterialUnderstandingPanel
                              dataSmoke="material-keyclip-understanding"
                              title={KEYCLIP_UNDERSTANDING_TITLE}
                              missingText={KEYCLIP_UNDERSTANDING_MISSING}
                              understanding={keyclipUnderstanding}
                            />
                          </div>

                          {reviewMode && (
                            <div className="border-t border-[color:var(--ui-border)] px-3 py-2">
                              <button
                                type="button"
                                onClick={() => setEvidenceOpenById(previous => ({ ...previous, [uid]: !showReviewDetails }))}
                                className="text-xs font-semibold text-[color:var(--ui-accent)]"
                              >
                                {showReviewDetails ? '收起证据详情' : '证据详情'}
                              </button>
                              {showReviewDetails ? (
                                <div className="mt-2 rounded-md border border-[color:var(--ui-border)] bg-white p-2 text-xs text-[color:var(--ui-text-muted)]">
                                  <div className="mb-2 font-semibold text-[color:var(--ui-text)]">证据详情</div>
                                  <div className="space-y-1">
                                    {evidenceRows.length === 0
                                      ? <div>暂无证据字段</div>
                                      : evidenceRows.map((row, index) => (
                                        <div key={`${uid}-${index}`}>
                                          <span className="font-semibold text-[color:var(--ui-text)]">{row.label}</span><span>: </span>{row.value}
                                        </div>
                                      ))
                                    }
                                  </div>
                                </div>
                              ) : null}
                            </div>
                          )}
                        </EvidenceCard>
                      )
                    })}
                  </div>
                )}
              </section>
          </section>

          <section>
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-[color:var(--ui-text)]">待确认关键素材</h2>
            </div>
            {filteredCandidates.length === 0 ? (
              <EmptyEvidence title="暂无待确认素材" description="当前实验没有待确认的关键素材。" />
            ) : (
              <div className="space-y-3">
                {filteredCandidates.map(group => {
                  const gid = group.candidate_group_id || group.parent_segment_id || 'candidate-group'
                  const isBusy = Boolean(busyAction[gid])
                  const mediaUrl = resolveCandidateAsset(group)
                  const confidence = Number.isFinite(finiteNumber(group.quality_score))
                    ? (finiteNumber(group.quality_score) as number).toFixed(3)
                    : '-'

                  return (
                    <EvidenceCard key={gid} className="p-4">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <EvidenceBadge tone={group.review_status ? toneForStatus(group.review_status) : 'warning'}>{resolveCandidateActionLabel(group)}</EvidenceBadge>
                            <EvidenceBadge tone="slate">待确认素材</EvidenceBadge>
                          </div>
                          <h3 className="mt-2 text-sm font-semibold text-[color:var(--ui-text)]">{safeText(group.display_title, resolveCandidateWindow(group))}</h3>
                          <p className="mt-1 text-xs text-[color:var(--ui-text-muted)]">实验片段：{resolveCandidateWindow(group)}</p>
                          <p className="mt-1 text-xs text-[color:var(--ui-text-muted)]">时间锚点：{resolveCandidateTime(group)} | 类型：{zhMaterialLabel(safeText(group.action_name, group.interaction_family, group.canonical_action_type, '-'))}</p>
                          <p className="mt-1 text-xs text-[color:var(--ui-text-muted)]">置信度：{confidence}</p>
                        </div>
                      </div>

                      <div className="mt-3 rounded-lg bg-[color:var(--ui-bg-muted)]">
                        <div className="relative aspect-video w-full overflow-hidden rounded-md">
                          {mediaUrl ? (
                            <img src={mediaUrl} alt="待确认关键素材" className="h-full w-full object-cover" />
                          ) : (
                            <div className="absolute inset-0 animate-pulse bg-gradient-to-br from-[color:var(--ui-bg-muted)] to-white" />
                          )}
                        </div>
                      </div>

                      <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => void performCandidateAction(gid, 'confirm')}
                          className={`${primaryButtonClass('success')} disabled:cursor-not-allowed disabled:opacity-50`}
                        >
                          <CheckCircle2 className="h-4 w-4" />
                          {isBusy ? '处理中...' : '确认'}
                        </button>
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => void performCandidateAction(gid, 'upgrade')}
                          className={`${secondaryButtonClass()} disabled:cursor-not-allowed disabled:opacity-50`}
                        >
                          <CheckCircle2 className="h-4 w-4" />
                          {isBusy ? '处理中...' : '加入关键素材'}
                        </button>
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => void performCandidateAction(gid, 'reject')}
                          className={`${secondaryButtonClass('warning')} disabled:cursor-not-allowed disabled:opacity-50`}
                        >
                          <FileText className="h-4 w-4" />
                          {isBusy ? '处理中...' : '拒绝'}
                        </button>
                        <button
                          type="button"
                          disabled={Boolean(renaming[gid])}
                          onClick={() => onRenameCandidate(group)}
                          className={`${secondaryButtonClass()} disabled:cursor-not-allowed disabled:opacity-50`}
                        >
                          {renaming[gid] ? '处理中...' : '重命名'}
                        </button>
                      </div>
                    </EvidenceCard>
                  )
                })}
              </div>
            )}
          </section>

          {showAdvanced && (
            <details className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/80 p-4 text-xs text-[color:var(--ui-text-muted)] shadow-[var(--ui-shadow-subtle)]">
              <summary className="cursor-pointer font-medium text-[color:var(--ui-text)]">高级信息</summary>
              <div className="mt-3 space-y-2">
                <p>关键素材：{filteredFormalMaterials.length}</p>
                <p>关键帧：{keyframeCount}</p>
                <p>关键片段：{keyclipCount}</p>
                <p>待确认：{pendingReviewCount}</p>
                <p>已确认：{confirmedCount}</p>
                <p>待确认候选：{diagnostics?.candidate_pending_total || 0}</p>
                <p>缺失文件：{diagnostics?.missing_file_count || 0}</p>
              </div>
            </details>
          )}
        </>
      )}
    </div>
    </ExperimentPageShell>
  )
}
