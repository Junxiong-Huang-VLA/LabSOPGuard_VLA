import { useEffect, useMemo, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import {
  ArrowLeft,
  BadgeCheck,
  Boxes,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock3,
  ExternalLink,
  FileText,
  Filter,
  Image,
  Maximize2,
  RefreshCw,
  Search,
  ShieldAlert,
  X,
} from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, primaryButtonClass, secondaryButtonClass, toneForStatus } from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import { experimentFileUrl } from '../mediaUrl'
import type { MaterialCandidateFile, MaterialCandidateGroup, MaterialDiagnosticsResponse, MaterialSearchItem } from '../types'

type MaterialActionGroup = {
  key: string
  label: string
  canonicalObject?: string
  sopPhase?: string
  items: MaterialSearchItem[]
  bestItems: MaterialSearchItem[]
  clips: number
  frames: number
  start: number
  end: number
  objects: string[]
}

const CANONICAL_ACTION_LABELS: Record<string, string> = {
  'hand-bottle': 'Hand-bottle',
  'hand-balance': 'Hand-balance',
  'hand-spatula': 'Hand-spatula',
  'hand-paper': 'Hand-paper',
  'hand-container': 'Hand-container',
}

const REJECTION_REASONS = [
  { code: 'wrong_object', label: '对象错' },
  { code: 'wrong_action', label: '动作错' },
  { code: 'wrong_time_window', label: '时间窗错' },
  { code: 'duplicate', label: '重复素材' },
  { code: 'bad_visibility', label: '画面不可用' },
  { code: 'not_experiment_action', label: '非实验动作' },
  { code: 'low_evidence', label: '证据不足' },
]

type MaterialLike = MaterialSearchItem | MaterialCandidateGroup | MaterialCandidateFile

function asArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map(item => String(item)).filter(Boolean) : []
}

function payloadOf(item: MaterialLike) {
  const payload = 'payload' in item ? item.payload : undefined
  return payload && typeof payload === 'object' ? payload as Record<string, unknown> : {}
}

function pathText(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value
    if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  }
  return undefined
}

function itemPreview(item: MaterialSearchItem, experimentId?: string) {
  const paths = item.published_paths || {}
  return experimentFileUrl(item.preview_url || item.frame_path || paths.preview || paths.keyframe || undefined, experimentId)
}

function itemClip(item: MaterialSearchItem, experimentId?: string) {
  const paths = item.published_paths || {}
  return experimentFileUrl(item.clip_url || item.clip_file_path || paths.clip || undefined, experimentId)
}

function itemReport(item: MaterialSearchItem, experimentId?: string) {
  const paths = item.published_paths || {}
  return experimentFileUrl(item.report_url || paths.report || undefined, experimentId)
}

function itemKindLabel(item: MaterialSearchItem, experimentId?: string) {
  if (itemReport(item, experimentId)) return 'report'
  if (itemClip(item, experimentId)) return 'clip'
  return 'frame'
}

function itemStart(item: MaterialSearchItem) {
  const value = Number(item.time_start ?? item.timestamp_sec ?? item.local_timestamp_sec ?? 0)
  return Number.isFinite(value) ? value : 0
}

function itemEnd(item: MaterialSearchItem) {
  const value = Number(item.time_end ?? itemStart(item))
  return Number.isFinite(value) ? value : itemStart(item)
}

function formatRange(item: MaterialSearchItem) {
  return `${itemStart(item).toFixed(2)}-${itemEnd(item).toFixed(2)}s`
}

function materialKey(item: MaterialSearchItem, index: number) {
  return String(item.item_id || item.event_id || item.material_id || item.display_name || `material-${index}`)
}

function candidateGroupKey(group: MaterialCandidateGroup, index: number) {
  return String(group.candidate_group_id || group.micro_segment_id || group.parent_segment_id || `candidate-group-${index}`)
}

function candidateFileId(file: MaterialCandidateFile, index: number) {
  return String(file.candidate_id || file.item_id || file.url || file.path || file.preview_url || file.clip_url || `candidate-${index}`)
}

function candidateFileUrl(item: MaterialCandidateFile, experimentId?: string) {
  return experimentFileUrl(pathText(item.url, item.preview_url, item.clip_url, item.frame_path, item.clip_file_path, item.stored_file, item.source_file), experimentId)
}

function candidateFileName(file: MaterialCandidateFile) {
  return pathText(file.stored_filename, file.file_name, file.display_name, file.source_file, file.stored_file)?.replace(/\\/g, '/').split('/').pop() || '专业报告'
}

function candidateFileIsReport(file: MaterialCandidateFile) {
  const typeText = `${file.asset_kind || ''} ${file.material_type || ''} ${file.role || ''} ${candidateFileName(file)}`.toLowerCase()
  return typeText.includes('report') || typeText.includes('报告') || typeText.endsWith('.pdf')
}

function formatBytes(value: unknown) {
  const size = Number(value)
  if (!Number.isFinite(size) || size <= 0) return ''
  if (size >= 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`
  if (size >= 1024) return `${Math.round(size / 1024)} KB`
  return `${Math.round(size)} B`
}

function normalizedCandidateStatus(value: unknown) {
  const status = String(value || '').trim().toLowerCase()
  if (!status || status === 'unknown' || status === 'none' || status === 'null') return 'pending'
  if (status === 'accepted') return 'approved'
  return status
}

function candidateStatusText(group: MaterialCandidateGroup) {
  return normalizedCandidateStatus(group.status || group.review_status)
}

function candidateGroupApproved(group: MaterialCandidateGroup) {
  const status = candidateStatusText(group)
  return status.includes('approved') || status.includes('accepted')
}

function candidateGroupBlocked(group: MaterialCandidateGroup) {
  const status = candidateStatusText(group)
  const yoloStatus = String(group.yolo_recheck?.status || group.pipeline_status || '').toLowerCase()
  return ['blocked', 'rejected', 'failed'].some(keyword => status.includes(keyword) || yoloStatus.includes(keyword))
}

function candidateCanApprove(group: MaterialCandidateGroup) {
  const status = candidateStatusText(group)
  if (candidateGroupApproved(group) || candidateGroupBlocked(group) || status.includes('not_selected')) return false
  return !status || ['pending', 'review', 'candidate', 'needs_review'].some(keyword => status.includes(keyword))
}

function candidateFilesForGroup(group: MaterialCandidateGroup) {
  const byId = new Map<string, MaterialCandidateFile>()
  for (const file of [...(group.files || []), ...(group.keyframes || []), ...(group.clips || [])]) {
    const id = candidateFileId(file, byId.size)
    if (!byId.has(id)) byId.set(id, file)
  }
  return Array.from(byId.values())
}

function canonicalActionType(item: MaterialLike) {
  const payload = payloadOf(item)
  return pathText(item.canonical_action_type, payload.canonical_action_type)
}

function canonicalObject(item: MaterialLike) {
  const payload = payloadOf(item)
  return pathText(item.canonical_object, item.primary_object, payload.canonical_object, payload.primary_object)
}

function sopPhase(item: MaterialLike) {
  const payload = payloadOf(item)
  return pathText(item.sop_phase, payload.sop_phase)
}

function materialActionLabel(item: MaterialSearchItem) {
  const canonical = canonicalActionType(item)
  if (canonical) return CANONICAL_ACTION_LABELS[canonical] || canonical
  const raw = pathText(item.event_type, item.display_name, asArray(item.actions)[0], item.payload?.action_name, item.payload?.event_type)
  return cleanDisplayText(raw, '未分类物理交互')
}

function materialActionKey(item: MaterialSearchItem) {
  return canonicalActionType(item) || materialActionLabel(item)
}

function materialAssetKind(item: MaterialSearchItem, experimentId?: string): 'clip' | 'frame' {
  return itemClip(item, experimentId) ? 'clip' : 'frame'
}

function itemQualityScore(item: MaterialLike) {
  const payload = payloadOf(item)
  const extra = payload.extra && typeof payload.extra === 'object' ? payload.extra as Record<string, unknown> : {}
  const value = Number(item.quality_score ?? payload.quality_score ?? extra.quality_score ?? 0)
  return Number.isFinite(value) ? value : 0
}

function itemYoloEvidenceCount(item: MaterialLike) {
  const payload = payloadOf(item)
  const record = item as Record<string, unknown>
  const ownYolo = 'yolo_recheck' in item && item.yolo_recheck && typeof item.yolo_recheck === 'object' ? item.yolo_recheck as Record<string, unknown> : undefined
  const payloadYolo = payload.yolo_recheck && typeof payload.yolo_recheck === 'object' ? payload.yolo_recheck as Record<string, unknown> : undefined
  const value = Number(record.yolo_evidence_count ?? ownYolo?.valid_evidence_count ?? payloadYolo?.valid_evidence_count ?? payload.yolo_evidence_count ?? 0)
  return Number.isFinite(value) ? value : 0
}

function itemView(item: MaterialSearchItem | MaterialCandidateFile) {
  const payload = payloadOf(item)
  return pathText(item.view, item.camera_id, item.stream_id, payload.view, payload.camera_id, payload.stream_id) || '-'
}

function materialRank(item: MaterialSearchItem, experimentId?: string) {
  const bestScore = Number(item.best_score ?? item.payload?.best_score)
  if (Number.isFinite(bestScore) && bestScore > 0) return bestScore * 1000 + (itemClip(item, experimentId) ? 0.5 : 0)
  return itemQualityScore(item) * 100 + itemYoloEvidenceCount(item) + (item.recommended ? 2 : 0) + (itemClip(item, experimentId) ? 0.5 : 0)
}

function bestItemsForGroup(items: MaterialSearchItem[], experimentId?: string) {
  if (!items.length) return []
  return [items.slice().sort((left, right) => materialRank(right, experimentId) - materialRank(left, experimentId) || itemStart(left) - itemStart(right))[0]]
}

function itemSearchText(item: MaterialSearchItem) {
  return [
    item.display_name,
    item.event_type,
    item.canonical_action_type,
    item.canonical_object,
    item.sop_phase,
    item.item_id,
    item.event_id,
    ...(item.object_labels || []),
    ...(item.actions || []),
    ...asArray(item.payload?.object_labels),
    ...asArray(item.payload?.actions),
  ].join(' ').toLowerCase()
}

function bestReasonText(item: MaterialSearchItem) {
  const reason = pathText(item.best_reason, item.payload?.best_reason)
  if (reason) return cleanDisplayText(reason)
  const score = Number(item.best_score ?? item.payload?.best_score)
  const scoreText = Number.isFinite(score) && score > 0 ? `best_score ${score.toFixed(3)}` : `quality ${itemQualityScore(item).toFixed(2)}`
  const view = cleanDisplayText(itemView(item))
  const object = cleanDisplayText(canonicalObject(item) || '-')
  const action = cleanDisplayText(canonicalActionType(item) || materialActionLabel(item))
  return `${action}: ${scoreText}; YOLO ${itemYoloEvidenceCount(item)}; view ${view}; ${formatRange(item)}; object ${object}`
}

export default function MaterialSearch() {
  const { id } = useParams<{ id: string }>()
  const location = useLocation()
  const reviewMode = location.pathname.endsWith('/review')

  const [items, setItems] = useState<MaterialSearchItem[]>([])
  const [candidateGroups, setCandidateGroups] = useState<MaterialCandidateGroup[]>([])
  const [diagnostics, setDiagnostics] = useState<MaterialDiagnosticsResponse | null>(null)
  const [query, setQuery] = useState('')
  const [objectFilter, setObjectFilter] = useState('')
  const [actionFilter, setActionFilter] = useState('')
  const [materialScope, setMaterialScope] = useState<'best' | 'all'>('best')
  const [assetMode, setAssetMode] = useState<'all' | 'frame' | 'clip'>('all')
  const [candidateFilter, setCandidateFilter] = useState<'pending' | 'approved' | 'all'>('pending')
  const [candidateActionFilter, setCandidateActionFilter] = useState<'all' | 'hand-balance' | 'hand-container' | 'missing_best'>('all')
  const [batchRejectionReason, setBatchRejectionReason] = useState('')
  const [batchNote, setBatchNote] = useState('')
  const [batchBusy, setBatchBusy] = useState(false)
  const [approvingGroup, setApprovingGroup] = useState<string | null>(null)
  const [decidingGroup, setDecidingGroup] = useState<string | null>(null)
  const [rejectionReasons, setRejectionReasons] = useState<Record<string, string>>({})
  const [candidateNotes, setCandidateNotes] = useState<Record<string, string>>({})
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<Record<string, string[]>>({})
  const [approvalNotice, setApprovalNotice] = useState<string | null>(null)
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({})
  const [zoomed, setZoomed] = useState<MaterialSearchItem | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function load() {
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const [published, candidates, materialDiagnostics] = await Promise.all([
        experimentApi.getPublishedMaterials(id, { limit: 500 }, { force: true }),
        experimentApi.getMaterialCandidates(id, { force: true }).catch(() => ({ items: [] })),
        experimentApi.getMaterialDiagnostics(id).catch(() => null),
      ])
      setItems((published.items || []) as MaterialSearchItem[])
      setCandidateGroups((candidates.items || []) as MaterialCandidateGroup[])
      setDiagnostics(materialDiagnostics as MaterialDiagnosticsResponse | null)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '关键素材加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [id, reviewMode])

  const formalObjects = useMemo(() => {
    return Array.from(new Set(items.flatMap(item => [canonicalObject(item), ...asArray(item.object_labels), ...asArray(item.payload?.object_labels)]).filter(Boolean))).slice(0, 80)
  }, [items])

  const formalActions = useMemo(() => {
    return Array.from(new Set(items.map(item => canonicalActionType(item) || materialActionLabel(item)).filter(Boolean))).slice(0, 80)
  }, [items])

  const filteredItems = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    return items.filter(item => {
      const matchesQuery = !keyword || itemSearchText(item).includes(keyword)
      const labels = [canonicalObject(item), ...asArray(item.object_labels), ...asArray(item.payload?.object_labels)].filter(Boolean)
      const matchesObject = !objectFilter || labels.includes(objectFilter)
      const matchesAction = !actionFilter || (canonicalActionType(item) || materialActionLabel(item)) === actionFilter
      const matchesKind = assetMode === 'all' || materialAssetKind(item, id) === assetMode
      return matchesQuery && matchesObject && matchesAction && matchesKind
    })
  }, [actionFilter, assetMode, id, items, objectFilter, query])

  const materialGroups = useMemo(() => {
    const grouped = new Map<string, MaterialActionGroup>()
    for (const item of filteredItems) {
      const label = materialActionLabel(item)
      const key = materialActionKey(item)
      const current = grouped.get(key) || {
        key,
        label,
        canonicalObject: canonicalObject(item),
        sopPhase: sopPhase(item),
        items: [],
        bestItems: [],
        clips: 0,
        frames: 0,
        start: Number.POSITIVE_INFINITY,
        end: 0,
        objects: [],
      }
      current.items.push(item)
      current.clips += itemClip(item, id) ? 1 : 0
      current.frames += itemPreview(item, id) ? 1 : 0
      current.start = Math.min(current.start, itemStart(item))
      current.end = Math.max(current.end, itemEnd(item))
      current.objects = Array.from(new Set([...current.objects, canonicalObject(item), ...asArray(item.object_labels), ...asArray(item.payload?.object_labels)].filter(Boolean) as string[]))
      grouped.set(key, current)
    }
    return Array.from(grouped.values())
      .map(group => ({ ...group, bestItems: bestItemsForGroup(group.items, id) }))
      .sort((left, right) => left.start - right.start || left.label.localeCompare(right.label))
  }, [filteredItems, id])

  const clips = items.filter(item => Boolean(itemClip(item, id))).length
  const frames = items.filter(item => Boolean(itemPreview(item, id))).length
  const formalActionSet = useMemo(() => new Set(items.map(item => canonicalActionType(item)).filter(Boolean) as string[]), [items])
  const pendingCandidates = candidateGroups.filter(candidateCanApprove).length
  const approvedCandidates = candidateGroups.filter(candidateGroupApproved).length
  const rejectedCandidates = candidateGroups.filter(group => candidateStatusText(group).includes('rejected')).length
  const deferredCandidates = candidateGroups.filter(group => candidateStatusText(group).includes('deferred')).length

  function selectedIdsForGroup(group: MaterialCandidateGroup) {
    const selected = selectedCandidateIds[group.candidate_group_id]
    if (selected?.length) return selected
    return candidateFilesForGroup(group)
      .filter(file => file.recommended || file.exists !== false)
      .map(candidateFileId)
  }

  function toggleCandidate(group: MaterialCandidateGroup, file: MaterialCandidateFile, fileIndex: number) {
    const candidateId = candidateFileId(file, fileIndex)
    setSelectedCandidateIds(previous => {
      const current = new Set(previous[group.candidate_group_id] || selectedIdsForGroup(group))
      if (current.has(candidateId)) current.delete(candidateId)
      else current.add(candidateId)
      return { ...previous, [group.candidate_group_id]: Array.from(current) }
    })
  }

  async function approveCandidate(group: MaterialCandidateGroup) {
    if (!id || !candidateCanApprove(group)) return
    const selectedIds = selectedIdsForGroup(group)
    setApprovingGroup(group.candidate_group_id)
    setError(null)
    setApprovalNotice(null)
    try {
      const response = await experimentApi.approveMaterialCandidate(id, group.candidate_group_id, {
        reviewer: 'frontend-review',
        notes: `Approved from material review queue (${selectedIds.length} selected assets)`,
        reason_code: 'representative_yolo_hand_object_evidence',
        reason: `Approved ${canonicalActionType(group) || group.action_name || 'candidate'} because selected files provide representative YOLO-backed hand-object evidence.`,
        candidate_ids: selectedIds,
        selected_keyframe_ids: selectedIds,
        selected_clip_ids: selectedIds,
      })
      const selectedFiles = candidateFilesForGroup(group).filter((file, fileIndex) => selectedIds.includes(candidateFileId(file, fileIndex)))
      const reportOnly = selectedFiles.length > 0 && selectedFiles.every(candidateFileIsReport)
      const published = response?.published_materials as { total?: number; items?: unknown[] } | undefined
      const publishedTotal = Number(published?.total ?? published?.items?.length ?? items.length)
      setSelectedCandidateIds(previous => {
        const next = { ...previous }
        delete next[group.candidate_group_id]
        return next
      })
      setApprovalNotice(reportOnly
        ? `已批准 ${selectedIds.length} 个专业 PDF；文件只进入“专业报告”文件夹，不进入正式关键素材库。`
        : `已批准 ${selectedIds.length} 个候选文件；正式关键素材库当前 ${publishedTotal} 个关键帧/关键片段。`)
      await load()
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '候选素材审批失败')
    } finally {
      setApprovingGroup(null)
    }
  }

  async function decideCandidate(group: MaterialCandidateGroup, decision: 'false_positive' | 'deferred') {
    if (!id) return
    const reasonCode = rejectionReasons[group.candidate_group_id] || ''
    const note = candidateNotes[group.candidate_group_id] || ''
    if (decision === 'false_positive' && !reasonCode) {
      setError('判定误筛前需要选择原因：对象错、动作错、时间窗错、重复素材或画面不可用等。')
      return
    }
    setDecidingGroup(group.candidate_group_id)
    setError(null)
    setApprovalNotice(null)
    try {
      await experimentApi.decideMaterialCandidate(id, group.candidate_group_id, {
        decision,
        reviewer: 'frontend-review',
        reason_code: decision === 'false_positive' ? reasonCode : undefined,
        notes: note,
      })
      setApprovalNotice(decision === 'false_positive' ? '已判定为误筛，默认审核队列会隐藏该候选，审计记录已保留。' : '已标记为暂不处理，候选会从默认待审队列移出并保留记录。')
      await load()
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '候选处置失败')
    } finally {
      setDecidingGroup(null)
    }
  }

  async function approveCandidateBatch(groups: MaterialCandidateGroup[]) {
    if (!id) return
    const targets = groups.filter(candidateCanApprove)
    if (!targets.length) return
    setBatchBusy(true)
    setError(null)
    setApprovalNotice(null)
    try {
      let approvedFiles = 0
      for (const group of targets) {
        const selectedIds = selectedIdsForGroup(group)
        if (!selectedIds.length) continue
        await experimentApi.approveMaterialCandidate(id, group.candidate_group_id, {
          reviewer: 'frontend-review',
          notes: batchNote || `Batch approved from canonical review view (${selectedIds.length} selected assets)`,
          reason_code: 'taxonomy_gap_representative',
          reason: `Batch approved ${canonicalActionType(group) || group.action_name || 'candidate'} to close the canonical material-library coverage gap.`,
          candidate_ids: selectedIds,
          selected_keyframe_ids: selectedIds,
          selected_clip_ids: selectedIds,
        })
        approvedFiles += selectedIds.length
      }
      setApprovalNotice(`Batch approved ${approvedFiles} selected candidate files; health panel refreshed.`)
      await load()
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Batch approval failed')
    } finally {
      setBatchBusy(false)
    }
  }

  async function rejectCandidateBatch(groups: MaterialCandidateGroup[]) {
    if (!id) return
    const targets = groups.filter(candidateCanApprove)
    if (!targets.length) return
    if (!batchRejectionReason) {
      setError('Batch false-positive review requires a reason.')
      return
    }
    setBatchBusy(true)
    setError(null)
    setApprovalNotice(null)
    try {
      for (const group of targets) {
        await experimentApi.decideMaterialCandidate(id, group.candidate_group_id, {
          decision: 'false_positive',
          reviewer: 'frontend-review',
          reason_code: batchRejectionReason,
          notes: batchNote || `Batch rejected from canonical review view: ${batchRejectionReason}`,
        })
      }
      setApprovalNotice(`Batch rejected ${targets.length} candidate groups; audit log retained and health panel refreshed.`)
      await load()
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Batch false-positive review failed')
    } finally {
      setBatchBusy(false)
    }
  }

  function toggleGroup(groupKey: string) {
    setCollapsedGroups(previous => ({ ...previous, [groupKey]: !previous[groupKey] }))
  }

  if (reviewMode) {
    const visibleCandidates = candidateGroups.filter(group => {
      if (candidateFilter === 'pending') return candidateCanApprove(group)
      if (candidateFilter === 'approved') return candidateGroupApproved(group)
      return true
    }).filter(group => {
      const canonical = canonicalActionType(group)
      if (candidateActionFilter === 'hand-balance') return canonical === 'hand-balance'
      if (candidateActionFilter === 'hand-container') return canonical === 'hand-container'
      if (candidateActionFilter === 'missing_best') return Boolean(canonical && !formalActionSet.has(canonical))
      return true
    })
    const batchTargets = visibleCandidates.filter(candidateCanApprove)
    return (
      <div className="space-y-5">
        <PageHero
          eyebrow={id ? <Link to={`/experiments/${id}/materials`} className="hover:text-slate-900">正式关键素材库</Link> : 'Materials'}
          title="候选素材审核"
          description="这里只处理候选关键帧、候选关键片段和专业 PDF 的入库审批；正式素材浏览已经拆到独立的关键素材库页面。"
          actions={id ? (
            <>
              <button type="button" onClick={() => void load()} className={secondaryButtonClass()}><RefreshCw className="h-4 w-4" />刷新</button>
              <Link to={`/experiments/${id}/materials`} className={primaryButtonClass('emerald')}><Boxes className="h-4 w-4" />正式素材库</Link>
            </>
          ) : null}
          tabs={id ? <Tabs id={id} active="review" /> : null}
        />

        <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4" data-smoke="material-review-metrics">
          <MetricTile label="候选组" value={candidateGroups.length} helper={`${visibleCandidates.length} visible`} tone="blue" Icon={Filter} />
          <MetricTile label="待审批" value={pendingCandidates} helper="can approve" tone="amber" Icon={Clock3} />
          <MetricTile label="已批准" value={approvedCandidates} helper={`${items.length} formal assets`} tone="emerald" Icon={CheckCircle2} />
          <MetricTile label="正式素材" value={items.length} helper={`${clips} clips / ${frames} frames`} tone="cyan" Icon={Image} />
        </section>

        {approvalNotice && (
          <EvidenceCard className="border-emerald-200 bg-emerald-50 p-4 text-sm font-bold text-emerald-800" data-smoke="material-approval-notice" aria-live="polite">
            <div className="flex items-center gap-2"><CheckCircle2 className="h-4 w-4" />{approvalNotice}</div>
          </EvidenceCard>
        )}
        {error && <EvidenceCard className="border-red-200 bg-red-50 p-4 text-red-700">{error}</EvidenceCard>}

        <EvidenceCard className="p-4">
          <div className="flex flex-wrap gap-2">
            {(['pending', 'approved', 'all'] as const).map(filter => (
              <button
                key={filter}
                type="button"
                onClick={() => setCandidateFilter(filter)}
                className={filter === candidateFilter ? primaryButtonClass(filter === 'pending' ? 'amber' : filter === 'approved' ? 'emerald' : 'slate') : secondaryButtonClass()}
              >
                {filter === 'pending' ? '只看待审批' : filter === 'approved' ? '只看已入库' : '全部候选'}
              </button>
            ))}
            <span className="inline-flex items-center rounded-md bg-slate-50 px-3 py-1.5 text-xs font-bold text-slate-500">
              rejected {rejectedCandidates} / deferred {deferredCandidates}
            </span>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {([
              ['all', '全部 canonical'],
              ['hand-balance', '只看 hand-balance'],
              ['hand-container', '只看 hand-container'],
              ['missing_best', '只看缺最佳素材'],
            ] as const).map(([filter, label]) => (
              <button
                key={filter}
                type="button"
                onClick={() => setCandidateActionFilter(filter)}
                className={filter === candidateActionFilter ? primaryButtonClass(filter === 'missing_best' ? 'amber' : 'blue') : secondaryButtonClass()}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto_auto]">
            <select value={batchRejectionReason} onChange={event => setBatchRejectionReason(event.target.value)} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-bold text-slate-700">
              <option value="">批量误筛原因</option>
              {REJECTION_REASONS.map(reason => <option key={reason.code} value={reason.code}>{reason.label}</option>)}
            </select>
            <input value={batchNote} onChange={event => setBatchNote(event.target.value)} placeholder="批量审核备注" className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold outline-none focus:border-blue-400" />
            <button type="button" onClick={() => void approveCandidateBatch(batchTargets)} disabled={!batchTargets.length || batchBusy} className={`${primaryButtonClass('emerald')} disabled:cursor-not-allowed disabled:opacity-50`}>
              <CheckCircle2 className="h-4 w-4" />批量批准 {batchTargets.length}
            </button>
            <button type="button" onClick={() => void rejectCandidateBatch(batchTargets)} disabled={!batchTargets.length || !batchRejectionReason || batchBusy} className={`${secondaryButtonClass('red')} disabled:cursor-not-allowed disabled:opacity-50`}>
              <ShieldAlert className="h-4 w-4" />批量误筛
            </button>
          </div>
        </EvidenceCard>

        {loading ? <EmptyEvidence title="正在加载候选素材..." /> : visibleCandidates.length === 0 ? <EmptyEvidence title="当前没有匹配的候选素材" /> : (
          <div className="grid gap-4 xl:grid-cols-2" data-smoke="material-review-grid" data-count={visibleCandidates.length}>
            {visibleCandidates.map((group, groupIndex) => (
              <CandidateGroupCard
                key={candidateGroupKey(group, groupIndex)}
                group={group}
                experimentId={id}
                selectedIds={selectedIdsForGroup(group)}
                approving={approvingGroup === group.candidate_group_id}
                deciding={decidingGroup === group.candidate_group_id}
                rejectionReason={rejectionReasons[group.candidate_group_id] || ''}
                note={candidateNotes[group.candidate_group_id] || ''}
                onToggle={toggleCandidate}
                onApprove={() => void approveCandidate(group)}
                onDecision={(decision) => void decideCandidate(group, decision)}
                onReasonChange={(reason) => setRejectionReasons(previous => ({ ...previous, [group.candidate_group_id]: reason }))}
                onNoteChange={(note) => setCandidateNotes(previous => ({ ...previous, [group.candidate_group_id]: note }))}
              />
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <PageHero
        eyebrow={id ? <Link to={`/experiments/${id}/workspace`} className="hover:text-slate-900">分析概览</Link> : 'Materials'}
        title="正式关键素材库"
        description="这里只展示已经人工批准入库的关键帧和关键片段，并按物理交互动作分类。候选审批和专业 PDF 归档不混在这个浏览界面里。"
        actions={id ? (
          <>
            <Link to={`/experiments/${id}/workspace`} className={secondaryButtonClass()}><ArrowLeft className="h-4 w-4" />工作台</Link>
            <button type="button" onClick={() => void load()} className={secondaryButtonClass()}><RefreshCw className="h-4 w-4" />刷新</button>
            <Link to={`/experiments/${id}/materials/review`} className={secondaryButtonClass('amber')}><Filter className="h-4 w-4" />候选审核</Link>
            <Link to={`/experiments/${id}/materials/timeline`} className={primaryButtonClass('blue')}><Clock3 className="h-4 w-4" />素材时间轴</Link>
          </>
        ) : null}
        tabs={id ? <Tabs id={id} active="materials" /> : null}
      />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4" data-smoke="experiment-material-metrics" data-total={items.length} data-clips={clips}>
        <MetricTile label="正式素材" value={items.length} helper={`${filteredItems.length} visible`} tone="blue" Icon={Boxes} />
        <MetricTile label="交互类别" value={materialGroups.length} helper="physical action groups" tone="violet" Icon={BadgeCheck} />
        <MetricTile label="关键帧" value={frames} helper="approved frames" tone="emerald" Icon={Image} />
        <MetricTile label="关键片段" value={clips} helper="approved clips" tone="cyan" Icon={Image} />
      </section>

      <MaterialHealthPanel diagnostics={diagnostics} candidateGroups={candidateGroups} />

      <EvidenceCard className="p-4">
        <div className="mb-3 flex flex-wrap gap-2">
          {(['best', 'all'] as const).map(scope => (
            <button key={scope} type="button" onClick={() => setMaterialScope(scope)} className={scope === materialScope ? primaryButtonClass(scope === 'best' ? 'emerald' : 'slate') : secondaryButtonClass()}>
              {scope === 'best' ? '最佳素材' : '全部入库素材'}
            </button>
          ))}
          {(['all', 'frame', 'clip'] as const).map(mode => (
            <button key={mode} type="button" onClick={() => setAssetMode(mode)} className={mode === assetMode ? primaryButtonClass('blue') : secondaryButtonClass()}>
              {mode === 'all' ? '全部类型' : mode === 'frame' ? '关键帧' : '关键片段'}
            </button>
          ))}
        </div>
        <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_14rem_14rem]">
          <label className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input value={query} onChange={event => setQuery(event.target.value)} placeholder="搜索对象、动作或素材编号" className="w-full rounded-lg border border-slate-200 py-2 pl-9 pr-3 text-sm font-semibold outline-none focus:border-blue-400" />
          </label>
          <select value={objectFilter} onChange={event => setObjectFilter(event.target.value)} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-bold text-slate-700">
            <option value="">全部对象</option>
            {formalObjects.map(item => <option key={item} value={item}>{cleanDisplayText(item)}</option>)}
          </select>
          <select value={actionFilter} onChange={event => setActionFilter(event.target.value)} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-bold text-slate-700">
            <option value="">全部动作类别</option>
            {formalActions.map(item => <option key={item} value={item}>{cleanDisplayText(item)}</option>)}
          </select>
        </div>
      </EvidenceCard>

      {error && <EvidenceCard className="border-red-200 bg-red-50 p-4 text-red-700">{error}</EvidenceCard>}
      {loading ? <EmptyEvidence title="正在加载正式关键素材..." /> : materialGroups.length === 0 ? <EmptyEvidence title="暂无匹配的正式关键素材" /> : (
        <div className="space-y-4" data-smoke="formal-material-library" data-count={filteredItems.length}>
          {materialGroups.map(group => {
            const collapsed = Boolean(collapsedGroups[group.key])
            const groupItems = materialScope === 'all' || collapsed ? group.items : group.bestItems
            return (
              <EvidenceCard key={group.key} className="overflow-hidden" data-smoke="material-action-group" data-count={group.items.length}>
                <button type="button" onClick={() => toggleGroup(group.key)} className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left transition hover:bg-slate-50">
                  <div className="flex min-w-0 items-center gap-3">
                    {collapsed ? <ChevronDown className="h-4 w-4 shrink-0 text-slate-500" /> : <ChevronRight className="h-4 w-4 shrink-0 text-slate-500" />}
                    <div className="min-w-0">
                      <h2 className="truncate text-lg font-black text-slate-950">{cleanDisplayText(group.label)}</h2>
                      <p className="mt-1 text-xs font-semibold text-slate-500">{group.start.toFixed(2)}-{group.end.toFixed(2)}s · {group.objects.slice(0, 6).map(item => cleanDisplayText(item)).join(' / ') || 'object evidence'}</p>
                    </div>
                  </div>
                  <div className="flex shrink-0 flex-wrap justify-end gap-2">
                    <EvidenceBadge tone="amber">{materialScope === 'all' || collapsed ? '全部' : '最佳'}</EvidenceBadge>
                    <EvidenceBadge tone="blue">{group.items.length} 个素材</EvidenceBadge>
                    <EvidenceBadge tone="emerald">{group.frames} 帧</EvidenceBadge>
                    <EvidenceBadge tone="cyan">{group.clips} 片段</EvidenceBadge>
                  </div>
                </button>
                {true && (
                  <div className="grid gap-3 border-t border-slate-100 p-4 sm:grid-cols-2 xl:grid-cols-4" data-smoke="experiment-material-grid" data-count={groupItems.length}>
                    {groupItems.map((item, itemIndex) => (
                      <MaterialCard key={materialKey(item, itemIndex)} item={item} experimentId={id} isBest={group.bestItems.includes(item)} onZoom={() => setZoomed(item)} />
                    ))}
                  </div>
                )}
              </EvidenceCard>
            )
          })}
        </div>
      )}

      {zoomed && <MaterialZoomModal item={zoomed} experimentId={id} onClose={() => setZoomed(null)} />}
    </div>
  )
}

function MaterialHealthPanel({ diagnostics, candidateGroups }: { diagnostics: MaterialDiagnosticsResponse | null; candidateGroups: MaterialCandidateGroup[] }) {
  const [open, setOpen] = useState(false)
  const pending = diagnostics?.candidate_pending_total ?? candidateGroups.filter(candidateCanApprove).length
  const approvedUnsynced = diagnostics?.approved_unsynced_candidates || []
  const formalTotal = diagnostics?.formal_material_total ?? diagnostics?.published_total ?? 0
  const bestTotal = diagnostics?.best_material_total ?? 0
  const accessible = diagnostics?.url_accessible_count ?? 0
  const missing = diagnostics?.missing_file_count ?? 0
  const lastSync = diagnostics?.last_formal_sync_at ? new Date(diagnostics.last_formal_sync_at).toLocaleString() : '-'
  return (
    <EvidenceCard className="p-4" data-smoke="material-library-health">
      <button type="button" onClick={() => setOpen(value => !value)} className="flex w-full items-center justify-between gap-3 text-left">
        <div>
          <h2 className="text-sm font-black text-slate-950">同步健康</h2>
          <p className="mt-1 text-xs font-semibold text-slate-500">正式 {formalTotal} · 最佳 {bestTotal} · 待审 {pending} · 最近入库 {lastSync}</p>
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          <EvidenceBadge tone={missing ? 'red' : 'emerald'}>{accessible} accessible / {missing} missing</EvidenceBadge>
          <EvidenceBadge tone={approvedUnsynced.length ? 'amber' : 'emerald'}>{approvedUnsynced.length} approved unsynced</EvidenceBadge>
          {open ? <ChevronDown className="h-4 w-4 text-slate-500" /> : <ChevronRight className="h-4 w-4 text-slate-500" />}
        </div>
      </button>
      {open && (
        <div className="mt-3 border-t border-slate-100 pt-3">
          {approvedUnsynced.length === 0 ? (
            <p className="text-sm font-semibold text-emerald-700">没有发现“已批准但未同步”的候选。</p>
          ) : (
            <div className="space-y-2">
              {approvedUnsynced.slice(0, 20).map((item, index) => (
                <div key={`${item.candidate_id || index}`} className="rounded-md bg-amber-50 px-3 py-2 text-xs font-semibold text-amber-800">
                  {item.candidate_id || 'candidate'} · {item.candidate_group_id || '-'} · {item.canonical_action_type || item.display_name || '-'}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </EvidenceCard>
  )
}

function MaterialCard({ item, experimentId, isBest, onZoom }: { item: MaterialSearchItem; experimentId?: string; isBest?: boolean; onZoom: () => void }) {
  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white transition hover:border-blue-200 hover:shadow-md">
      <div className="relative">
        <MaterialPreview item={item} experimentId={experimentId} />
        <button type="button" onClick={onZoom} className="absolute right-2 top-2 inline-flex h-8 w-8 items-center justify-center rounded-md bg-white/95 text-slate-700 shadow-sm ring-1 ring-slate-200 hover:text-blue-700" title="放大预览">
          <Maximize2 className="h-4 w-4" />
        </button>
      </div>
      <div className="p-3">
        <div className="mb-2 flex flex-wrap gap-1.5">
          <EvidenceBadge tone="blue">{formatRange(item)}</EvidenceBadge>
          <EvidenceBadge tone={itemClip(item, experimentId) ? 'emerald' : 'slate'}>{itemKindLabel(item, experimentId)}</EvidenceBadge>
          {isBest && <EvidenceBadge tone="amber">best</EvidenceBadge>}
        </div>
        <h3 className="line-clamp-2 min-h-10 text-sm font-black text-slate-950">{cleanDisplayText(item.display_name || item.event_type || item.item_id, '关键素材')}</h3>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {[canonicalActionType(item), canonicalObject(item), ...(item.object_labels || []), item.event_type].filter(Boolean).slice(0, 4).map((label, labelIndex) => (
            <span key={`${String(label)}-${labelIndex}`} className="rounded bg-slate-100 px-2 py-1 text-xs font-bold text-slate-600">{cleanDisplayText(label)}</span>
          ))}
        </div>
        <div className="mt-3 rounded-md bg-slate-50 p-2 text-xs font-semibold text-slate-600">
          <div className="font-black text-slate-700">{isBest ? '为什么它是最佳' : '入库依据'}</div>
          <p className="mt-1 leading-relaxed">{bestReasonText(item)}</p>
          <div className="mt-2 grid grid-cols-2 gap-1">
            <span>best {Number(item.best_score ?? item.payload?.best_score ?? 0).toFixed(3)}</span>
            <span>YOLO {itemYoloEvidenceCount(item)}</span>
            <span>view {cleanDisplayText(itemView(item))}</span>
            <span>object {cleanDisplayText(canonicalObject(item) || '-')}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function MaterialPreview({ item, experimentId }: { item: MaterialSearchItem; experimentId?: string }) {
  const preview = itemPreview(item, experimentId)
  const clip = itemClip(item, experimentId)
  if (preview) return <img src={preview} alt={cleanDisplayText(item.display_name || item.item_id, 'material')} className="aspect-video w-full bg-slate-100 object-cover" data-smoke="experiment-formal-image" />
  if (clip) return <video src={clip} className="aspect-video w-full bg-slate-950 object-contain" controls playsInline preload="metadata" data-smoke="experiment-formal-video" />
  return <div className="flex aspect-video items-center justify-center bg-slate-100 text-sm font-semibold text-slate-400">no preview</div>
}

function MaterialZoomModal({ item, experimentId, onClose }: { item: MaterialSearchItem; experimentId?: string; onClose: () => void }) {
  const clip = itemClip(item, experimentId)
  const preview = itemPreview(item, experimentId)
  const media = clip || preview
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4" role="dialog" aria-modal="true">
      <div className="max-h-full w-full max-w-6xl overflow-hidden rounded-lg bg-white shadow-2xl">
        <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
          <div className="min-w-0">
            <h3 className="truncate font-black text-slate-950">{cleanDisplayText(item.display_name || item.event_type || item.item_id, '关键素材')}</h3>
            <p className="text-xs font-semibold text-slate-500">{formatRange(item)} · {cleanDisplayText(item.event_type || materialActionLabel(item))}</p>
          </div>
          <div className="flex shrink-0 gap-2">
            {media && <a href={media} target="_blank" rel="noreferrer" className={secondaryButtonClass('blue')}><ExternalLink className="h-4 w-4" />打开文件</a>}
            <button type="button" onClick={onClose} className={secondaryButtonClass()}><X className="h-4 w-4" />关闭</button>
          </div>
        </div>
        <div className="bg-slate-950 p-3">
          {clip ? (
            <video src={clip} className="max-h-[75vh] w-full bg-slate-950 object-contain" controls autoPlay playsInline />
          ) : preview ? (
            <img src={preview} alt={cleanDisplayText(item.display_name || item.item_id, 'material')} className="max-h-[75vh] w-full object-contain" />
          ) : (
            <div className="flex h-96 items-center justify-center text-sm font-bold text-white">no preview</div>
          )}
        </div>
      </div>
    </div>
  )
}

function CandidateGroupCard({
  group,
  experimentId,
  selectedIds,
  approving,
  deciding,
  rejectionReason,
  note,
  onToggle,
  onApprove,
  onDecision,
  onReasonChange,
  onNoteChange,
}: {
  group: MaterialCandidateGroup
  experimentId?: string
  selectedIds: string[]
  approving: boolean
  deciding: boolean
  rejectionReason: string
  note: string
  onToggle: (group: MaterialCandidateGroup, file: MaterialCandidateFile, fileIndex: number) => void
  onApprove: () => void
  onDecision: (decision: 'false_positive' | 'deferred') => void
  onReasonChange: (reason: string) => void
  onNoteChange: (note: string) => void
}) {
  const files = candidateFilesForGroup(group)
  const canApprove = candidateCanApprove(group)
  const approved = candidateGroupApproved(group)
  const yoloStatus = String(group.yolo_recheck?.status || group.pipeline_status || 'unknown')
  const vlmStatus = String(group.vlm_semantics?.status || group.pipeline_stage || 'not_available')
  const decisionReason = pathText(group.approval_reason, group.rejection_reason, group.review_notes)
  return (
    <EvidenceCard className="p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-black text-slate-950">{cleanDisplayText(group.action_name || group.primary_object || group.candidate_group_id, '候选素材')}</h3>
          <p className="mt-1 text-xs font-semibold text-slate-500">{canonicalActionType(group) || '-'} · {canonicalObject(group) || group.primary_object || '-'} · {group.micro_segment_id || group.parent_segment_id || group.candidate_group_id}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <EvidenceBadge tone={toneForStatus(group.status || group.review_status)}>{group.status || group.review_status || 'pending'}</EvidenceBadge>
          <EvidenceBadge tone={toneForStatus(yoloStatus)}>{yoloStatus}</EvidenceBadge>
        </div>
      </div>
      <div className="mt-3 grid gap-2 text-xs font-bold text-slate-600 sm:grid-cols-4">
        <span>关键帧 {group.keyframes?.length || 0}</span>
        <span>关键片段 {group.clips?.length || 0}</span>
        <span>候选文件 {files.length}</span>
        <span>质量 {group.quality_score == null ? '-' : Number(group.quality_score).toFixed(2)}</span>
      </div>
      <div className="mt-3 grid gap-2 text-xs font-semibold text-slate-600 sm:grid-cols-3">
        <Gate label="YOLO" value={yoloStatus} />
        <Gate label="VLM" value={vlmStatus} />
        <Gate label="Pipeline" value={group.pipeline_status || group.pipeline_stage || group.status || 'pending'} />
      </div>
      {decisionReason && (
        <div className="mt-3 rounded-md bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-600">
          审核原因：{cleanDisplayText(decisionReason)}
        </div>
      )}
      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        {files.map((file, fileIndex) => (
          <CandidatePreview
            key={candidateFileId(file, fileIndex)}
            file={file}
            experimentId={experimentId}
            selected={selectedIds.includes(candidateFileId(file, fileIndex))}
            disabled={!canApprove}
            onToggle={() => onToggle(group, file, fileIndex)}
          />
        ))}
      </div>
      {canApprove && (
        <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-black text-slate-500">误筛原因</span>
            {REJECTION_REASONS.map(reason => (
              <button
                key={reason.code}
                type="button"
                onClick={() => onReasonChange(reason.code)}
                className={`rounded-md border px-2 py-1 text-xs font-bold transition ${rejectionReason === reason.code ? 'border-red-200 bg-red-50 text-red-700' : 'border-slate-200 bg-white text-slate-600 hover:border-red-200 hover:text-red-700'}`}
              >
                {reason.label}
              </button>
            ))}
          </div>
          <textarea
            value={note}
            onChange={event => onNoteChange(event.currentTarget.value)}
            placeholder="审核备注"
            className="mt-3 min-h-20 w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 outline-none transition placeholder:text-slate-400 focus:border-blue-300"
          />
        </div>
      )}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs font-semibold text-slate-500">已选 {selectedIds.length} · 前端审核通过后才会同步正式素材。</p>
        <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onApprove}
          disabled={!canApprove || selectedIds.length === 0 || approving || deciding}
          className={`${canApprove ? primaryButtonClass('emerald') : secondaryButtonClass()} disabled:cursor-not-allowed disabled:opacity-50`}
        >
          <CheckCircle2 className="h-4 w-4" />
          {approving ? '审批中' : approved ? '已入库' : '批准入库'}
        </button>
          {canApprove && (
            <>
              <button type="button" onClick={() => onDecision('false_positive')} disabled={deciding || !rejectionReason} className={`${secondaryButtonClass('red')} disabled:cursor-not-allowed disabled:opacity-50`}>
                <ShieldAlert className="h-4 w-4" />判定误筛
              </button>
              <button type="button" onClick={() => onDecision('deferred')} disabled={deciding} className={`${secondaryButtonClass()} disabled:cursor-not-allowed disabled:opacity-50`}>
                <Clock3 className="h-4 w-4" />暂不处理
              </button>
            </>
          )}
        </div>
      </div>
      <CandidateStateNotice group={group} canApprove={canApprove} />
    </EvidenceCard>
  )
}

function CandidatePreview({
  file,
  experimentId,
  selected,
  disabled,
  onToggle,
}: {
  file: MaterialCandidateFile
  experimentId?: string
  selected: boolean
  disabled: boolean
  onToggle: () => void
}) {
  const url = candidateFileUrl(file, experimentId)
  const typeText = `${file.asset_kind || ''} ${file.material_type || ''} ${file.role || ''}`.toLowerCase()
  const isClip = typeText.includes('clip') || typeText.includes('片段') || Boolean(file.clip_url)
  const isReport = candidateFileIsReport(file)
  if (!url) return <div className="flex aspect-video items-center justify-center rounded-lg bg-slate-100 text-xs font-bold text-slate-400">no file</div>
  const overlay = (
    <label className="absolute left-2 top-2 inline-flex items-center gap-1 rounded-md bg-white/95 px-2 py-1 text-xs font-black text-slate-700 shadow-sm">
      <input type="checkbox" checked={selected} disabled={disabled} onChange={onToggle} className="h-3.5 w-3.5 accent-emerald-600" />
      入库
    </label>
  )
  if (isReport) {
    const filename = candidateFileName(file)
    const ext = filename.includes('.') ? filename.split('.').pop()?.toUpperCase() : 'PDF'
    return (
      <div className={`relative flex aspect-video flex-col items-start justify-between rounded-lg border bg-slate-50 p-4 ${selected ? 'border-emerald-300' : 'border-slate-200'}`}>
        {overlay}
        <div className="mt-8 flex w-full min-w-0 items-start gap-3">
          <span className="rounded-lg bg-blue-50 p-2 text-blue-700"><FileText className="h-5 w-5" /></span>
          <div className="min-w-0">
            <div className="text-xs font-black uppercase text-blue-700">{ext || 'PDF'} 报告候选</div>
            <div className="mt-1 line-clamp-2 break-all text-sm font-black text-slate-950">{filename}</div>
            <div className="mt-1 text-xs font-semibold text-slate-500">{formatBytes(file.size_bytes) || '可打开预览'} · {cleanDisplayText(String(file.role || 'professional_report_pdf'))}</div>
          </div>
        </div>
        <a href={url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 text-sm font-black text-blue-700 hover:text-blue-900">打开报告</a>
      </div>
    )
  }
  if (isClip) {
    return (
      <div className="relative">
        {overlay}
        <video src={url} className={`aspect-video w-full rounded-lg border bg-slate-950 object-contain ${selected ? 'border-emerald-300' : 'border-slate-200'}`} controls playsInline preload="metadata" />
      </div>
    )
  }
  return (
    <div className="relative">
      {overlay}
      <img src={url} alt={cleanDisplayText(file.display_name || file.candidate_id, 'candidate')} className={`aspect-video w-full rounded-lg border bg-slate-100 object-cover ${selected ? 'border-emerald-300' : 'border-slate-200'}`} />
    </div>
  )
}

function CandidateStateNotice({ group, canApprove }: { group: MaterialCandidateGroup; canApprove: boolean }) {
  if (canApprove) return null
  const status = candidateStatusText(group)
  if (candidateGroupApproved(group)) {
    return (
      <div className="mt-3 flex items-center gap-2 rounded-lg bg-emerald-50 px-3 py-2 text-xs font-bold text-emerald-700">
        <CheckCircle2 className="h-4 w-4" />
        已批准入库。关键帧和关键片段显示在正式关键素材库；专业 PDF 只同步到“专业报告”文件夹。
      </div>
    )
  }
  if (status.includes('not_selected')) {
    return (
      <div className="mt-3 flex items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs font-bold text-slate-600">
        <Clock3 className="h-4 w-4" />
        同组审核已完成，本条不是本次入库的最佳素材。
      </div>
    )
  }
  if (status.includes('deferred')) {
    return (
      <div className="mt-3 flex items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs font-bold text-slate-600">
        <Clock3 className="h-4 w-4" />
        已暂不处理；记录保留在全部候选中。
      </div>
    )
  }
  if (status.includes('rejected')) {
    return (
      <div className="mt-3 flex items-center gap-2 rounded-lg bg-amber-50 px-3 py-2 text-xs font-bold text-amber-700">
        <ShieldAlert className="h-4 w-4" />
        已判定为误筛，默认审核队列隐藏。
      </div>
    )
  }
  if (candidateGroupBlocked(group)) {
    return (
      <div className="mt-3 flex items-center gap-2 rounded-lg bg-red-50 px-3 py-2 text-xs font-bold text-red-700">
        <ShieldAlert className="h-4 w-4" />
        复核未通过或证据链被阻断，不能进入正式素材库。
      </div>
    )
  }
  return (
    <div className="mt-3 flex items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs font-bold text-slate-600">
      <Clock3 className="h-4 w-4" />
      当前候选状态为 {status}，请刷新后继续审核。
    </div>
  )
}

function Gate({ label, value }: { label: string; value: unknown }) {
  return (
    <span className="rounded-lg bg-slate-50 px-2 py-1">
      <b className="mr-1 text-slate-400">{label}</b>
      {cleanDisplayText(String(value || 'unknown'))}
    </span>
  )
}

function Tabs({ id, active }: { id: string; active: 'materials' | 'review' }) {
  const cls = 'rounded-md px-3 py-1.5 text-sm font-bold text-slate-600 hover:bg-slate-100'
  const activeCls = 'rounded-md bg-slate-900 px-3 py-1.5 text-sm font-bold text-white'
  return (
    <nav className="flex flex-wrap gap-2">
      <Link to={`/experiments/${id}/workspace`} onMouseEnter={() => prefetchExperimentRoute(id, 'workspace')} className={cls}>分析概览</Link>
      <Link to={`/experiments/${id}/report`} className={cls}>分析报告</Link>
      <Link to={`/experiments/${id}/materials`} className={active === 'materials' ? activeCls : cls}>正式关键素材</Link>
      <Link to={`/experiments/${id}/materials/review`} className={active === 'review' ? activeCls : cls}>候选审核</Link>
      <Link to={`/experiments/${id}/materials/timeline`} className={cls}>素材时间轴</Link>
      <Link to={`/experiments/${id}/key-actions`} onMouseEnter={() => prefetchExperimentRoute(id, 'keyActions')} className={cls}>关键动作</Link>
    </nav>
  )
}
