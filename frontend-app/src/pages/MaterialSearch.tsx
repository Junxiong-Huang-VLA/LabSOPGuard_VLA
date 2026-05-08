import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, BadgeCheck, Boxes, CheckCircle2, Clock3, Filter, Image, Search, ShieldAlert } from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, primaryButtonClass, secondaryButtonClass, toneForStatus } from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import { experimentFileUrl, mediaUrl } from '../mediaUrl'
import type { MaterialCandidateFile, MaterialCandidateGroup, MaterialDiagnosticsEvidenceItem, MaterialDiagnosticsResponse, MaterialSearchItem } from '../types'

function asArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map(item => String(item)) : []
}

function itemPreview(item: MaterialSearchItem, experimentId?: string) {
  const paths = item.published_paths || {}
  return experimentFileUrl(item.preview_url || item.frame_path || paths.preview || paths.keyframe || undefined, experimentId)
}

function itemClip(item: MaterialSearchItem, experimentId?: string) {
  const paths = item.published_paths || {}
  return experimentFileUrl(item.clip_url || item.clip_file_path || paths.clip || undefined, experimentId)
}

function formatRange(item: MaterialSearchItem) {
  const start = Number(item.time_start ?? item.timestamp_sec ?? item.local_timestamp_sec ?? 0)
  const end = Number(item.time_end ?? start)
  return `${Number.isFinite(start) ? start.toFixed(2) : '-'}-${Number.isFinite(end) ? end.toFixed(2) : '-'}s`
}

function keyed(value: unknown, index: number, prefix: string) {
  const raw = String(value ?? '').trim()
  return `${prefix}-${raw || 'item'}-${index}`
}

function materialKey(item: MaterialSearchItem, index: number) {
  return keyed(item.event_id || item.item_id || item.display_name || item.event_type, index, 'material')
}

function candidateGroupKey(group: MaterialCandidateGroup, index: number) {
  return keyed(group.candidate_group_id || group.micro_segment_id || group.parent_segment_id, index, 'candidate-group')
}

function candidateFileKey(file: MaterialCandidateFile, index: number) {
  return keyed(file.candidate_id || file.url || file.path || file.preview_url || file.clip_url, index, 'candidate-file')
}

function candidateFileId(file: MaterialCandidateFile, index: number) {
  return String(file.candidate_id || file.item_id || file.url || file.path || file.preview_url || file.clip_url || `candidate-${index}`)
}

function candidateFileUrl(item: MaterialCandidateFile) {
  return mediaUrl(item.url || item.preview_url || item.clip_url || item.frame_path || item.clip_file_path || undefined)
}

function candidateCanApprove(group: MaterialCandidateGroup) {
  const status = String(group.status || group.review_status || '').toLowerCase()
  const yoloStatus = String(group.yolo_recheck?.status || group.pipeline_status || '').toLowerCase()
  const blocked = ['blocked', 'rejected', 'failed'].some(keyword => status.includes(keyword) || yoloStatus.includes(keyword))
  if (blocked || ['approved', 'accepted'].some(keyword => status.includes(keyword))) return false
  return !status || ['pending', 'review', 'candidate', 'needs_review'].some(keyword => status.includes(keyword))
}

export default function MaterialSearch() {
  const { id } = useParams<{ id: string }>()
  const [items, setItems] = useState<MaterialSearchItem[]>([])
  const [candidateGroups, setCandidateGroups] = useState<MaterialCandidateGroup[]>([])
  const [diagnostics, setDiagnostics] = useState<MaterialDiagnosticsResponse | null>(null)
  const [query, setQuery] = useState('')
  const [objectFilter, setObjectFilter] = useState('')
  const [actionFilter, setActionFilter] = useState('')
  const [reviewFilter, setReviewFilter] = useState('all')
  const [approvingGroup, setApprovingGroup] = useState<string | null>(null)
  const [selectedCandidateIds, setSelectedCandidateIds] = useState<Record<string, string[]>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function load() {
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const [published, candidates, nextDiagnostics] = await Promise.all([
        experimentApi.getPublishedMaterials(id, undefined, { force: true }),
        experimentApi.getMaterialCandidates(id, { force: true }).catch(() => ({ items: [] })),
        experimentApi.getMaterialDiagnostics(id).catch(() => null),
      ])
      setItems((published.items || []) as MaterialSearchItem[])
      setCandidateGroups(candidates.items || [])
      setDiagnostics(nextDiagnostics)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '关键素材加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [id])

  const objects = useMemo(() => Array.from(new Set(items.flatMap(item => asArray(item.object_labels).concat(asArray(item.payload?.object_labels))))).filter(Boolean).slice(0, 40), [items])
  const actions = useMemo(() => Array.from(new Set(items.flatMap(item => asArray(item.actions).concat(asArray(item.event_types), String(item.event_type || ''))))).filter(Boolean).slice(0, 40), [items])
  const filtered = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    return items.filter(item => {
      const text = [item.display_name, item.event_type, item.item_id, item.event_id, ...(item.object_labels || []), ...(item.actions || [])].join(' ').toLowerCase()
      const matchesQuery = !keyword || text.includes(keyword)
      const matchesObject = !objectFilter || asArray(item.object_labels).includes(objectFilter)
      const matchesAction = !actionFilter || asArray(item.actions).includes(actionFilter) || asArray(item.event_types).includes(actionFilter) || item.event_type === actionFilter
      const matchesReview = reviewFilter === 'all' || String(item.review_status || item.payload?.review_status || '').toLowerCase() === reviewFilter
      return matchesQuery && matchesObject && matchesAction && matchesReview
    })
  }, [actionFilter, items, objectFilter, query, reviewFilter])

  const strong = items.filter(item => String(item.evidence_level || item.payload?.evidence_level || '').toLowerCase().includes('strong')).length
  const pending = items.filter(item => String(item.review_status || item.payload?.review_status || '').toLowerCase().includes('pending')).length
  const clips = items.filter(item => Boolean(itemClip(item, id))).length
  const pendingCandidates = candidateGroups.filter(candidateCanApprove).length

  function selectedIdsForGroup(group: MaterialCandidateGroup) {
    const selected = selectedCandidateIds[group.candidate_group_id]
    if (selected?.length) return selected
    return [...group.keyframes, ...group.clips]
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
    try {
      await experimentApi.approveMaterialCandidate(id, group.candidate_group_id, {
        reviewer: 'frontend-review',
        notes: `Approved from key evidence library (${selectedIds.length} selected assets)`,
        candidate_ids: selectedIds,
        selected_keyframe_ids: selectedIds,
        selected_clip_ids: selectedIds,
      })
      await load()
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '候选素材审批失败')
    } finally {
      setApprovingGroup(null)
    }
  }

  return (
    <div className="space-y-5">
      <PageHero
        eyebrow={id ? <Link to={`/experiments/${id}/workspace`} className="hover:text-slate-900">分析概览</Link> : 'Materials'}
        title="关键证据库"
        description="集中审阅由片段、关键帧、物体标签、动作标签和审核状态组成的实验素材证据。"
        actions={id ? (
          <>
            <Link to={`/experiments/${id}/workspace`} className={secondaryButtonClass()}><ArrowLeft className="h-4 w-4" />工作台</Link>
            <Link to={`/experiments/${id}/materials/timeline`} className={primaryButtonClass('blue')}><Clock3 className="h-4 w-4" />时间轴</Link>
          </>
        ) : null}
        tabs={id ? <Tabs id={id} /> : null}
      />

      <section
        className="grid gap-4 md:grid-cols-2 xl:grid-cols-4"
        data-smoke="experiment-material-metrics"
        data-total={items.length}
        data-pending={pending + pendingCandidates}
        data-clips={clips}
      >
        <MetricTile label="素材总数" value={items.length} helper={`${filtered.length} visible`} tone="blue" Icon={Boxes} />
        <MetricTile label="待审核" value={pending + pendingCandidates} helper={`${pendingCandidates} candidate groups`} tone="amber" Icon={Filter} />
        <MetricTile label="强证据" value={strong} helper="strong evidence" tone="emerald" Icon={BadgeCheck} />
        <MetricTile label="视频片段" value={clips} helper="clip assets" tone="cyan" Icon={Image} />
      </section>

      {candidateGroups.length > 0 && (
        <EvidenceCard className="p-5">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-black text-slate-950">待审核素材候选</h2>
              <p className="mt-1 text-sm font-semibold text-slate-500">候选只在 YOLO 物理证据通过后允许进入正式关键素材库。</p>
            </div>
            <EvidenceBadge tone={pendingCandidates ? 'amber' : 'slate'}>{pendingCandidates} 组可审批</EvidenceBadge>
          </div>
          <div className="grid gap-4 xl:grid-cols-2">
            {candidateGroups.map((group, groupIndex) => {
              const canApprove = candidateCanApprove(group)
              const yoloStatus = String(group.yolo_recheck?.status || group.pipeline_status || 'unknown')
              const vlmStatus = String(group.vlm_semantics?.status || group.pipeline_stage || 'not_available')
              const selectedIds = selectedIdsForGroup(group)
              const candidateFiles = [...group.keyframes.slice(0, 3), ...group.clips.slice(0, 2)]
              return (
                <div key={candidateGroupKey(group, groupIndex)} className="rounded-lg border border-slate-200 bg-white p-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h3 className="font-black text-slate-950">{cleanDisplayText(group.action_name || group.primary_object || group.candidate_group_id, '候选素材')}</h3>
                      <p className="mt-1 text-xs font-semibold text-slate-500">{group.micro_segment_id || group.parent_segment_id || group.candidate_group_id}</p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <EvidenceBadge tone={toneForStatus(group.status || group.review_status)}>{group.status || group.review_status || 'pending'}</EvidenceBadge>
                      <EvidenceBadge tone={toneForStatus(yoloStatus)}>{yoloStatus}</EvidenceBadge>
                    </div>
                  </div>
                  <div className="mt-3 grid gap-2 text-xs font-bold text-slate-600 sm:grid-cols-4">
                    <span>关键帧 {group.keyframes.length}</span>
                    <span>关键片段 {group.clips.length}</span>
                    <span>推荐 {group.recommended_count ?? selectedIds.length}</span>
                    <span>质量 {group.quality_score == null ? '-' : Number(group.quality_score).toFixed(2)}</span>
                  </div>
                  <div className="mt-3 grid gap-2 text-xs font-semibold text-slate-600 sm:grid-cols-3">
                    <Gate label="YOLO" value={yoloStatus} />
                    <Gate label="VLM" value={vlmStatus} />
                    <Gate label="Pipeline" value={group.pipeline_status || group.pipeline_stage || group.status || 'pending'} />
                  </div>
                  <div className="mt-3 grid gap-2 sm:grid-cols-3">
                    {candidateFiles.map((file, fileIndex) => (
                      <CandidatePreview
                        key={candidateFileKey(file, fileIndex)}
                        file={file}
                        selected={selectedIds.includes(candidateFileId(file, fileIndex))}
                        disabled={!canApprove}
                        onToggle={() => toggleCandidate(group, file, fileIndex)}
                      />
                    ))}
                  </div>
                  <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                    <p className="text-xs font-semibold text-slate-500">{cleanDisplayText(group.review_gate_policy || '前端审核通过后同步到 material_references')} · 已选 {selectedIds.length}</p>
                    <button
                      type="button"
                      onClick={() => void approveCandidate(group)}
                      disabled={!canApprove || selectedIds.length === 0 || approvingGroup === group.candidate_group_id}
                      className={`${canApprove ? primaryButtonClass('emerald') : secondaryButtonClass()} disabled:cursor-not-allowed disabled:opacity-50`}
                    >
                      <CheckCircle2 className="h-4 w-4" />
                      {approvingGroup === group.candidate_group_id ? '审批中' : '批准入库'}
                    </button>
                  </div>
                  {!canApprove && (
                    <div className="mt-3 flex items-center gap-2 rounded-lg bg-red-50 px-3 py-2 text-xs font-bold text-red-700">
                      <ShieldAlert className="h-4 w-4" />
                      YOLO 复核未通过或候选已处理，不能直接入正式素材库。
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </EvidenceCard>
      )}

      {diagnostics?.evidence_items?.length ? <MaterialDiagnosticsPanel diagnostics={diagnostics} /> : null}

      <EvidenceCard className="p-4">
        <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_12rem_12rem_12rem]">
          <label className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input value={query} onChange={event => setQuery(event.target.value)} placeholder="搜索对象、动作或事件" className="w-full rounded-lg border border-slate-200 py-2 pl-9 pr-3 text-sm font-semibold outline-none focus:border-blue-400" />
          </label>
          <select value={objectFilter} onChange={event => setObjectFilter(event.target.value)} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-bold text-slate-700"><option value="">全部对象</option>{objects.map(item => <option key={item} value={item}>{item}</option>)}</select>
          <select value={actionFilter} onChange={event => setActionFilter(event.target.value)} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-bold text-slate-700"><option value="">全部动作</option>{actions.map(item => <option key={item} value={item}>{item}</option>)}</select>
          <select value={reviewFilter} onChange={event => setReviewFilter(event.target.value)} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-bold text-slate-700"><option value="all">全部审核</option><option value="pending">待审核</option><option value="approved">已通过</option><option value="rejected">已驳回</option></select>
        </div>
      </EvidenceCard>

      {error && <EvidenceCard className="border-red-200 bg-red-50 p-4 text-red-700">{error}</EvidenceCard>}
      {loading ? <EmptyEvidence title="正在加载关键素材..." /> : filtered.length === 0 ? <EmptyEvidence title="暂无匹配素材" /> : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3" data-smoke="experiment-material-grid" data-count={filtered.length}>
          {filtered.map((item, itemIndex) => (
            <EvidenceCard key={materialKey(item, itemIndex)} className="overflow-hidden" data-smoke="experiment-material-card">
              <MaterialPreview item={item} experimentId={id} />
              <div className="p-4">
                <div className="mb-2 flex flex-wrap gap-2">
                  <EvidenceBadge tone="blue">{formatRange(item)}</EvidenceBadge>
                  <EvidenceBadge tone={itemClip(item, id) ? 'emerald' : 'slate'}>{itemClip(item, id) ? 'clip' : 'frame'}</EvidenceBadge>
                  {item.review_status && <EvidenceBadge tone="amber">{item.review_status}</EvidenceBadge>}
                </div>
                <h3 className="line-clamp-2 font-black text-slate-950">{cleanDisplayText(item.display_name || item.event_type || item.item_id, '关键素材')}</h3>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {[...(item.object_labels || []), ...(item.actions || []), item.event_type].filter(Boolean).slice(0, 8).map((label, labelIndex) => <span key={keyed(label, labelIndex, 'material-label')} className="rounded bg-slate-100 px-2 py-1 text-xs font-bold text-slate-600">{cleanDisplayText(label)}</span>)}
                </div>
                {itemClip(item, id) && <a href={itemClip(item, id)} target="_blank" rel="noreferrer" className="mt-3 inline-flex text-sm font-bold text-blue-700 hover:text-blue-900">打开片段</a>}
              </div>
            </EvidenceCard>
          ))}
        </div>
      )}
    </div>
  )
}

function shortPath(value: unknown) {
  const text = String(value || '')
  if (!text) return '-'
  const normalized = text.replace(/\\/g, '/')
  const parts = normalized.split('/')
  return parts.slice(-3).join('/')
}

function MaterialDiagnosticsPanel({ diagnostics }: { diagnostics: MaterialDiagnosticsResponse }) {
  const rows = diagnostics.evidence_items || []
  const accessible = rows.filter(item => item.url_accessible).length
  return (
    <EvidenceCard className="p-5" data-smoke="material-diagnostics-panel" data-count={rows.length} data-url-accessible={accessible}>
      <details open>
        <summary className="flex cursor-pointer list-none flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-black text-slate-950">证据链诊断</h2>
            <p className="mt-1 text-sm font-semibold text-slate-500">确认正式素材来自已审批候选，并且文件与 URL 都能交付。</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <EvidenceBadge tone="blue">正式 {diagnostics.formal_material_reference_count ?? rows.length}</EvidenceBadge>
            <EvidenceBadge tone={accessible === rows.length ? 'emerald' : 'amber'}>URL 200 {accessible}/{rows.length}</EvidenceBadge>
          </div>
        </summary>
        <div className="mt-4 grid gap-3">
          {rows.map((item, index) => (
            <DiagnosticsRow key={`${item.candidate_id || item.material_url || 'diagnostic'}-${index}`} item={item} />
          ))}
        </div>
      </details>
    </EvidenceCard>
  )
}

function DiagnosticsRow({ item }: { item: MaterialDiagnosticsEvidenceItem }) {
  return (
    <div
      className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs font-semibold text-slate-600 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1fr)]"
      data-smoke="material-diagnostics-row"
      data-url-accessible={item.url_accessible ? 'true' : 'false'}
      data-file-exists={item.material_exists ? 'true' : 'false'}
    >
      <div className="min-w-0">
        <div className="mb-2 flex flex-wrap gap-2">
          <EvidenceBadge tone="slate">{item.asset_kind || 'material'}</EvidenceBadge>
          <EvidenceBadge tone={item.material_exists ? 'emerald' : 'red'}>file {item.material_exists ? 'ok' : 'missing'}</EvidenceBadge>
          <EvidenceBadge tone={item.url_accessible ? 'emerald' : 'red'}>url {item.url_accessible ? '200' : 'fail'}</EvidenceBadge>
        </div>
        <div className="break-all font-black text-slate-900">{item.candidate_id || '-'}</div>
        <div className="mt-1 break-all text-slate-500">{item.candidate_group_id || '-'}</div>
      </div>
      <div className="min-w-0 space-y-1">
        <div><span className="text-slate-400">审批人</span> {item.approved_by || '-'}</div>
        <div><span className="text-slate-400">YOLO</span> {item.yolo_recheck_status || '-'} · {item.yolo_valid_evidence_count ?? '-'}</div>
        <div><span className="text-slate-400">VLM</span> {item.vlm_model || item.vlm_status || '-'}</div>
      </div>
      <div className="min-w-0 space-y-1">
        <div><span className="text-slate-400">源文件</span> {shortPath(item.source_candidate_file || item.source_file)}</div>
        <div><span className="text-slate-400">正式文件</span> {shortPath(item.stored_file)}</div>
        {item.material_url && <a href={item.material_url} target="_blank" rel="noreferrer" className="inline-flex max-w-full break-all font-black text-blue-700 hover:text-blue-900">打开正式 URL</a>}
      </div>
    </div>
  )
}

function MaterialPreview({ item, experimentId }: { item: MaterialSearchItem; experimentId?: string }) {
  const preview = itemPreview(item, experimentId)
  const clip = itemClip(item, experimentId)
  if (preview) {
    return <img src={preview} alt={cleanDisplayText(item.display_name || item.item_id, 'material')} className="aspect-video w-full bg-slate-100 object-cover" data-smoke="experiment-formal-image" />
  }
  if (clip) {
    return <video src={clip} className="aspect-video w-full bg-slate-950 object-contain" controls preload="metadata" data-smoke="experiment-formal-video" />
  }
  return <div className="flex aspect-video items-center justify-center bg-slate-100 text-sm font-semibold text-slate-400">no preview</div>
}

function Gate({ label, value }: { label: string; value: unknown }) {
  return (
    <span className="rounded-lg bg-slate-50 px-2 py-1">
      <b className="mr-1 text-slate-400">{label}</b>
      {cleanDisplayText(String(value || 'unknown'))}
    </span>
  )
}

function CandidatePreview({ file, selected, disabled, onToggle }: { file: MaterialCandidateFile; selected: boolean; disabled: boolean; onToggle: () => void }) {
  const url = candidateFileUrl(file)
  const isClip = file.asset_kind === '关键片段' || Boolean(file.clip_url) || String(file.material_type || '').includes('clip')
  if (!url) {
    return <div className="flex aspect-video items-center justify-center rounded-lg bg-slate-100 text-xs font-bold text-slate-400">no file</div>
  }
  const overlay = (
    <label className="absolute left-2 top-2 inline-flex items-center gap-1 rounded-md bg-white/95 px-2 py-1 text-xs font-black text-slate-700 shadow-sm">
      <input type="checkbox" checked={selected} disabled={disabled} onChange={onToggle} className="h-3.5 w-3.5 accent-emerald-600" />
      入库
    </label>
  )
  if (isClip) {
    return (
      <div className="relative">
        {overlay}
        <video src={url} className={`aspect-video w-full rounded-lg border bg-slate-950 object-contain ${selected ? 'border-emerald-300' : 'border-slate-200'}`} controls preload="metadata" />
      </div>
    )
  }
  return (
    <div className="relative">
      {overlay}
      <img src={url} alt={cleanDisplayText(file.display_name || file.candidate_id, 'candidate')} className={`aspect-video rounded-lg border bg-slate-100 object-cover ${selected ? 'border-emerald-300' : 'border-slate-200'}`} />
    </div>
  )
}

function Tabs({ id }: { id: string }) {
  const cls = 'rounded-md px-3 py-1.5 text-sm font-bold text-slate-600 hover:bg-slate-100'
  return (
    <nav className="flex flex-wrap gap-2">
      <Link to={`/experiments/${id}/workspace`} onMouseEnter={() => prefetchExperimentRoute(id, 'workspace')} className={cls}>分析概览</Link>
      <Link to={`/experiments/${id}/report`} className={cls}>分析报告</Link>
      <Link to={`/experiments/${id}/materials`} className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-bold text-white">关键素材</Link>
      <Link to={`/experiments/${id}/materials/timeline`} className={cls}>素材时间轴</Link>
      <Link to={`/experiments/${id}/key-actions`} className={cls}>关键动作</Link>
    </nav>
  )
}
