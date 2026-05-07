import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, BadgeCheck, Boxes, CheckCircle2, Clock3, Filter, Image, Search, ShieldAlert, Video } from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, primaryButtonClass, secondaryButtonClass, toneForStatus } from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import { mediaUrl } from '../mediaUrl'
import type { MaterialCandidateFile, MaterialCandidateGroup, MaterialSearchItem } from '../types'

function asArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map(item => String(item)) : []
}

function itemPreview(item: MaterialSearchItem) {
  const paths = item.published_paths || {}
  return mediaUrl(item.preview_url || item.frame_path || paths.preview || paths.keyframe || undefined)
}

function itemClip(item: MaterialSearchItem) {
  const paths = item.published_paths || {}
  return mediaUrl(item.clip_url || item.clip_file_path || paths.clip || undefined)
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

function candidateFileUrl(item: MaterialCandidateFile) {
  return mediaUrl(item.url || item.preview_url || item.clip_url || item.frame_path || item.clip_file_path || undefined)
}

function candidateCanApprove(group: MaterialCandidateGroup) {
  const status = String(group.status || group.review_status || '').toLowerCase()
  const yoloStatus = String(group.yolo_recheck?.status || group.pipeline_status || '').toLowerCase()
  const blocked = ['blocked', 'rejected', 'failed'].some(keyword => status.includes(keyword) || yoloStatus.includes(keyword))
  return !blocked && ['pending', 'review', 'candidate', ''].some(keyword => status.includes(keyword))
}

export default function MaterialSearch() {
  const { id } = useParams<{ id: string }>()
  const [items, setItems] = useState<MaterialSearchItem[]>([])
  const [candidateGroups, setCandidateGroups] = useState<MaterialCandidateGroup[]>([])
  const [query, setQuery] = useState('')
  const [objectFilter, setObjectFilter] = useState('')
  const [actionFilter, setActionFilter] = useState('')
  const [reviewFilter, setReviewFilter] = useState('all')
  const [approvingGroup, setApprovingGroup] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function load() {
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const [published, candidates] = await Promise.all([
        experimentApi.getPublishedMaterials(id, undefined, { force: true }),
        experimentApi.getMaterialCandidates(id, { force: true }).catch(() => ({ items: [] })),
      ])
      setItems((published.items || []) as MaterialSearchItem[])
      setCandidateGroups(candidates.items || [])
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
  const clips = items.filter(item => Boolean(itemClip(item))).length
  const pendingCandidates = candidateGroups.filter(candidateCanApprove).length

  async function approveCandidate(group: MaterialCandidateGroup) {
    if (!id || !candidateCanApprove(group)) return
    setApprovingGroup(group.candidate_group_id)
    setError(null)
    try {
      await experimentApi.approveMaterialCandidate(id, group.candidate_group_id, {
        reviewer: 'frontend-review',
        notes: 'Approved from key evidence library',
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

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
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
                  <div className="mt-3 grid gap-2 text-xs font-bold text-slate-600 sm:grid-cols-3">
                    <span>关键帧 {group.keyframes.length}</span>
                    <span>关键片段 {group.clips.length}</span>
                    <span>VLM {vlmStatus}</span>
                  </div>
                  <div className="mt-3 grid gap-2 sm:grid-cols-3">
                    {[...group.keyframes.slice(0, 2), ...group.clips.slice(0, 1)].map((file, fileIndex) => (
                      <CandidatePreview key={candidateFileKey(file, fileIndex)} file={file} />
                    ))}
                  </div>
                  <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                    <p className="text-xs font-semibold text-slate-500">{cleanDisplayText(group.review_gate_policy || '前端审核通过后同步到 material_references')}</p>
                    <button
                      type="button"
                      onClick={() => void approveCandidate(group)}
                      disabled={!canApprove || approvingGroup === group.candidate_group_id}
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
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map((item, itemIndex) => (
            <EvidenceCard key={materialKey(item, itemIndex)} className="overflow-hidden">
              {itemPreview(item) ? <img src={itemPreview(item)} alt={cleanDisplayText(item.display_name || item.item_id, 'material')} className="aspect-video w-full bg-slate-100 object-cover" /> : <div className="flex aspect-video items-center justify-center bg-slate-100 text-sm font-semibold text-slate-400">no preview</div>}
              <div className="p-4">
                <div className="mb-2 flex flex-wrap gap-2">
                  <EvidenceBadge tone="blue">{formatRange(item)}</EvidenceBadge>
                  <EvidenceBadge tone={itemClip(item) ? 'emerald' : 'slate'}>{itemClip(item) ? 'clip' : 'frame'}</EvidenceBadge>
                  {item.review_status && <EvidenceBadge tone="amber">{item.review_status}</EvidenceBadge>}
                </div>
                <h3 className="line-clamp-2 font-black text-slate-950">{cleanDisplayText(item.display_name || item.event_type || item.item_id, '关键素材')}</h3>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {[...(item.object_labels || []), ...(item.actions || []), item.event_type].filter(Boolean).slice(0, 8).map((label, labelIndex) => <span key={keyed(label, labelIndex, 'material-label')} className="rounded bg-slate-100 px-2 py-1 text-xs font-bold text-slate-600">{cleanDisplayText(label)}</span>)}
                </div>
                {itemClip(item) && <a href={itemClip(item)} target="_blank" rel="noreferrer" className="mt-3 inline-flex text-sm font-bold text-blue-700 hover:text-blue-900">打开片段</a>}
              </div>
            </EvidenceCard>
          ))}
        </div>
      )}
    </div>
  )
}

function CandidatePreview({ file }: { file: MaterialCandidateFile }) {
  const url = candidateFileUrl(file)
  const isClip = file.asset_kind === '关键片段' || Boolean(file.clip_url)
  if (!url) {
    return <div className="flex aspect-video items-center justify-center rounded-lg bg-slate-100 text-xs font-bold text-slate-400">no file</div>
  }
  if (isClip) {
    return (
      <a href={url} target="_blank" rel="noreferrer" className="group block overflow-hidden rounded-lg border border-slate-200 bg-slate-100">
        <div className="flex aspect-video items-center justify-center">
          <Video className="h-5 w-5 text-slate-500 group-hover:text-blue-600" />
        </div>
      </a>
    )
  }
  return <img src={url} alt={cleanDisplayText(file.display_name || file.candidate_id, 'candidate')} className="aspect-video rounded-lg border border-slate-200 bg-slate-100 object-cover" />
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
