import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  AlertTriangle,
  BookOpenCheck,
  Boxes,
  BrainCircuit,
  CalendarDays,
  DatabaseZap,
  MessageSquareText,
  RefreshCw,
  Send,
} from 'lucide-react'
import { experimentApi } from '../api'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, primaryButtonClass, secondaryButtonClass, toneForStatus } from '../components/EvidenceUI'
import ExperimentPageShell from '../components/ExperimentSideNav'
import { cleanDisplayText } from '../displayText'
import { mediaUrl } from '../mediaUrl'
import type { VideoMemoryEvidenceItem, VideoMemoryExperimentSummary, VideoMemoryOverview, VideoMemoryQueryAnswer } from '../types'

const DAY_MS = 24 * 60 * 60 * 1000

const ACTION_LABELS: Record<string, string> = {
  hand_object_interaction: '手物交互',
  object_move: '物体移动',
  object_movement_detected: '物体移动',
  object_movement_candidate: '物体移动候选',
  liquid_transfer: '液体转移',
  liquid_transfer_candidate: '液体转移候选',
  container_state_change: '容器状态变化',
  panel_operation: '设备面板操作',
  equipment_panel_operation_candidate: '设备面板候选',
  hand_object_contact: '手物接触',
  weighing_paper_operation: '称量纸操作',
  balance_operation: '天平操作',
  pipetting_operation: '移液操作',
  bottle_operation: '试剂瓶操作',
  micro_segment: '动作片段',
  yolo_candidate: '候选动作',
  vlm_supported: '语义证据支持',
  multi_view_support: '双视角支持',
}

const OBJECT_LABELS: Record<string, string> = {
  paper: '称量纸',
  reagent_bottle: '试剂瓶',
  balance: '天平',
  spatula: '药匙',
  container: '容器',
  beaker: '烧杯',
  pipette: '移液器',
  tube: '试管',
  sample_bottle: '样品瓶',
  gloved_hand: '手部',
}

function numberValue(value: unknown) {
  const parsed = Number(value ?? 0)
  return Number.isFinite(parsed) ? parsed : 0
}

function formatNumber(value: unknown) {
  return numberValue(value).toLocaleString('zh-CN')
}

function getCount(overview: VideoMemoryOverview, ...keys: string[]) {
  for (const key of keys) {
    const value = overview.counts?.[key]
    const parsed = numberValue(value)
    if (parsed > 0) return parsed
  }
  return 0
}

function formatDate(value?: string | null) {
  if (!value) return '未记录'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '未记录'
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function formatDay(value?: string | null) {
  if (!value) return '日期未记录'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '日期未记录'
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(date)
}

function labelValue(value?: unknown, fallback = '未记录') {
  const raw = String(value || '').trim()
  if (!raw) return fallback
  return ACTION_LABELS[raw] || OBJECT_LABELS[raw] || cleanDisplayText(raw.replace(/_/g, ' '), fallback)
}

function statusLabel(value?: unknown) {
  const raw = String(value || '').trim().toLowerCase()
  return {
    completed: '已完成',
    analyzed: '已分析',
    partial_failed: '部分待复核',
    failed: '待复核',
    running: '处理中',
    queued: '排队中',
    ready: '已就绪',
    core_ready: '核心已就绪',
    partial: '部分覆盖',
    complete: '完整覆盖',
  }[raw] || labelValue(value, '未记录')
}

function retentionDays(overview: VideoMemoryOverview) {
  return Math.max(1, numberValue(overview.coverage.window_days_expected) || numberValue(overview.coverage.retention_days) || numberValue(overview.scope.retention_days) || 30)
}

function observedWindowDays(overview: VideoMemoryOverview) {
  const retention = retentionDays(overview)
  const explicitDays = numberValue(
    overview.coverage.window_days_available
      ?? overview.counts.window_days_available
      ?? overview.counts.covered_day_count
      ?? overview.counts.coverage_day_count
      ?? overview.counts.observed_day_count,
  )
  if (explicitDays > 0) return Math.min(retention, Math.max(1, Math.ceil(explicitDays)))

  const start = overview.coverage.start_time ? new Date(overview.coverage.start_time).getTime() : NaN
  const endValue = overview.coverage.end_time || overview.coverage.latest_observed_at || overview.generated_at
  const end = endValue ? new Date(endValue).getTime() : NaN
  if (Number.isFinite(start) && Number.isFinite(end) && end >= start) {
    return Math.min(retention, Math.max(1, Math.ceil((end - start) / DAY_MS)))
  }
  return overview.coverage.is_partial ? (getCount(overview, 'included_evidence_items') > 0 ? 1 : 0) : retention
}

function coverageLabel(overview: VideoMemoryOverview) {
  return `${observedWindowDays(overview)}/${retentionDays(overview)} 天数据`
}

function experimentDate(item: VideoMemoryExperimentSummary) {
  return item.latest_observed_at || item.created_at
}

function evidenceAction(item: VideoMemoryEvidenceItem) {
  return labelValue(item.action?.type || item.action?.interaction_type || item.action?.description, '关键动作')
}

function evidenceObject(item: VideoMemoryEvidenceItem) {
  return labelValue(item.action?.primary_object, '')
}

function evidenceTitle(item: VideoMemoryEvidenceItem) {
  const action = evidenceAction(item)
  const object = evidenceObject(item)
  return cleanDisplayText(item.ui?.title || item.action?.description || [action, object].filter(Boolean).join(' / ') || item.evidence_id, item.evidence_id)
}

function evidenceTime(item: VideoMemoryEvidenceItem) {
  const start = numberValue(item.time_range?.start_sec)
  const end = numberValue(item.time_range?.end_sec)
  if (!start && !end) return formatDate(item.observed_at)
  return `${start.toFixed(2)}s 至 ${end.toFixed(2)}s`
}

function evidencePreviewUrl(item: VideoMemoryEvidenceItem) {
  const keyframe = item.ui?.keyframe_url || item.views?.find(view => view.keyframe_url)?.keyframe_url
  const clip = item.ui?.clip_url || item.views?.find(view => view.clip_url)?.clip_url
  return {
    keyframe: keyframe ? mediaUrl(keyframe) : '',
    clip: clip ? mediaUrl(clip) : '',
  }
}

function buildActionDistribution(items: VideoMemoryEvidenceItem[]) {
  const counts = new Map<string, number>()
  for (const item of items) {
    const label = evidenceAction(item)
    counts.set(label, (counts.get(label) || 0) + 1)
  }
  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
}

function buildDailyExperiments(experiments: VideoMemoryExperimentSummary[]) {
  const byDay = new Map<string, VideoMemoryExperimentSummary[]>()
  for (const experiment of experiments) {
    const day = formatDay(experimentDate(experiment))
    const rows = byDay.get(day) || []
    rows.push(experiment)
    byDay.set(day, rows)
  }
  return Array.from(byDay.entries()).slice(0, 10)
}

function reviewItems(overview: VideoMemoryOverview) {
  const items: string[] = []
  if (overview.coverage.is_partial) items.push(`当前只整理了 ${coverageLabel(overview)}，还有部分日期需要补齐。`)
  if (!overview.core.available) items.push('日报仍在整理中，部分实验内容可能暂时没有纳入。')
  const riskyExperiments = overview.experiments.filter(item => String(item.status || '').toLowerCase().includes('failed'))
  if (riskyExperiments.length > 0) items.push(`${riskyExperiments.length} 个实验需要确认是否已正确完成分析。`)
  const retrievalReady = getCount(overview, 'retrieval_ready_count')
  const evidenceItems = getCount(overview, 'included_evidence_items', 'source_evidence_items')
  if (evidenceItems > 0 && retrievalReady === 0) items.push('已经识别到实验操作，但关键素材还没有整理成可回看的列表。')
  return items
}

function snapshotId(overview: VideoMemoryOverview) {
  const value = overview.snapshot?.snapshot_id
  return typeof value === 'string' && value.trim() ? value : undefined
}

function answerClaims(answer: VideoMemoryQueryAnswer) {
  return answer.evidence_trace?.claims?.length ? answer.evidence_trace.claims : (answer.claims || [])
}

function valueHasContent(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(valueHasContent)
  if (value && typeof value === 'object') return Object.values(value as Record<string, unknown>).some(valueHasContent)
  return String(value ?? '').trim().length > 0
}

function claimHasEvidence(claim: Record<string, unknown>) {
  return [
    'evidence_item_ids',
    'ledger_event_ids',
    'evidence_bundle_ids',
    'material_ids',
    'evidence_links',
    'keyframe_refs',
    'keyclip_refs',
  ].some(key => valueHasContent(claim[key]))
}

function answerHasEvidence(answer: VideoMemoryQueryAnswer | null) {
  if (!answer) return false
  return answerClaims(answer).some(claimHasEvidence)
}

function guardedAnswerText(answer: VideoMemoryQueryAnswer) {
  if (!answerHasEvidence(answer)) return '暂时没有找到足够清楚的实验记录来回答这个问题。'
  return cleanDisplayText(answer.answer_summary || answer.answer?.text || '已根据当前日报中的实验记录回答。')
}

export default function VideoMemoryOverviewPage() {
  const { id } = useParams<{ id?: string }>()
  const [overview, setOverview] = useState<VideoMemoryOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [queryText, setQueryText] = useState('')
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryAnswer, setQueryAnswer] = useState<VideoMemoryQueryAnswer | null>(null)
  const [queryError, setQueryError] = useState<string | null>(null)

  async function load(force = false) {
    setLoading(true)
    setError(null)
    try {
      const data = id
        ? await experimentApi.getExperimentVideoMemoryOverview(id, { retention_days: 30, max_items: 200 }, { force })
        : await experimentApi.getVideoMemoryOverview({ retention_days: 30, max_items: 200 }, { force })
      setOverview(data)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '实验室日报加载失败')
      setOverview(null)
    } finally {
      setLoading(false)
    }
  }

  async function runQuery(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault()
    const query = queryText.trim()
    if (!query) return
    setQueryLoading(true)
    setQueryError(null)
    try {
      const data = await experimentApi.queryVideoMemory({
        query,
        limit: 3,
        snapshot_id: overview ? snapshotId(overview) : undefined,
        filters: id ? { experiment_id: id } : undefined,
      })
      setQueryAnswer(data)
    } catch (exc) {
      setQueryError(exc instanceof Error ? exc.message : '证据链查询失败')
      setQueryAnswer(null)
    } finally {
      setQueryLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [id])

  useEffect(() => {
    setQueryAnswer(null)
    setQueryError(null)
    setQueryText('')
  }, [id])

  const topEvidence = useMemo(() => (overview?.evidence_items || []).slice(0, 8), [overview])
  const dailyExperiments = useMemo(() => buildDailyExperiments(overview?.experiments || []), [overview])
  const actionDistribution = useMemo(() => buildActionDistribution(overview?.evidence_items || []), [overview])
  const reviewNotes = useMemo(() => overview ? reviewItems(overview) : [], [overview])

  if (loading && !overview) return <EmptyEvidence title="正在加载实验室日报..." />
  if (error) return <EvidenceCard className="border-red-200 bg-red-50 p-5 text-red-700">{error}</EvidenceCard>
  if (!overview) return <EmptyEvidence title="暂无实验室日报" />

  const evidenceCount = getCount(overview, 'included_evidence_items', 'source_evidence_items') || overview.evidence_items.length
  const clusterCount = getCount(overview, 'cluster_count') || overview.clusters.length
  const coveredDays = observedWindowDays(overview)
  const experimentCount = overview.experiments.length
  const materialCount = getCount(overview, 'retrieval_ready_count', 'formal_material_reference_count', 'published_material_count')
  const queryHasEvidence = answerHasEvidence(queryAnswer)

  const content = (
    <div className="space-y-5">
      <PageHero
        eyebrow={<Link to={id ? `/experiments/${id}/workspace` : '/experiments'} className="hover:text-slate-900">{id ? '实验工作台' : '实验列表'}</Link>}
        title={id ? '实验日报' : '实验室日报'}
        description={id ? '汇总这个实验做了什么、留下了哪些关键素材、还有哪些需要确认。' : '把多天实验整理成一份可读日报：做了哪些实验、沉淀了哪些素材、哪些地方还需要确认。'}
        actions={(
          <>
            <button type="button" onClick={() => void load(true)} className={secondaryButtonClass()}>
              <RefreshCw className="h-4 w-4" />
              刷新
            </button>
            {!id && <Link to="/experiments" className={primaryButtonClass('blue')}><Boxes className="h-4 w-4" />实验列表</Link>}
          </>
        )}
      />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricTile label="已整理天数" value={coveredDays} helper={`近 ${retentionDays(overview)} 天实验记录`} tone="blue" Icon={CalendarDays} />
        <MetricTile label="实验记录" value={experimentCount} helper="纳入日报的实验" tone="emerald" Icon={BookOpenCheck} />
        <MetricTile label="关键操作" value={evidenceCount} helper={`${clusterCount} 类常见操作`} tone="violet" Icon={BrainCircuit} />
        <MetricTile label="关键素材" value={materialCount} helper="可回看、可复用的素材" tone="cyan" Icon={DatabaseZap} />
      </section>

      <EvidenceCard className="p-5">
        <h2 className="text-lg font-black text-slate-950">今天这份日报说明什么</h2>
        <p className="mt-2 text-sm leading-7 text-slate-700">
          系统已经把 {coverageLabel(overview)} 里的 {experimentCount} 个实验整理进日报，
          提炼出 {evidenceCount} 条关键操作记录，并沉淀了 {materialCount} 个可回看的关键素材。
          这份日报可以帮助你快速了解最近实验室做了什么、哪些素材值得复用、哪些内容还需要人工确认。
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          <EvidenceBadge tone={overview.coverage.is_partial ? 'amber' : 'emerald'}>{overview.coverage.is_partial ? '部分日期仍在补齐' : '日期覆盖完整'}</EvidenceBadge>
          <EvidenceBadge tone={overview.core.available ? 'emerald' : 'amber'}>{overview.core.available ? '日报已生成' : '日报仍在整理'}</EvidenceBadge>
          <EvidenceBadge tone="slate">生成时间 {formatDate(overview.generated_at)}</EvidenceBadge>
        </div>
      </EvidenceCard>

      <section className="grid gap-5 xl:grid-cols-[minmax(0,1.2fr)_minmax(22rem,0.8fr)]">
        <main className="space-y-5">
          <EvidenceCard className="p-5">
            <h2 className="mb-4 text-lg font-black text-slate-950">每天做了哪些实验</h2>
            {dailyExperiments.length === 0 ? (
              <EmptyEvidence title="暂无实验记录" description="有实验进入日报后，会按日期汇总在这里。" />
            ) : (
              <div className="space-y-3">
                {dailyExperiments.map(([day, rows]) => (
                  <div key={day} className="rounded-xl border border-slate-200 p-4">
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <h3 className="font-black text-slate-950">{day}</h3>
                      <EvidenceBadge tone="blue">{rows.length} 个实验</EvidenceBadge>
                    </div>
                    <div className="grid gap-2 md:grid-cols-2">
                      {rows.slice(0, 6).map(item => (
                        <Link key={item.experiment_id} to={`/experiments/${item.experiment_id}/report`} className="rounded-lg border border-slate-100 bg-slate-50 p-3 transition hover:bg-white hover:shadow-sm">
                          <div className="line-clamp-2 text-sm font-black text-slate-950">{cleanDisplayText(item.title || item.experiment_id, item.experiment_id)}</div>
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            <EvidenceBadge tone={toneForStatus(item.status)}>{statusLabel(item.status)}</EvidenceBadge>
                            <EvidenceBadge tone="slate">{formatNumber(item.counts?.evidence_items || 0)} 条关键操作</EvidenceBadge>
                            <EvidenceBadge tone="slate">{formatNumber(item.counts?.micro_segments || 0)} 个视频片段</EvidenceBadge>
                          </div>
                        </Link>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </EvidenceCard>

          <EvidenceCard className="p-5">
            <h2 className="mb-4 text-lg font-black text-slate-950">值得回看的关键素材</h2>
            {topEvidence.length === 0 ? (
              <EmptyEvidence title="暂无关键素材" description="关键帧和关键片段确认后，会作为可回看素材显示在这里。" />
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                {topEvidence.map(item => {
                  const preview = evidencePreviewUrl(item)
                  return (
                    <EvidenceCard key={item.evidence_id} className="overflow-hidden">
                      {preview.keyframe ? (
                        <img src={preview.keyframe} alt={evidenceTitle(item)} className="aspect-video w-full bg-slate-100 object-cover" />
                      ) : preview.clip ? (
                        <video src={preview.clip} className="aspect-video w-full bg-slate-950 object-contain" controls preload="metadata" />
                      ) : (
                        <div className="flex aspect-video items-center justify-center bg-slate-100 text-sm font-bold text-slate-400">暂无预览</div>
                      )}
                      <div className="p-3">
                        <div className="line-clamp-2 text-sm font-black text-slate-950">{evidenceTitle(item)}</div>
                        <div className="mt-2 space-y-1 text-xs font-semibold text-slate-500">
                          <div>发生时间 {evidenceTime(item)}</div>
                          <div>实验操作 {evidenceAction(item)}</div>
                          {evidenceObject(item) ? <div>相关物品 {evidenceObject(item)}</div> : null}
                        </div>
                      </div>
                    </EvidenceCard>
                  )
                })}
              </div>
            )}
          </EvidenceCard>
        </main>

        <aside className="space-y-5">
          <EvidenceCard className="p-4">
            <h2 className="mb-4 text-lg font-black text-slate-950">最近常见实验操作</h2>
            {actionDistribution.length === 0 ? (
              <EmptyEvidence title="暂无操作分布" />
            ) : (
              <div className="space-y-3">
                {actionDistribution.map(([label, count]) => {
                  const width = Math.max(8, Math.min(100, Math.round((count / Math.max(1, evidenceCount)) * 100)))
                  return (
                    <div key={label}>
                      <div className="mb-1 flex justify-between gap-3 text-sm font-bold text-slate-700">
                        <span>{label}</span>
                        <span>{count}</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                        <div className="h-full rounded-full bg-blue-500" style={{ width: `${width}%` }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </EvidenceCard>

          <EvidenceCard className="p-4">
            <h2 className="mb-4 text-lg font-black text-slate-950">需要你确认的事项</h2>
            {reviewNotes.length === 0 ? (
              <p className="text-sm font-semibold text-slate-600">暂无需要优先处理的事项。</p>
            ) : (
              <div className="space-y-2">
                {reviewNotes.map((note, index) => (
                  <div key={`${note}-${index}`} className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm font-semibold text-amber-900">
                    {note}
                  </div>
                ))}
              </div>
            )}
          </EvidenceCard>

          <EvidenceCard className="p-4">
            <h2 className="mb-4 text-lg font-black text-slate-950">素材与实验沉淀</h2>
            <div className="space-y-2 text-sm font-semibold text-slate-600">
              <Detail label="实验记录" value={`${experimentCount} 个`} />
              <Detail label="关键操作" value={`${evidenceCount} 条`} />
              <Detail label="常见操作类型" value={`${clusterCount} 类`} />
              <Detail label="关键素材" value={`${materialCount} 个`} />
              <Detail label="双视角素材" value={`${getCount(overview, 'dual_view_item_count')} 个`} />
            </div>
          </EvidenceCard>

          <EvidenceCard className="p-4">
            <h2 className="mb-3 text-lg font-black text-slate-950">建议下一步</h2>
            <p className="text-sm leading-6 text-slate-600">
              优先确认待处理实验和缺失素材；对高频实验操作补齐第一人称与第三人称画面；把质量较好的关键帧和关键片段沉淀到对应实验的关键素材页。
            </p>
          </EvidenceCard>
        </aside>
      </section>

      <EvidenceCard className="p-5">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-black text-slate-950">问问这份日报</h2>
            <p className="text-sm font-semibold text-slate-500">可以询问最近做过哪些实验、哪些操作最常见、哪些素材值得复用。</p>
          </div>
          <MessageSquareText className="h-5 w-5 text-blue-600" />
        </div>
        <form onSubmit={runQuery} className="flex flex-col gap-2 sm:flex-row">
          <input
            value={queryText}
            onChange={event => setQueryText(event.target.value)}
            placeholder="例如：最近有哪些移液相关素材可以复用？"
            className="h-10 min-w-0 flex-1 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-800 outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
          />
          <button type="submit" disabled={queryLoading || !queryText.trim()} className={`${primaryButtonClass('blue')} h-10 disabled:cursor-not-allowed disabled:opacity-50`}>
            <Send className="h-4 w-4" />
            {queryLoading ? '查询中' : '查询'}
          </button>
        </form>
        {queryError ? <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm font-semibold text-red-700">{queryError}</div> : null}
        {queryAnswer ? (
          <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
            <div className="mb-2 flex flex-wrap gap-2">
              <EvidenceBadge tone={queryHasEvidence ? 'emerald' : 'amber'}>{queryHasEvidence ? '已找到相关记录' : '记录不足'}</EvidenceBadge>
            </div>
            <div className={`flex gap-2 text-sm font-semibold leading-6 ${queryHasEvidence ? 'text-slate-700' : 'text-amber-800'}`}>
              {!queryHasEvidence && <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />}
              <span>{guardedAnswerText(queryAnswer)}</span>
            </div>
          </div>
        ) : null}
      </EvidenceCard>
    </div>
  )

  return id ? <ExperimentPageShell experimentId={id}>{content}</ExperimentPageShell> : content
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3 border-b border-slate-100 pb-2">
      <span className="shrink-0 text-slate-400">{label}</span>
      <span className="min-w-0 break-words text-right text-slate-800">{value}</span>
    </div>
  )
}
