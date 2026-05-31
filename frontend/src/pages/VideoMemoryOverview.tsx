import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  AlertTriangle,
  BookOpenCheck,
  BrainCircuit,
  Boxes,
  Clock3,
  DatabaseZap,
  Layers3,
  MessageSquareText,
  RefreshCw,
  Route,
  SearchCheck,
  Send,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { experimentApi } from '../api'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, primaryButtonClass, secondaryButtonClass, toneForStatus, type Tone } from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import type { VideoMemoryCluster, VideoMemoryEvidenceItem, VideoMemoryOverview, VideoMemoryQueryAnswer, VideoMemoryTaskStatus } from '../types'

const DAY_MS = 24 * 60 * 60 * 1000

const TASK_LABELS: Record<string, string> = {
  T1: '动作事件标准化',
  T2: 'VLM 缓存与证据追踪',
  T3: '30 天动作记忆聚类评分',
  T4: '30 天动作记忆聚类生命周期',
  T5: '实验室日报',
  T6: '证据链回答',
  T7: '人工反馈记录 dry-run',
  T8: '证据包相对资源 URI',
  T9: 'YOLO 支撑的动作事件候选证据',
  T10: '微片段记忆单元',
  T11: '双视角时间对齐',
  T12: '物理变化日志覆盖',
  T13: '检索与向量元数据',
  T14: 'API 与态势快照页面',
}

const STATUS_LABELS: Record<string, string> = {
  core_ready: '核心就绪',
  thin_wrapper_partial: '部分就绪',
  waiting_for_core: '等待核心模块',
  ready: '数据就绪',
  needs_source_data: '需要来源数据',
  completed: '已完成',
  complete: '完整窗口',
  partial: '部分窗口',
  running: '运行中',
  queued: '排队中',
  failed: '失败',
  supported: '有证据支持',
  unsupported: '无可引用证据',
}

const CLUSTER_LABELS: Record<string, string> = {
  promoted: '已提升',
  active: '活跃',
  candidate: '候选',
  archived: '已归档',
}

const TOKEN_LABELS: Record<string, string> = {
  hand_object_interaction: '手物交互',
  object_move: '物体移动',
  micro_segment: '微片段',
  yolo_candidate: 'YOLO 候选',
  micro_segment_confirmed: '微片段确认',
  vlm_supported: 'VLM 支持',
  multi_view_support: '双视角支持',
  repeat_observation: '重复观察',
  recent_observation: '近期观察',
  confidence_mean: '平均置信度',
  trace_support: '证据链支持',
}

function numberValue(value: unknown) {
  const parsed = Number(value ?? 0)
  return Number.isFinite(parsed) ? parsed : 0
}

function formatNumber(value: unknown) {
  return numberValue(value).toLocaleString('en-US')
}

function formatDate(value?: string | null) {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '-'
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function displayToken(value?: unknown, fallback = '-') {
  const raw = String(value || '').trim()
  if (!raw) return fallback
  return TOKEN_LABELS[raw] || STATUS_LABELS[raw] || CLUSTER_LABELS[raw] || cleanDisplayText(raw.replace(/_/g, ' '), fallback)
}

function taskTone(task: VideoMemoryTaskStatus): Tone {
  const value = String(task.status || '').toLowerCase()
  if (value.includes('ready')) return 'emerald'
  if (value.includes('partial')) return 'amber'
  if (value.includes('waiting')) return 'slate'
  if (value.includes('needs')) return 'amber'
  return toneForStatus(value)
}

function taskLabel(task: VideoMemoryTaskStatus) {
  return TASK_LABELS[task.task_id] || cleanDisplayText(task.label, task.task_id)
}

function statusLabel(value?: unknown) {
  return displayToken(value, '未知状态')
}

function clusterTone(cluster: VideoMemoryCluster): Tone {
  return toneForStatus(cluster.lifecycle_state)
}

function clusterLabel(cluster: VideoMemoryCluster) {
  return CLUSTER_LABELS[String(cluster.lifecycle_state || '')] || statusLabel(cluster.lifecycle_state)
}

function evidenceTitle(item: VideoMemoryEvidenceItem) {
  return cleanDisplayText(item.ui?.title || item.action?.description || item.evidence_id, item.evidence_id)
}

function evidenceTime(item: VideoMemoryEvidenceItem) {
  const start = numberValue(item.time_range?.start_sec)
  const end = numberValue(item.time_range?.end_sec)
  if (!start && !end) return formatDate(item.observed_at)
  return `${start.toFixed(2)}-${end.toFixed(2)}s`
}

function evidencePreview(item: VideoMemoryEvidenceItem) {
  const keyframe = item.ui?.keyframe_url || item.views.find(view => view.keyframe_url)?.keyframe_url
  const clip = item.ui?.clip_url || item.views.find(view => view.clip_url)?.clip_url
  if (keyframe) return <img src={keyframe} alt={evidenceTitle(item)} className="aspect-video w-full bg-slate-100 object-cover" />
  if (clip) return <video src={clip} className="aspect-video w-full bg-slate-950 object-contain" controls preload="metadata" />
  return <div className="flex aspect-video items-center justify-center bg-slate-100 text-sm font-bold text-slate-400">仅元数据</div>
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

  if (!overview.coverage.is_partial) return retention
  return numberValue(overview.counts.included_evidence_items) > 0 ? 1 : 0
}

function coverageWindowLabel(overview: VideoMemoryOverview) {
  return `基于 ${observedWindowDays(overview)}/${retentionDays(overview)} 天数据`
}

function limitationLabel(value: string) {
  if (value.includes('metadata-only')) return '态势快照只读取元数据，不解码视频像素。'
  if (value.includes('thin wrapper')) return '核心视频记忆模块不可用时，LabSOPGuard 使用轻量封装生成评分与快照。'
  if (value.includes('package://')) return '证据包资源使用 package:// 相对 URI；浏览器链接只指向实验输出目录内可服务文件。'
  return cleanDisplayText(value)
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

function answerWindowNotice(answer: VideoMemoryQueryAnswer) {
  const explicit = answer.partial_window_notice_zh || answer.window_scope?.partial_window_notice_zh
  if (explicit) return explicit
  const available = numberValue(answer.window_scope?.window_days_available ?? answer.window_days_available)
  const expected = numberValue(answer.window_scope?.window_days_expected ?? answer.window_days_expected) || 30
  if (answer.partial_window || !answer.is_full_30_day_memory) return `基于 ${available}/${expected} 天数据`
  return `基于完整 ${expected}/${expected} 天数据`
}

function guardedAnswerText(answer: VideoMemoryQueryAnswer) {
  if (!answerHasEvidence(answer)) return '未找到可引用动作事件或证据包，不能形成强结论。'
  return cleanDisplayText(answer.answer_summary || answer.answer?.text || '找到可追溯证据，已按证据链回答。')
}

function claimTitle(claim: Record<string, unknown>, index: number) {
  return cleanDisplayText(String(claim.claim_text || claim.cluster_id || claim.evidence_bundle_id || `证据链 ${index + 1}`))
}

export default function VideoMemoryOverviewPage() {
  const { id } = useParams<{ id?: string }>()
  const [overview, setOverview] = useState<VideoMemoryOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [queryText, setQueryText] = useState('')
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryError, setQueryError] = useState<string | null>(null)
  const [queryAnswer, setQueryAnswer] = useState<VideoMemoryQueryAnswer | null>(null)

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
      setQueryError(exc instanceof Error ? exc.message : '证据链回答查询失败')
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

  const topEvidence = useMemo(() => (overview?.evidence_items || []).slice(0, 12), [overview])
  const topClusters = useMemo(() => (overview?.clusters || []).slice(0, 8), [overview])

  if (loading && !overview) return <EmptyEvidence title="正在加载实验室日报..." />
  if (error) return <EvidenceCard className="border-red-200 bg-red-50 p-5 text-red-700">{error}</EvidenceCard>
  if (!overview) return <EmptyEvidence title="暂无实验室日报" />

  const counts = overview.counts || {}
  const title = id ? '实验室日报' : '实验室日报'
  const coreStatus = overview.core.available ? '核心就绪' : statusLabel(overview.core.status)
  const coverageLabel = coverageWindowLabel(overview)
  const queryHasEvidence = answerHasEvidence(queryAnswer)
  const queryClaims = queryAnswer ? answerClaims(queryAnswer).filter(claimHasEvidence).slice(0, 3) : []

  return (
    <div className="space-y-5">
      <PageHero
        eyebrow={<Link to={id ? `/experiments/${id}/workspace` : '/experiments'} className="hover:text-slate-900">{id ? '实验工作台' : '实验队列'}</Link>}
        title={title}
        description={`以动作事件、证据包、每日动作账本和动作序列模式生成的长期视频记忆入口。日报作为下钻视图，不替代实验室日报。${coverageLabel}。`}
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

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4" data-smoke="video-memory-metrics">
        <MetricTile label="动作事件" value={formatNumber(counts.included_evidence_items)} helper={`${formatNumber(counts.source_evidence_items)} 条来源记录`} tone="blue" Icon={BrainCircuit} />
        <MetricTile label="30 天动作记忆聚类" value={formatNumber(counts.cluster_count)} helper={`${formatNumber(counts.promoted_cluster_count)} 个主聚类`} tone="emerald" Icon={Layers3} />
        <MetricTile label="每日动作账本" value={formatNumber(counts.covered_day_count)} helper={`${formatNumber(counts.micro_segment_count)} 个微片段`} tone="cyan" Icon={BookOpenCheck} />
        <MetricTile label="记忆状态" value={coreStatus} helper={`${overview.core.missing_symbols?.length || 0} 个缺失模块`} tone={overview.core.available ? 'emerald' : 'amber'} Icon={DatabaseZap} />
      </section>

      <section className="grid gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(20rem,0.8fr)]">
        <EvidenceCard className="p-4" data-smoke="video-memory-tasks">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <h3 className="text-lg font-black text-slate-950">态势快照构建状态</h3>
              <p className="text-sm font-semibold text-slate-500">{overview.contract_version}</p>
            </div>
            <EvidenceBadge tone={overview.coverage.is_partial ? 'amber' : 'emerald'}>{overview.coverage.is_partial ? `部分窗口 · ${coverageLabel}` : `完整窗口 · ${coverageLabel}`}</EvidenceBadge>
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            {overview.tasks.map(task => (
              <div key={task.task_id} className="rounded-lg border border-slate-200 p-3">
                <div className="mb-2 flex items-start justify-between gap-2">
                  <div className="font-mono text-xs font-black text-blue-700">{task.task_id}</div>
                  <EvidenceBadge tone={taskTone(task)}>{statusLabel(task.status)}</EvidenceBadge>
                </div>
                <div className="font-black text-slate-950">{taskLabel(task)}</div>
                {task.required_symbols && task.required_symbols.length > 0 && (
                  <div className="mt-2 break-words text-xs font-semibold text-slate-500">{task.required_symbols.join(', ')}</div>
                )}
              </div>
            ))}
          </div>
        </EvidenceCard>

        <EvidenceCard className="p-4">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-lg font-black text-slate-950">态势窗口</h3>
            <EvidenceBadge tone={overview.coverage.is_partial ? 'amber' : 'emerald'}>{coverageLabel}</EvidenceBadge>
          </div>
          <div className="mt-4 space-y-3 text-sm font-semibold text-slate-600">
            <div className="flex items-center gap-2"><Clock3 className="h-4 w-4 text-blue-600" />{formatDate(overview.coverage.start_time)} 至 {formatDate(overview.coverage.end_time)}</div>
            <div className="flex items-center gap-2"><ShieldCheck className="h-4 w-4 text-emerald-600" />{coverageLabel}，保留窗口 {retentionDays(overview)} 天</div>
            <div className="flex items-center gap-2"><SearchCheck className="h-4 w-4 text-cyan-600" />{formatNumber(counts.retrieval_ready_count)} 个可检索证据包</div>
            <div className="flex items-center gap-2"><Route className="h-4 w-4 text-violet-600" />{formatNumber(counts.dual_view_item_count)} 个双视角动作事件</div>
            <div className="flex items-center gap-2"><Sparkles className="h-4 w-4 text-amber-600" />{formatNumber(counts.vlm_cache_trace_count)} 条 VLM 缓存轨迹</div>
          </div>
          {overview.limitations && overview.limitations.length > 0 && (
            <div className="mt-4 rounded-lg bg-slate-50 p-3 text-xs font-semibold leading-5 text-slate-500">
              {overview.limitations.map(item => <div key={item}>{limitationLabel(item)}</div>)}
            </div>
          )}
        </EvidenceCard>
      </section>

      <EvidenceCard className="p-4" data-smoke="video-memory-query">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-lg font-black text-slate-950">证据链回答</h3>
            <p className="text-sm font-semibold text-slate-500">{coverageLabel}</p>
          </div>
          <EvidenceBadge tone={overview.coverage.is_partial ? 'amber' : 'emerald'}>{overview.coverage.is_partial ? '部分窗口' : '完整窗口'}</EvidenceBadge>
        </div>
        <form onSubmit={runQuery} className="flex flex-col gap-2 sm:flex-row">
          <div className="relative min-w-0 flex-1">
            <MessageSquareText className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              value={queryText}
              onChange={event => setQueryText(event.target.value)}
              placeholder="查询动作事件、证据包或动作序列模式"
              className="h-10 w-full rounded-lg border border-slate-200 bg-white pl-9 pr-3 text-sm font-semibold text-slate-800 outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
            />
          </div>
          <button type="submit" disabled={queryLoading || !queryText.trim()} className={`${primaryButtonClass('blue')} h-10 disabled:cursor-not-allowed disabled:opacity-50`}>
            <Send className="h-4 w-4" />
            {queryLoading ? '查询中' : '查询'}
          </button>
        </form>
        {queryError && <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm font-semibold text-red-700">{queryError}</div>}
        {queryAnswer ? (
          <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
            <div className="mb-2 flex flex-wrap gap-2">
              <EvidenceBadge tone={queryHasEvidence ? 'emerald' : 'amber'}>{queryHasEvidence ? '有证据支持' : '无可引用证据'}</EvidenceBadge>
              <EvidenceBadge tone={queryAnswer.partial_window ? 'amber' : 'emerald'}>{answerWindowNotice(queryAnswer)}</EvidenceBadge>
              <EvidenceBadge tone="slate">置信度 {Number(queryAnswer.confidence || 0).toFixed(2)}</EvidenceBadge>
            </div>
            <div className={`flex gap-2 text-sm font-semibold leading-6 ${queryHasEvidence ? 'text-slate-700' : 'text-amber-800'}`}>
              {!queryHasEvidence && <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />}
              <span>{guardedAnswerText(queryAnswer)}</span>
            </div>
            {queryClaims.length > 0 && (
              <div className="mt-3 grid gap-2 md:grid-cols-3">
                {queryClaims.map((claim, index) => (
                  <div key={String(claim.claim_id || claim.evidence_bundle_id || index)} className="rounded-lg border border-slate-200 bg-white p-3">
                    <div className="line-clamp-2 font-black text-slate-950">{claimTitle(claim, index)}</div>
                    <div className="mt-2 space-y-1 text-xs font-semibold text-slate-500">
                      <div>证据包 {formatNumber(Array.isArray(claim.evidence_bundle_ids) ? claim.evidence_bundle_ids.length : claim.evidence_bundle_id ? 1 : 0)}</div>
                      <div>动作事件 {formatNumber(Array.isArray(claim.ledger_event_ids) ? claim.ledger_event_ids.length : claim.ledger_event_id ? 1 : 0)}</div>
                      <div>人工反馈记录 {statusLabel(claim.human_confirmation_status)}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <EmptyEvidence title="等待证据链问题" />
        )}
      </EvidenceCard>

      {overview.experiments.length > 0 && (
        <EvidenceCard className="p-4" data-smoke="video-memory-experiments">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h3 className="text-lg font-black text-slate-950">实验覆盖</h3>
            <EvidenceBadge tone="blue">{overview.experiments.length}</EvidenceBadge>
          </div>
          <div className="grid gap-3 xl:grid-cols-2">
            {overview.experiments.slice(0, 12).map(item => (
              <div key={item.experiment_id} className="rounded-lg border border-slate-200 p-3">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <Link to={`/experiments/${item.experiment_id}/video-memory`} className="font-black text-slate-950 hover:text-blue-700">{cleanDisplayText(item.title || item.experiment_id, item.experiment_id)}</Link>
                  <EvidenceBadge tone={toneForStatus(item.status)}>{statusLabel(item.status || 'unknown')}</EvidenceBadge>
                </div>
                <div className="mt-2 grid gap-2 text-xs font-bold text-slate-500 sm:grid-cols-4">
                  <span>{formatNumber(item.counts?.evidence_items)} 动作事件</span>
                  <span>{formatNumber(item.counts?.micro_segments)} 微片段</span>
                  <span>{formatNumber(item.counts?.vectors)} 向量</span>
                  <span>{statusLabel(item.alignment?.status || '未记录对齐')}</span>
                </div>
              </div>
            ))}
          </div>
        </EvidenceCard>
      )}

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1.3fr)_minmax(22rem,0.7fr)]">
        <EvidenceCard className="p-4" data-smoke="video-memory-evidence">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h3 className="text-lg font-black text-slate-950">最近动作事件</h3>
            <EvidenceBadge tone="blue">{topEvidence.length}</EvidenceBadge>
          </div>
            {topEvidence.length === 0 ? <EmptyEvidence title="当前实验室日报暂无动作事件" /> : (
            <div className="grid gap-4 md:grid-cols-2">
              {topEvidence.map(item => (
                <div key={item.evidence_id} className="overflow-hidden rounded-lg border border-slate-200">
                  {evidencePreview(item)}
                  <div className="p-3">
                    <div className="mb-2 flex flex-wrap gap-1.5">
                      <EvidenceBadge tone="blue">{evidenceTime(item)}</EvidenceBadge>
                      <EvidenceBadge tone={item.views.length >= 2 ? 'emerald' : 'amber'}>{item.views.length} 视角</EvidenceBadge>
                      {item.retrieval?.score !== undefined && <EvidenceBadge tone="slate">{Number(item.retrieval.score).toFixed(2)}</EvidenceBadge>}
                    </div>
                    <div className="line-clamp-2 font-black text-slate-950">{evidenceTitle(item)}</div>
                    <div className="mt-1 break-all text-xs font-semibold text-slate-500">{item.evidence_id}</div>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {[item.action?.type, item.action?.primary_object, item.action?.interaction_type].filter(Boolean).map(label => (
                        <span key={String(label)} className="rounded bg-slate-100 px-2 py-1 text-xs font-bold text-slate-600">{displayToken(label)}</span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </EvidenceCard>

        <EvidenceCard className="p-4" data-smoke="video-memory-clusters">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h3 className="text-lg font-black text-slate-950">30 天动作记忆聚类</h3>
            <EvidenceBadge tone="emerald">{topClusters.length}</EvidenceBadge>
          </div>
          {topClusters.length === 0 ? <EmptyEvidence title="暂无动作记忆聚类" /> : (
            <div className="space-y-3">
              {topClusters.map(cluster => (
                <div key={cluster.cluster_id} className="rounded-lg border border-slate-200 p-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="break-all font-black text-slate-950">{cluster.cluster_id}</div>
                      <div className="mt-1 text-xs font-bold text-slate-500">{formatDate(cluster.last_observed_at)} / {cluster.evidence_item_ids?.length || cluster.evidence_count || 0} 条证据</div>
                    </div>
                    <EvidenceBadge tone={clusterTone(cluster)}>{clusterLabel(cluster)}</EvidenceBadge>
                  </div>
                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
                    <div className="h-full rounded-full bg-emerald-500" style={{ width: `${Math.round(Math.max(0, Math.min(1, Number(cluster.score || 0))) * 100)}%` }} />
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {(cluster.score_reasons || []).slice(0, 4).map(reason => <EvidenceBadge key={reason}>{displayToken(reason)}</EvidenceBadge>)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </EvidenceCard>
      </section>
    </div>
  )
}
