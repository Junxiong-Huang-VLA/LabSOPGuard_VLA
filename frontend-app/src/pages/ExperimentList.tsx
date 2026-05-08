import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Archive,
  BarChart3,
  CheckCircle2,
  CircleDot,
  ClipboardCheck,
  Clock3,
  FileText,
  FlaskConical,
  Layers3,
  MoreVertical,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  UploadCloud,
  X,
  XCircle,
} from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import { cleanDisplayText } from '../displayText'
import type { Experiment } from '../types'

type StatusFilter = 'all' | 'created' | 'uploaded' | 'queued' | 'running' | 'completed' | 'failed' | 'archived'
type SortMode = 'newest' | 'oldest' | 'confidence' | 'steps'
type Tone = 'blue' | 'sky' | 'amber' | 'emerald' | 'red' | 'slate' | 'violet'

const EXPERIMENT_PAGE_SIZE = 100

const STATUS_OPTIONS: Array<{ key: StatusFilter; label: string; shortLabel: string; tone: Tone; Icon: typeof FlaskConical }> = [
  { key: 'all', label: '全部 All', shortLabel: '全部', tone: 'blue', Icon: FlaskConical },
  { key: 'created', label: '已创建 Created', shortLabel: '已创建', tone: 'slate', Icon: CircleDot },
  { key: 'uploaded', label: '已上传 Uploaded', shortLabel: '已上传', tone: 'sky', Icon: UploadCloud },
  { key: 'queued', label: '排队中 Queued', shortLabel: '排队中', tone: 'amber', Icon: Clock3 },
  { key: 'running', label: '分析中 Running', shortLabel: '分析中', tone: 'violet', Icon: BarChart3 },
  { key: 'completed', label: '已完成 Completed', shortLabel: '已完成', tone: 'emerald', Icon: CheckCircle2 },
  { key: 'failed', label: '失败 Failed', shortLabel: '失败', tone: 'red', Icon: XCircle },
  { key: 'archived', label: '已归档 Archived', shortLabel: '已归档', tone: 'slate', Icon: Archive },
]

const toneClasses: Record<Tone, { stat: string; badge: string; active: string; icon: string }> = {
  blue: {
    stat: 'border-blue-200 bg-blue-50 text-blue-700',
    badge: 'bg-blue-50 text-blue-700 ring-blue-200',
    active: 'border-blue-600 bg-blue-600 text-white shadow-sm shadow-blue-200',
    icon: 'bg-blue-100 text-blue-700',
  },
  sky: {
    stat: 'border-sky-200 bg-sky-50 text-sky-700',
    badge: 'bg-sky-50 text-sky-700 ring-sky-200',
    active: 'border-sky-600 bg-sky-600 text-white shadow-sm shadow-sky-200',
    icon: 'bg-sky-100 text-sky-700',
  },
  amber: {
    stat: 'border-amber-200 bg-amber-50 text-amber-700',
    badge: 'bg-amber-50 text-amber-700 ring-amber-200',
    active: 'border-amber-500 bg-amber-500 text-white shadow-sm shadow-amber-200',
    icon: 'bg-amber-100 text-amber-700',
  },
  emerald: {
    stat: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    badge: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
    active: 'border-emerald-600 bg-emerald-600 text-white shadow-sm shadow-emerald-200',
    icon: 'bg-emerald-100 text-emerald-700',
  },
  red: {
    stat: 'border-red-200 bg-red-50 text-red-700',
    badge: 'bg-red-50 text-red-700 ring-red-200',
    active: 'border-red-600 bg-red-600 text-white shadow-sm shadow-red-200',
    icon: 'bg-red-100 text-red-700',
  },
  slate: {
    stat: 'border-slate-200 bg-white text-slate-700',
    badge: 'bg-slate-100 text-slate-700 ring-slate-200',
    active: 'border-slate-900 bg-slate-900 text-white shadow-sm shadow-slate-200',
    icon: 'bg-slate-100 text-slate-600',
  },
  violet: {
    stat: 'border-violet-200 bg-violet-50 text-violet-700',
    badge: 'bg-violet-50 text-violet-700 ring-violet-200',
    active: 'border-violet-600 bg-violet-600 text-white shadow-sm shadow-violet-200',
    icon: 'bg-violet-100 text-violet-700',
  },
}

function isArchived(item: Experiment, localArchiveIds: Set<string>) {
  return Boolean(item.archived_at) || localArchiveIds.has(item.experiment_id)
}

function normalizedStatus(item: Experiment): Exclude<StatusFilter, 'all' | 'archived'> {
  const value = String(item.status || '').toLowerCase()
  if (item.processing_error || value === 'failed') return 'failed'
  if (value === 'analyzed' || value === 'completed') return 'completed'
  if (value === 'running') return 'running'
  if (value === 'queued') return 'queued'
  if (value === 'uploaded') return 'uploaded'
  return 'created'
}

function statusLabel(item: Experiment, archived: boolean) {
  if (archived) return '已归档'
  return STATUS_OPTIONS.find(option => option.key === normalizedStatus(item))?.shortLabel || '已创建'
}

function statusTone(item: Experiment, archived: boolean) {
  if (archived) return 'slate'
  return STATUS_OPTIONS.find(option => option.key === normalizedStatus(item))?.tone || 'slate'
}

function formatNumber(value: unknown, digits = 2) {
  if (value === null || value === undefined || value === '') return '-'
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue.toFixed(digits) : '-'
}

function formatDate(value?: string | null) {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '-'
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

function timeValue(value?: string | null) {
  const parsed = Date.parse(value || '')
  return Number.isFinite(parsed) ? parsed : 0
}

function displayTitle(item: Experiment) {
  return cleanDisplayText(item.title, item.experiment_id || '未命名实验')
}

function displayDescription(item: Experiment) {
  return cleanDisplayText(item.description, '')
}

function confidenceValue(item: Experiment) {
  return item.avg_confidence === null || item.avg_confidence === undefined ? Number.NEGATIVE_INFINITY : Number(item.avg_confidence)
}

function keyActionSummary(item: Experiment) {
  const summary = item.key_action_summary || {}
  return {
    status: String(summary.status || 'not_started'),
    segments: Number(summary.segment_count || 0),
    micros: Number(summary.micro_segment_count || 0),
    interactions: Number(summary.interaction_count || summary.raw_yolo_interaction_count || 0),
    vectors: Number(summary.vector_count || 0),
  }
}

function keyActionReady(item: Experiment) {
  const summary = keyActionSummary(item)
  return summary.status === 'completed' && (summary.segments > 0 || summary.micros > 0 || summary.vectors > 0)
}

export default function ExperimentList() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [sortMode, setSortMode] = useState<SortMode>('newest')
  const [visibleCount, setVisibleCount] = useState(18)
  const [openMenuId, setOpenMenuId] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<Experiment | null>(null)
  const [localArchiveIds, setLocalArchiveIds] = useState<Set<string>>(() => new Set())

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const firstPage = await experimentApi.list({ limit: EXPERIMENT_PAGE_SIZE }, { force: true })
      const loaded = [...(firstPage.experiments || [])]
      const total = Number(firstPage.total ?? loaded.length)
      let safetyPageCount = 0
      while (loaded.length < total && safetyPageCount < 100) {
        const nextPage = await experimentApi.list({ limit: EXPERIMENT_PAGE_SIZE, offset: loaded.length }, { force: true })
        const nextExperiments = nextPage.experiments || []
        if (nextExperiments.length === 0) break
        loaded.push(...nextExperiments)
        safetyPageCount += 1
      }
      setExperiments(loaded)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '实验列表加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const uniqueExperiments = useMemo(() => {
    const seen = new Map<string, Experiment>()
    for (const item of experiments) {
      if (!seen.has(item.experiment_id)) seen.set(item.experiment_id, item)
    }
    return Array.from(seen.values())
  }, [experiments])

  const statusCounts = useMemo(() => {
    const counts: Record<StatusFilter, number> = {
      all: 0,
      created: 0,
      uploaded: 0,
      queued: 0,
      running: 0,
      completed: 0,
      failed: 0,
      archived: 0,
    }

    for (const item of uniqueExperiments) {
      if (isArchived(item, localArchiveIds)) {
        counts.archived += 1
      } else {
        counts.all += 1
        counts[normalizedStatus(item)] += 1
      }
    }

    return counts
  }, [localArchiveIds, uniqueExperiments])

  const filtered = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    return uniqueExperiments.filter(item => {
      const archived = isArchived(item, localArchiveIds)
      const status = archived ? 'archived' : normalizedStatus(item)
      const statusMatch = statusFilter === 'all' ? !archived : status === statusFilter
      const queryMatch = !keyword || [
        item.title,
        displayTitle(item),
        item.description,
        displayDescription(item),
        item.experiment_id,
      ].some(value => String(value || '').toLowerCase().includes(keyword))
      return statusMatch && queryMatch
    }).sort((left, right) => {
      if (sortMode === 'oldest') return timeValue(left.created_at) - timeValue(right.created_at)
      if (sortMode === 'confidence') return confidenceValue(right) - confidenceValue(left)
      if (sortMode === 'steps') return Number(right.total_steps || 0) - Number(left.total_steps || 0)
      return timeValue(right.created_at) - timeValue(left.created_at)
    })
  }, [localArchiveIds, query, sortMode, statusFilter, uniqueExperiments])

  useEffect(() => {
    setVisibleCount(18)
    setOpenMenuId(null)
  }, [query, sortMode, statusFilter])

  const visibleExperiments = filtered.slice(0, visibleCount)

  async function archiveExperiment(id: string) {
    const archived = await experimentApi.archive(id)
    setExperiments(previous => previous.map(item => (
      item.experiment_id === id ? { ...item, ...archived, archived_at: archived.archived_at || new Date().toISOString() } : item
    )))
    setLocalArchiveIds(previous => new Set(previous).add(id))
    setOpenMenuId(null)
  }

  async function unarchiveExperiment(id: string) {
    const unarchived = await experimentApi.unarchive(id)
    setExperiments(previous => previous.map(item => (
      item.experiment_id === id ? { ...item, ...unarchived, archived_at: null } : item
    )))
    setLocalArchiveIds(previous => {
      const next = new Set(previous)
      next.delete(id)
      return next
    })
    setOpenMenuId(null)
  }

  async function deleteExperiment(id: string) {
    await experimentApi.delete(id)
    setExperiments(previous => previous.filter(item => item.experiment_id !== id))
    setLocalArchiveIds(previous => {
      const next = new Set(previous)
      next.delete(id)
      return next
    })
    setDeleteTarget(null)
    setOpenMenuId(null)
  }

  return (
    <div className="space-y-6">
      <section className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div className="min-w-0">
          <h2 className="text-2xl font-black tracking-tight text-slate-950 sm:text-3xl">实验列表 Experiment List</h2>
          <p className="mt-2 text-sm font-semibold leading-6 text-slate-500">从这里创建实验、上传视频并启动分析。</p>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:flex sm:flex-wrap">
          <button type="button" onClick={() => void load()} className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-4 text-sm font-bold text-slate-700 transition hover:bg-slate-50">
            <RefreshCw className="h-4 w-4" />
            刷新
          </button>
          <Link to="/upload" className="inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 text-sm font-bold text-white shadow-sm shadow-blue-200 transition hover:bg-blue-700">
            <Plus className="h-4 w-4" />
            新建实验 New
          </Link>
        </div>
      </section>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8">
        {STATUS_OPTIONS.map(option => {
          const Icon = option.Icon
          return (
            <div
              key={option.key}
              className={`min-h-[108px] rounded-lg border p-4 ${toneClasses[option.tone].stat}`}
            >
              <div className="flex items-start justify-between gap-3">
                <span className="text-xs font-black leading-5">{option.label}</span>
                <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${toneClasses[option.tone].icon}`}>
                  <Icon className="h-4 w-4" />
                </span>
              </div>
              <div className="mt-4 text-3xl font-black leading-none">{statusCounts[option.key]}</div>
            </div>
          )
        })}
      </section>

      <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm shadow-slate-200/70">
        <div className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {STATUS_OPTIONS.map(option => (
              <button
                key={option.key}
                type="button"
                onClick={() => setStatusFilter(option.key)}
                className={`rounded-lg border px-3 py-2 text-sm font-bold transition ${
                  statusFilter === option.key
                    ? toneClasses[option.tone].active
                    : 'border-slate-200 bg-slate-50 text-slate-600 hover:bg-slate-100'
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-end">
            <div className="relative min-w-0 md:w-72">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                value={query}
                onChange={event => setQuery(event.target.value)}
                placeholder="搜索实验标题或 ID"
                className="h-10 w-full rounded-lg border border-slate-200 bg-white pl-9 pr-3 text-sm font-semibold text-slate-800 outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
              />
            </div>
            <div className="text-sm font-bold text-slate-500">共 {filtered.length} 项</div>
            <select
              value={sortMode}
              onChange={event => setSortMode(event.target.value as SortMode)}
              className="h-10 rounded-lg border border-slate-200 bg-white px-3 text-sm font-bold text-slate-700 outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100 md:w-auto"
            >
              <option value="newest">最新创建</option>
              <option value="oldest">最早创建</option>
              <option value="confidence">置信度最高</option>
              <option value="steps">步骤最多</option>
            </select>
          </div>
        </div>
      </section>

      {error && <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm font-semibold text-red-700">{error}</div>}
      {loading && <EmptyState title="正在加载实验列表..." />}
      {!loading && filtered.length === 0 && <EmptyState title="暂无匹配实验" description="可以点击新建实验上传视频并启动分析。" />}

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {visibleExperiments.map(item => {
          const archived = isArchived(item, localArchiveIds)
          const title = displayTitle(item)
          const tone = statusTone(item, archived)
          return (
            <article key={item.experiment_id} className="flex min-h-[246px] flex-col rounded-lg border border-slate-200 bg-white p-5 shadow-sm shadow-slate-200/70 transition hover:-translate-y-0.5 hover:shadow-md">
              <div className="flex items-start justify-between gap-3">
                <span className={`inline-flex min-h-6 items-center rounded-md px-2 py-0.5 text-xs font-black ring-1 ${toneClasses[tone].badge}`}>
                  {statusLabel(item, archived)}
                </span>
                <div className="relative">
                  <button
                    type="button"
                    aria-label={`更多操作 ${title}`}
                    onClick={() => setOpenMenuId(openMenuId === item.experiment_id ? null : item.experiment_id)}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 text-slate-500 transition hover:bg-slate-50 hover:text-slate-900"
                  >
                    <MoreVertical className="h-4 w-4" />
                  </button>
                  {openMenuId === item.experiment_id && (
                    <div className="absolute right-0 top-10 z-10 w-44 rounded-lg border border-slate-200 bg-white p-1 shadow-lg">
                      <button type="button" onClick={() => void (archived ? unarchiveExperiment(item.experiment_id) : archiveExperiment(item.experiment_id))} className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm font-bold text-slate-700 hover:bg-slate-50">
                        <Archive className="h-4 w-4" />
                        {archived ? '取消归档 Unarchive' : '归档 Archive'}
                      </button>
                      <button type="button" onClick={() => setDeleteTarget(item)} className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm font-bold text-red-700 hover:bg-red-50">
                        <Trash2 className="h-4 w-4" />
                        删除 Delete
                      </button>
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-4 min-w-0">
                <h3 className="line-clamp-2 text-lg font-black leading-6 text-slate-950">{title}</h3>
                <p className="mt-2 truncate text-xs font-semibold text-slate-400">{item.experiment_id}</p>
              </div>

              <dl className="mt-5 grid grid-cols-2 gap-3 text-sm">
                <Info label="步骤总数" value={String(item.total_steps || 0)} />
                <Info label="平均置信度" value={formatNumber(item.avg_confidence, 3)} />
                <Info label="创建时间" value={formatDate(item.created_at)} wide />
              </dl>

              <KeyActionReadiness item={item} />

              <div className="mt-auto grid grid-cols-1 gap-2 pt-5 sm:grid-cols-2 xl:grid-cols-4">
                <Link
                  to={`/experiments/${item.experiment_id}/workspace`}
                  onMouseEnter={() => prefetchExperimentRoute(item.experiment_id, 'workspace')}
                  className="inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 text-sm font-bold text-white transition hover:bg-blue-700"
                >
                  <FlaskConical className="h-4 w-4" />
                  查看详情
                </Link>
                <Link
                  to={`/experiments/${item.experiment_id}/key-actions`}
                  onMouseEnter={() => prefetchExperimentRoute(item.experiment_id, 'keyActions')}
                  className={`inline-flex h-10 items-center justify-center gap-2 rounded-lg px-4 text-sm font-bold transition ${
                    keyActionReady(item)
                      ? 'bg-emerald-600 text-white hover:bg-emerald-700'
                      : 'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
                  }`}
                >
                  <Layers3 className="h-4 w-4" />
                  关键动作
                </Link>
                <Link
                  to={`/experiments/${item.experiment_id}/key-actions/review`}
                  onMouseEnter={() => prefetchExperimentRoute(item.experiment_id, 'reviewQueue')}
                  className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 text-sm font-bold text-amber-700 transition hover:bg-white"
                >
                  <ClipboardCheck className="h-4 w-4" />
                  Review
                </Link>
                <Link
                  to={`/experiments/${item.experiment_id}/report`}
                  onMouseEnter={() => prefetchExperimentRoute(item.experiment_id, 'report')}
                  className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-4 text-sm font-bold text-slate-700 transition hover:bg-slate-50"
                >
                  <FileText className="h-4 w-4" />
                  分析报告
                </Link>
              </div>
            </article>
          )
        })}
      </section>

      {!loading && filtered.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white p-4 text-sm font-bold text-slate-600">
          <span>显示 {Math.min(visibleCount, filtered.length)} / {filtered.length}</span>
          {visibleCount < filtered.length && (
            <button type="button" onClick={() => setVisibleCount(count => count + 18)} className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-bold text-slate-700 transition hover:bg-slate-50">
              加载更多
            </button>
          )}
        </div>
      )}

      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/40 p-4">
          <section role="dialog" aria-label="删除实验" className="w-full max-w-md rounded-lg bg-white p-5 shadow-xl">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-lg font-black text-slate-950">删除实验</h2>
                <p className="mt-2 text-sm font-semibold leading-6 text-slate-600">永久删除 {displayTitle(deleteTarget)} 前，请确认已经完成归档或备份。</p>
              </div>
              <button type="button" onClick={() => setDeleteTarget(null)} className="rounded-md p-1 text-slate-500 hover:bg-slate-100">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button type="button" onClick={() => setDeleteTarget(null)} className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-bold text-slate-700 transition hover:bg-slate-50">取消</button>
              <button type="button" onClick={() => void deleteExperiment(deleteTarget.experiment_id)} className="rounded-lg bg-red-600 px-4 py-2 text-sm font-bold text-white transition hover:bg-red-700">确认删除</button>
            </div>
          </section>
        </div>
      )}
    </div>
  )
}

function Info({ label, value, wide = false }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={`rounded-lg bg-slate-50 px-3 py-2 ${wide ? 'col-span-2' : ''}`}>
      <dt className="text-xs font-bold text-slate-400">{label}</dt>
      <dd className="mt-1 truncate font-semibold text-slate-800">{value}</dd>
    </div>
  )
}

function KeyActionReadiness({ item }: { item: Experiment }) {
  const summary = keyActionSummary(item)
  const ready = keyActionReady(item)
  return (
    <div className={`mt-4 rounded-lg border px-3 py-2 ${ready ? 'border-emerald-200 bg-emerald-50' : 'border-slate-200 bg-slate-50'}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className={`text-xs font-black ${ready ? 'text-emerald-700' : 'text-slate-500'}`}>
          Key Action {ready ? 'Ready' : 'Pending'}
        </span>
        <span className="text-xs font-bold text-slate-500">{summary.status}</span>
      </div>
      <div className="mt-2 grid grid-cols-4 gap-2 text-center text-xs font-bold text-slate-600">
        <span><b className="block text-sm text-slate-950">{summary.segments}</b>segment</span>
        <span><b className="block text-sm text-slate-950">{summary.micros}</b>micro</span>
        <span><b className="block text-sm text-slate-950">{summary.interactions}</b>交互</span>
        <span><b className="block text-sm text-slate-950">{summary.vectors}</b>索引</span>
      </div>
    </div>
  )
}

function EmptyState({ title, description }: { title: string; description?: string }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-600">
      <div className="font-black text-slate-900">{title}</div>
      {description && <div className="mt-1 leading-6">{description}</div>}
    </div>
  )
}
