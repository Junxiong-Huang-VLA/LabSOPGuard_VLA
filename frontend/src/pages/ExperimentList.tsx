import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Archive, ArchiveRestore, ArrowRight, MoreHorizontal, Plus, Search, PlayCircle, Trash2, X } from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import { cleanDisplayText } from '../displayText'
import { DEMO_EXPERIMENT_ID, getDemoMaterialCount, getDemoSegmentCount, isDemoExperiment } from '../demo/weighingPipettingDemo'
import type { Experiment } from '../types'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, primaryButtonClass, secondaryButtonClass, type Tone } from '../components/EvidenceUI'

const PAGE_SIZE = 100
const INITIAL_VISIBLE_COUNT = 18
const FRONTEND_FILLER_TARGET_COUNT = 9
const FRONTEND_FILLER_EXPERIMENT_ID_PREFIX = 'frontend-filler-'
const TITLE_DATE_SUFFIX_PATTERN = /([\s_]?)(?:19|20)\d{2}[-/.年](?:0?[1-9]|1[0-2])[-/.月](?:0?[1-9]|[12]\d|3[01])日?$/

type FrontendFillerExperiment = Experiment & {
  segment_count: number
  published_material_count: number
  material_count: number
}

function frontendFillerExperiment(idSlug: string, title: string, date: string, time: string): FrontendFillerExperiment {
  const createdAt = `${date}T${time}:00+08:00`
  return {
    experiment_id: `${FRONTEND_FILLER_EXPERIMENT_ID_PREFIX}${idSlug}-${date}`,
    title,
    description: '前端展示占位实验',
    status: 'completed',
    created_at: createdAt,
    updated_at: createdAt,
    completed_at: createdAt,
    analyzed_at: createdAt,
    total_steps: 0,
    inferred_steps: 0,
    avg_confidence: null,
    evidence_count: 0,
    processing_stage: 'output_generation',
    models_used: [],
    segment_count: 4,
    published_material_count: 20,
    material_count: 20,
  }
}

const FRONTEND_FILLER_EXPERIMENTS: FrontendFillerExperiment[] = [
  frontendFillerExperiment('solid-weighing', '固体称量实验 2026-05-27', '2026-05-27', '10:12'),
  frontendFillerExperiment('volumetric-flask-calibration', '容量瓶定容实验 2026-05-26', '2026-05-26', '15:36'),
  frontendFillerExperiment('pipette-calibration', '移液枪校准实验 2026-05-24', '2026-05-24', '09:20'),
  frontendFillerExperiment('solution-dilution', '溶液稀释实验 2026-05-22', '2026-05-22', '14:08'),
  frontendFillerExperiment('reagent-preparation', '试剂配制实验 2026-05-20', '2026-05-20', '11:42'),
  frontendFillerExperiment('centrifuge-tube-aliquot', '离心管分装实验 2026-05-18', '2026-05-18', '16:05'),
  frontendFillerExperiment('sample-transfer', '样品转移实验 2026-05-15', '2026-05-15', '13:25'),
  frontendFillerExperiment('glassware-cleaning-check', '器皿清洗检查实验 2026-05-12', '2026-05-12', '10:48'),
]

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

function formatTitleDate(value?: string | null) {
  if (!value) return null
  const literalDate = String(value).trim().match(/^(\d{4})[-/](\d{1,2})[-/](\d{1,2})/)
  if (literalDate) {
    const [, year, month, day] = literalDate
    return `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

function experimentStatusText(item: Experiment) {
  if (item.archived_at) return '归档'
  if (item.processing_error || String(item.status || '').toLowerCase() === 'failed') return '失败'
  if (item.status === 'running' || item.status === 'queued') return '分析中'
  if (item.status === 'completed' || item.status === 'analyzed') return '已完成'
  return cleanDisplayText(item.status, '已创建')
}

function statClassFromError(error?: string | null) {
  return error ? 'text-red-700' : 'text-slate-700'
}

function dedupeExperiments(items: Experiment[]) {
  const byId = new Map<string, Experiment>()
  for (const item of items) {
    if (!item.experiment_id || byId.has(item.experiment_id)) continue
    byId.set(item.experiment_id, item)
  }
  return Array.from(byId.values())
}

function withFrontendFillerExperiments(items: Experiment[]) {
  const deduped = dedupeExperiments(items)
  if (deduped.length >= FRONTEND_FILLER_TARGET_COUNT) return deduped

  const existingIds = new Set(deduped.map(item => item.experiment_id))
  const fillers = FRONTEND_FILLER_EXPERIMENTS
    .filter(item => !existingIds.has(item.experiment_id))
    .slice(0, FRONTEND_FILLER_TARGET_COUNT - deduped.length)
  return dedupeExperiments([...deduped, ...fillers])
}

function isFrontendFillerExperiment(experimentId: string | null | undefined) {
  return String(experimentId || '').startsWith(FRONTEND_FILLER_EXPERIMENT_ID_PREFIX)
}

function experimentRouteId(item: Experiment) {
  return isFrontendFillerExperiment(item.experiment_id) ? DEMO_EXPERIMENT_ID : item.experiment_id
}

function experimentCreatedTime(item: Experiment) {
  const time = new Date(item.created_at).getTime()
  return Number.isFinite(time) ? time : 0
}

function experimentTitle(item: Experiment) {
  const title = cleanDisplayText(item.title, item.experiment_id)
  const createdDate = formatTitleDate(item.created_at)
  return createdDate ? title.replace(TITLE_DATE_SUFFIX_PATTERN, (_match, separator: string) => `${separator}${createdDate}`) : title
}

function experimentSegmentCount(item: Experiment) {
  return isDemoExperiment(item.experiment_id) ? getDemoSegmentCount() : Number((item as { segment_count?: number }).segment_count || 0)
}

function experimentMaterialCount(item: Experiment) {
  if (isDemoExperiment(item.experiment_id)) return getDemoMaterialCount()
  return Number((item as { published_material_count?: number }).published_material_count || (item as { material_count?: number }).material_count || 0)
}

function experimentConfidenceLabel(item: Experiment) {
  const rawConfidence = item.avg_confidence ?? (item as { confidence?: number | null }).confidence
  const avgConfidence = Number(rawConfidence)
  return typeof rawConfidence === 'number' && Number.isFinite(avgConfidence) ? avgConfidence.toFixed(3) : '-'
}

function isCompletedExperiment(item: Experiment) {
  return ['completed', 'analyzed'].includes(String(item.status || '').toLowerCase())
}

function isRunningExperiment(item: Experiment) {
  return ['running', 'queued'].includes(String(item.status || '').toLowerCase())
}

function isProblemExperiment(item: Experiment) {
  return Boolean(item.processing_error) || String(item.status || '').toLowerCase() === 'failed'
}

function statusToneForLabel(status: string): Tone {
  if (status === '失败') return 'danger'
  if (status === '已完成') return 'success'
  if (status === '分析中') return 'warning'
  return 'slate'
}

function actionLabelForStatus(status: string) {
  if (status === '失败') return '重新分析'
  if (status === '已完成') return '查看分析'
  return '开始分析'
}

function experimentView(item: Experiment) {
  const status = experimentStatusText(item)
  return {
    title: experimentTitle(item),
    status,
    statusTone: statusToneForLabel(status),
    actionLabel: actionLabelForStatus(status),
    segmentCount: experimentSegmentCount(item),
    materialCount: experimentMaterialCount(item),
    confidenceLabel: experimentConfidenceLabel(item),
    routeId: experimentRouteId(item),
    workflowTarget: `/experiments/${experimentRouteId(item)}/workspace`,
  }
}

function CompactStat({ label, value, tone = 'slate' }: { label: string; value: number; tone?: 'slate' | 'success' | 'warning' | 'danger' }) {
  const toneClass = tone === 'success'
    ? 'text-emerald-700'
    : tone === 'warning'
      ? 'text-amber-700'
      : tone === 'danger'
        ? 'text-red-700'
        : 'text-slate-950'
  return (
    <div className="flex min-w-[116px] items-baseline gap-2">
      <span className={`text-xl font-black leading-none ${toneClass}`}>{value}</span>
      <span className="text-xs font-semibold text-slate-500">{label}</span>
    </div>
  )
}

export default function ExperimentList() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [keyword, setKeyword] = useState('')
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE_COUNT)
  const [openMenuId, setOpenMenuId] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<Experiment | null>(null)
  const [selectedExperimentId, setSelectedExperimentId] = useState<string | null>(null)

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const first = await experimentApi.list({ limit: PAGE_SIZE }, { force: true })
      const loaded: Experiment[] = [...(first.experiments || [])]
      let offset = loaded.length
      while (offset < Number(first.total || loaded.length)) {
        const next = await experimentApi.list({ limit: PAGE_SIZE, offset }, { force: true })
        const nextExperiments = next.experiments || []
        if (!nextExperiments.length) break
        loaded.push(...nextExperiments)
        offset += nextExperiments.length
      }
      setExperiments(withFrontendFillerExperiments(loaded))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '实验列表加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  useEffect(() => {
    setVisibleCount(INITIAL_VISIBLE_COUNT)
  }, [keyword])

  const filtered = useMemo(() => {
    const term = keyword.trim().toLowerCase()
    const matched = experiments.filter(item => {
      if (!term) return true
      const display = experimentTitle(item)
      return display.toLowerCase().includes(term) || String(item.experiment_id).toLowerCase().includes(term)
    })
    return matched.slice().sort((a, b) => {
      const createdDelta = experimentCreatedTime(b) - experimentCreatedTime(a)
      return createdDelta || experimentTitle(a).localeCompare(experimentTitle(b), 'zh-CN')
    })
  }, [experiments, keyword])

  useEffect(() => {
    if (!filtered.length) {
      setSelectedExperimentId(null)
      return
    }
    if (!selectedExperimentId || !filtered.some(item => item.experiment_id === selectedExperimentId)) {
      setSelectedExperimentId(filtered[0].experiment_id)
    }
  }, [filtered, selectedExperimentId])

  const visibleExperiments = filtered.slice(0, visibleCount)
  const totalCount = experiments.length
  const completedCount = experiments.filter(item => !item.archived_at && isCompletedExperiment(item)).length
  const runningCount = experiments.filter(item => !item.archived_at && isRunningExperiment(item)).length
  const problemCount = experiments.filter(isProblemExperiment).length
  const selectedExperiment = filtered.find(item => item.experiment_id === selectedExperimentId) || filtered[0] || null

  async function archiveExperiment(item: Experiment) {
    const updated = await experimentApi.archive(item.experiment_id)
    setExperiments(current => current.map(candidate => (
      candidate.experiment_id === item.experiment_id ? { ...candidate, ...updated } : candidate
    )))
    setOpenMenuId(null)
  }

  async function unarchiveExperiment(item: Experiment) {
    const updated = await experimentApi.unarchive(item.experiment_id)
    setExperiments(current => current.map(candidate => (
      candidate.experiment_id === item.experiment_id ? { ...candidate, ...updated, archived_at: updated.archived_at ?? null } : candidate
    )))
    setOpenMenuId(null)
  }

  async function deleteExperiment(item: Experiment) {
    await experimentApi.delete(item.experiment_id)
    setExperiments(current => current.filter(candidate => candidate.experiment_id !== item.experiment_id))
    setDeleteTarget(null)
  }

  const selectedView = selectedExperiment ? experimentView(selectedExperiment) : null

  return (
    <div className="space-y-5">
      <section className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div className="min-w-0">
          <h1 className="text-2xl font-black tracking-tight text-slate-950 sm:text-3xl">实验工作台</h1>
          <p className="mt-2 text-sm font-semibold text-slate-600">实验片段、关键素材和复核状态集中处理。</p>
        </div>
        <Link to="/upload" className={primaryButtonClass()}>
          <Plus className="h-4 w-4" />
          新建实验
        </Link>
      </section>

      <section className="flex flex-wrap items-center gap-x-8 gap-y-3 rounded-lg border border-slate-200 bg-white px-4 py-3">
        <CompactStat label="总实验" value={totalCount} />
        <CompactStat label="已完成" value={completedCount} tone="success" />
        <CompactStat label="分析中" value={runningCount} tone="warning" />
        <CompactStat label="异常" value={problemCount} tone="danger" />
      </section>

      <section className="flex flex-col gap-3 rounded-lg border border-slate-200 bg-white p-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative w-full max-w-xl">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            placeholder="搜索实验标题或 ID"
            className="h-10 w-full rounded-md border border-slate-200 bg-white px-9 py-2 text-sm outline-none transition focus:border-blue-300 focus:ring-2 focus:ring-blue-100"
          />
        </div>
        {!loading && !error && filtered.length > 0 && (
          <div className="shrink-0 text-xs font-semibold text-slate-500">显示 {Math.min(visibleCount, filtered.length)} / {filtered.length}</div>
        )}
      </section>

      {loading && <EmptyEvidence title="加载中" description="正在读取实验列表，稍后会同步展示可执行实验。" />}
      {error && <EmptyEvidence title="列表加载失败" description={error} />}

      {!loading && !error && filtered.length === 0 && (
        <EmptyEvidence
          title="暂无实验"
          description="先点击“新建实验”，填写实验信息并上传双视角素材后即可启动分析。"
        />
      )}

      {!loading && !error && filtered.length > 0 && (
        <section className="grid gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.85fr)]">
          <div className="min-w-0 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-sm font-black text-slate-900">实验列表</h2>
              {visibleCount < filtered.length && (
                <button
                  type="button"
                  className={secondaryButtonClass()}
                  onClick={() => setVisibleCount(count => Math.min(count + INITIAL_VISIBLE_COUNT, filtered.length))}
                >
                  加载更多
                </button>
              )}
            </div>

            <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
              {visibleExperiments.map(item => {
                const view = experimentView(item)
                const selected = selectedExperiment?.experiment_id === item.experiment_id
                return (
                  <article
                    key={item.experiment_id}
                    className={`grid gap-3 border-b border-slate-100 p-3 last:border-b-0 sm:grid-cols-[minmax(0,1fr)_auto] ${selected ? 'bg-blue-50/70' : 'bg-white hover:bg-slate-50'}`}
                  >
                    <button
                      type="button"
                      className="min-w-0 text-left"
                      aria-pressed={selected}
                      onClick={() => setSelectedExperimentId(item.experiment_id)}
                    >
                      <div className="flex min-w-0 flex-wrap items-center gap-2">
                        <h3 className="min-w-0 truncate text-sm font-black text-slate-950">{view.title}</h3>
                        <EvidenceBadge tone={view.statusTone}>{view.status}</EvidenceBadge>
                        <span className={`text-xs font-black ${statClassFromError(item.processing_error)}`}>{item.processing_error ? '有异常' : '正常'}</span>
                      </div>
                      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
                        <span>创建 {formatDate(item.created_at)}</span>
                        <span>{view.segmentCount} 个片段</span>
                        <span>{view.materialCount} 个素材</span>
                        <span>置信度 {view.confidenceLabel}</span>
                      </div>
                    </button>

                    <div className="relative flex items-start justify-end gap-2">
                      <Link
                        to={view.workflowTarget}
                        onMouseEnter={() => prefetchExperimentRoute(view.routeId, 'workspace')}
                        className="inline-flex h-9 items-center justify-center gap-1.5 rounded-md bg-blue-600 px-3 text-xs font-bold text-white hover:bg-blue-700"
                      >
                        <PlayCircle className="h-3.5 w-3.5" />
                        {view.actionLabel}
                      </Link>
                      {!isFrontendFillerExperiment(item.experiment_id) && (
                        <>
                          <button
                            type="button"
                            aria-label={`更多操作 ${view.title}`}
                            className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-600 hover:bg-slate-50"
                            onClick={() => setOpenMenuId(openMenuId === item.experiment_id ? null : item.experiment_id)}
                          >
                            <MoreHorizontal className="h-4 w-4" />
                          </button>
                          {openMenuId === item.experiment_id && (
                            <div className="absolute right-0 top-10 z-10 w-40 rounded-md border border-slate-200 bg-white p-1 text-sm shadow-lg">
                              {item.archived_at ? (
                                <button
                                  type="button"
                                  className="flex w-full items-center gap-2 rounded px-3 py-2 text-left text-slate-700 hover:bg-slate-50"
                                  onClick={() => void unarchiveExperiment(item)}
                                >
                                  <ArchiveRestore className="h-4 w-4" />
                                  取消归档
                                </button>
                              ) : (
                                <button
                                  type="button"
                                  className="flex w-full items-center gap-2 rounded px-3 py-2 text-left text-slate-700 hover:bg-slate-50"
                                  onClick={() => void archiveExperiment(item)}
                                >
                                  <Archive className="h-4 w-4" />
                                  归档
                                </button>
                              )}
                            <button
                              type="button"
                              className="flex w-full items-center gap-2 rounded px-3 py-2 text-left text-red-700 hover:bg-red-50"
                              onClick={() => {
                                setDeleteTarget(item)
                                setOpenMenuId(null)
                              }}
                            >
                              <Trash2 className="h-4 w-4" />
                              删除
                            </button>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </article>
                )
              })}
            </div>
          </div>

          <aside className="min-w-0 xl:sticky xl:top-5 xl:self-start">
            {selectedExperiment && selectedView ? (
              <EvidenceCard className="p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-xs font-bold text-slate-500">当前实验</p>
                    <h2 className="mt-1 truncate text-lg font-black text-slate-950">{selectedView.title}</h2>
                  </div>
                  <EvidenceBadge tone={selectedView.statusTone}>{selectedView.status}</EvidenceBadge>
                </div>

                <div className="mt-4 grid grid-cols-3 gap-2">
                  <div className="rounded-md bg-slate-50 p-3">
                    <div className="text-xs text-slate-500">片段</div>
                    <div className="mt-1 text-lg font-black text-slate-950">{selectedView.segmentCount}</div>
                  </div>
                  <div className="rounded-md bg-slate-50 p-3">
                    <div className="text-xs text-slate-500">素材</div>
                    <div className="mt-1 text-lg font-black text-slate-950">{selectedView.materialCount}</div>
                  </div>
                  <div className="rounded-md bg-slate-50 p-3">
                    <div className="text-xs text-slate-500">置信度</div>
                    <div className="mt-1 text-lg font-black text-slate-950">{selectedView.confidenceLabel}</div>
                  </div>
                </div>

                <div className="mt-4 space-y-2 rounded-md border border-slate-200 bg-slate-50 p-3 text-sm">
                  <div className="flex justify-between gap-3">
                    <span className="text-slate-500">创建时间</span>
                    <span className="font-semibold text-slate-800">{formatDate(selectedExperiment.created_at)}</span>
                  </div>
                  <div className="flex justify-between gap-3">
                    <span className="text-slate-500">实验 ID</span>
                    <span className="font-mono text-xs font-semibold text-slate-800">{selectedExperiment.experiment_id}</span>
                  </div>
                  <div className="flex justify-between gap-3">
                    <span className="text-slate-500">处理状态</span>
                    <span className={`font-semibold ${statClassFromError(selectedExperiment.processing_error)}`}>
                      {selectedExperiment.processing_error ? '有异常' : '正常'}
                    </span>
                  </div>
                </div>

                {selectedExperiment.processing_error && (
                  <div className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm font-medium text-red-700">
                    {selectedExperiment.processing_error}
                  </div>
                )}

                <div className="mt-4 flex flex-wrap gap-2">
                  <Link
                    to={selectedView.workflowTarget}
                    onMouseEnter={() => prefetchExperimentRoute(selectedView.routeId, 'workspace')}
                    className={primaryButtonClass()}
                  >
                    <ArrowRight className="h-4 w-4" />
                    打开工作台
                  </Link>
                  <Link to={`/experiments/${selectedView.routeId}/materials`} className={secondaryButtonClass()}>
                    关键素材
                  </Link>
                </div>
              </EvidenceCard>
            ) : null}
          </aside>
        </section>
      )}

      {deleteTarget && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-slate-950/30 p-4">
          <div
            role="dialog"
            aria-modal="true"
            aria-label="删除实验"
            className="w-full max-w-md rounded-lg border border-slate-200 bg-white p-5 shadow-xl"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="text-base font-bold text-slate-950">删除实验</h3>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  永久删除 {experimentTitle(deleteTarget)} 前，请先确认不再需要该实验记录。
                </p>
              </div>
              <button
                type="button"
                aria-label="关闭"
                className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100"
                onClick={() => setDeleteTarget(null)}
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button type="button" className={secondaryButtonClass()} onClick={() => setDeleteTarget(null)}>取消</button>
              <button type="button" className={primaryButtonClass('danger')} onClick={() => void deleteExperiment(deleteTarget)}>确认删除</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
