import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { FolderOpen, Plus, Search, PlayCircle } from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import { cleanDisplayText } from '../displayText'
import type { Experiment } from '../types'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, primaryButtonClass, secondaryButtonClass } from '../components/EvidenceUI'

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

function experimentStatusText(item: Experiment) {
  if (item.archived_at) return '归档'
  if (item.processing_error) return '失败'
  if (item.status === 'running' || item.status === 'queued') return '分析中'
  if (item.status === 'completed' || item.status === 'analyzed') return '已完成'
  return cleanDisplayText(item.status, '已创建')
}

function statClassFromError(error?: string | null) {
  return error ? 'text-red-700' : 'text-slate-700'
}

export default function ExperimentList() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [keyword, setKeyword] = useState('')

  async function load() {
    setLoading(true)
    setError(null)
    try {
      const first = await experimentApi.list({ limit: 200 }, { force: true })
      const loaded: Experiment[] = [...(first.experiments || [])]
      let offset = loaded.length
      while (offset < Number(first.total || loaded.length)) {
        const next = await experimentApi.list({ limit: 200, offset }, { force: true })
        const nextExperiments = next.experiments || []
        if (!nextExperiments.length) break
        loaded.push(...nextExperiments)
        offset += nextExperiments.length
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

  const filtered = useMemo(() => {
    const term = keyword.trim().toLowerCase()
    if (!term) return experiments
    return experiments.filter(item => {
      const display = cleanDisplayText(item.title, item.experiment_id)
      return display.toLowerCase().includes(term) || String(item.experiment_id).toLowerCase().includes(term)
    })
  }, [experiments, keyword])

  return (
    <div className="space-y-6">
      <PageHero
        title="实验"
        description="产品主流程：新建实验 → 选择视频 → 开始分析。先看窗口，再确认关键素材库。"
        actions={
          <>
            <Link to="/upload" className={primaryButtonClass()}>
              <Plus className="h-4 w-4" />
              新建实验
            </Link>
            <Link to="/upload" className={secondaryButtonClass()}>
              <FolderOpen className="h-4 w-4" />
              选择视频
            </Link>
            <Link to="/upload" className={secondaryButtonClass()}>
              <PlayCircle className="h-4 w-4" />
              开始分析
            </Link>
          </>
        }
      />

      <section className="grid gap-3 sm:grid-cols-3">
        <MetricTile label="总实验" value={experiments.length} tone="primary" helper="实验列表规模" />
        <MetricTile label="已完成" value={experiments.filter(item => String(item.status).toLowerCase() === 'completed').length} tone="success" helper="可直接查看窗口与素材" />
        <MetricTile label="分析中" value={experiments.filter(item => ['running', 'queued'].includes(String(item.status || '').toLowerCase())).length} tone="warning" helper="持续计算中的实验" />
      </section>

      <div className="rounded-xl border border-slate-200 bg-white p-4">
        <div className="relative max-w-md">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            value={keyword}
            onChange={(event) => setKeyword(event.target.value)}
            placeholder="搜索实验名称或 ID"
            className="h-10 w-full rounded-lg border border-slate-200 bg-white px-9 py-2 text-sm"
          />
        </div>
      </div>

      {loading && <EmptyEvidence title="加载中" description="正在读取实验列表，稍后会同步展示可执行实验。" />}
      {error && <EmptyEvidence title="列表加载失败" description={error} />}

      {!loading && !error && filtered.length === 0 && (
        <EmptyEvidence
          title="暂无实验"
          description="先点击“新建实验 / 选择视频”，上传双视角素材后点击“开始分析”。"
        />
      )}

      <section className="grid gap-4 xl:grid-cols-2">
        {filtered.map(item => {
          const status = experimentStatusText(item)
          const windowCount = Number((item as { segment_count?: number }).segment_count || 0)
          const materialCount = Number((item as { published_material_count?: number }).published_material_count || (item as { material_count?: number }).material_count || 0)
          const avgConfidence = Number(item.avg_confidence || item.confidence || 0)
          const statusTone: 'danger' | 'success' | 'primary' | 'slate' = status === '失败' ? 'danger' : status === '已完成' ? 'success' : status === '分析中' || status === 'queued' ? 'warning' : 'slate'
          const workflowTarget = `/experiments/${item.experiment_id}/workspace`
          return (
            <EvidenceCard key={item.experiment_id} className="p-5">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-base font-black text-slate-950">{cleanDisplayText(item.title, item.experiment_id)}</h3>
                    <EvidenceBadge tone={statusTone}>{status}</EvidenceBadge>
                  </div>
                  <p className="mt-2 text-xs text-slate-500">{item.experiment_id}</p>
                  <p className="mt-1 text-xs text-slate-500">创建时间 {formatDate(item.created_at)}</p>
                </div>
                <p className={`text-xs font-black ${statClassFromError(item.processing_error)}`}>{item.processing_error ? '有异常' : '正常'}</p>
              </div>

              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <div className="rounded-lg bg-slate-50 p-2 text-sm">
                  <div className="text-xs text-slate-500">窗口</div>
                  <div className="font-semibold text-slate-800">{windowCount}</div>
                </div>
                <div className="rounded-lg bg-slate-50 p-2 text-sm">
                  <div className="text-xs text-slate-500">关键素材</div>
                  <div className="font-semibold text-slate-800">{materialCount}</div>
                </div>
                <div className="rounded-lg bg-slate-50 p-2 text-sm">
                  <div className="text-xs text-slate-500">置信度</div>
                  <div className="font-semibold text-slate-800">{avgConfidence > 0 ? avgConfidence.toFixed(3) : '-'}</div>
                </div>
              </div>

              <div className="mt-4 flex items-center justify-end gap-2">
                <Link
                  to={workflowTarget}
                  onMouseEnter={() => prefetchExperimentRoute(item.experiment_id, 'workspace')}
                  className={primaryButtonClass()}
                >
                  <PlayCircle className="h-4 w-4" />
                  开始分析
                </Link>
              </div>
            </EvidenceCard>
          )
        })}
      </section>
    </div>
  )
}
