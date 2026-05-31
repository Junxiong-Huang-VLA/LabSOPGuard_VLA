import { useEffect, useRef, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { Boxes, RefreshCw, ShieldCheck } from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import AnalysisTimingSummary from '../components/AnalysisTimingSummary'
import ExperimentSegments from '../components/ExperimentSegments'
import { EmptyEvidence, EvidenceCard, MetricTile, PageHero, ProgressStrip, primaryButtonClass, secondaryButtonClass } from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import type { AnalysisOverview } from '../types'

const generatingStatuses = new Set(['analyzing', 'running', 'generating_outputs', 'writing_back', 'queued', 'capturing'])
const uploadTimingStoragePrefix = 'realityloop-upload-e2e:'

type ClientUploadTiming = {
  startedAtMs?: number
  completedAtMs?: number
}

function readClientUploadTiming(experimentId?: string): ClientUploadTiming | null {
  if (!experimentId) return null
  try {
    const raw = window.sessionStorage.getItem(`${uploadTimingStoragePrefix}${experimentId}`)
    if (!raw) return null
    const parsed = JSON.parse(raw) as ClientUploadTiming
    return Number.isFinite(parsed.startedAtMs) ? parsed : null
  } catch {
    return null
  }
}

function writeClientUploadTiming(experimentId: string, timing: ClientUploadTiming) {
  try {
    window.sessionStorage.setItem(`${uploadTimingStoragePrefix}${experimentId}`, JSON.stringify(timing))
  } catch {
    // 客户端耗时仅用于展示，不影响后台真值。
  }
}

function toRunStatusLabel(value?: string) {
  const raw = cleanDisplayText(value || '', '').toLowerCase()
  if (raw === 'completed' || raw === 'partial_completed') return 'completed'
  if (raw === 'failed' || raw === 'partial_failed') return 'failed'
  if (raw === 'running' || raw === 'analyzing') return 'running'
  if (raw === 'generating_outputs') return 'generating outputs'
  if (raw === 'writing_back') return 'writing back'
  if (raw === 'queued') return 'queued'
  return raw || 'in progress'
}

function formatDurationSec(value?: number | null) {
  if (!Number.isFinite(Number(value)) || Number(value) <= 0) return '-'
  const seconds = Math.max(0, Number(value))
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remain = Math.floor(seconds % 60)
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(remain).padStart(2, '0')}`
}

function parseCount(value?: unknown) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? Math.max(0, parsed) : 0
}

function safeExperimentId(overview: AnalysisOverview | null) {
  return cleanDisplayText(overview?.experiment?.experiment_id, '')
}

function stageHint(status: string) {
  if (status === 'running') return '当前阶段：摄像头对齐 → 实验切分 → 材料生成。请查看上方进度条与阶段提示，完成后直接进入窗口与关键素材库。'
  if (status === 'completed') return '分析已完成。请先核对窗口，再确认关键素材库中的官方素材与待确认。'
  if (status === 'failed') return '分析异常，请先检查视频来源与权限后再重试。'
  return '当前处于启动阶段，系统正在推进摄像头对齐。'
}

export default function ExperimentWorkspace() {
  const location = useLocation()
  const { id } = useParams<{ id: string }>()
  const [overview, setOverview] = useState<AnalysisOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [clientTiming, setClientTiming] = useState<ClientUploadTiming | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const loadSeqRef = useRef(0)
  const resultRef = useRef<HTMLDivElement | null>(null)

  const isAdvancedMode = () => {
    const query = new URLSearchParams(location.search)
    return query.get('advanced') === '1'
  }

  async function load(force = false) {
    if (!id) return
    const loadSeq = loadSeqRef.current + 1
    loadSeqRef.current = loadSeq
    const isCurrentLoad = () => loadSeqRef.current === loadSeq
    setError(null)
    if (!force) setLoading(true)
    try {
      const nextOverview = await experimentApi.getAnalysisOverview(id, { force })
      if (!isCurrentLoad()) return
      setOverview(nextOverview)
      setLoading(false)
    } catch (exc) {
      if (!isCurrentLoad()) return
      setError(exc instanceof Error ? exc.message : '实验结果读取失败')
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [id])

  useEffect(() => {
    setClientTiming(readClientUploadTiming(id))
  }, [id])

  useEffect(() => {
    if (!overview || !id) return
    const status = toRunStatusLabel(overview.run.status)
    if (!['completed', 'failed', 'partial_completed', 'partial_failed'].includes(status)) return
    if (!resultRef.current) return
    window.setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 180)
  }, [id, overview?.run.status])

  useEffect(() => {
    if (!overview || !clientTiming?.startedAtMs || clientTiming.completedAtMs) return
    if (!['completed', 'partial_completed', 'failed'].includes(toRunStatusLabel(overview.run.status))) return
    const completedTiming = { ...clientTiming, completedAtMs: Date.now() }
    setClientTiming(completedTiming)
    writeClientUploadTiming(id, completedTiming)
  }, [clientTiming, id, overview?.run.status])

  useEffect(() => {
    if (!id) return
    setShowAdvanced(isAdvancedMode())
  }, [location.search, id])

  useEffect(() => {
    if (!overview) return
    const status = toRunStatusLabel(overview.run.status)
    if (!generatingStatuses.has(status)) return
    const timer = window.setInterval(() => void load(true), 2500)
    return () => window.clearInterval(timer)
  }, [overview?.run.status, id])

  if (loading && !overview) return <EmptyEvidence title="加载中" description="正在读取实验进度" />
  if (error) return <EmptyEvidence title="实验加载异常" description={error} />
  if (!overview) return <EmptyEvidence title="实验数据不可用" description="请稍后重试或重新发起分析" />

  const experimentId = safeExperimentId(overview)
  const experimentName = cleanDisplayText(overview.experiment.experiment_name, `实验 ${experimentId}`)
  const runStatus = toRunStatusLabel(overview.run.status)
  const scene = overview.scene_summary || {}
  const detail = overview.scene_summary?.duration_sec || overview.run.timing?.server_end_to_end_sec || overview.run.elapsed_sec || 0
  const windows = parseCount(scene.segment_count)
  const officialMaterials = parseCount(scene.official_segment_count || overview.run.confirmed_step_count)
  const pendingCandidates = parseCount(overview.run.candidate_step_count)
  const durationText = formatDurationSec(detail)
  const clientEndToEnd = Number.isFinite(Number(clientTiming?.completedAtMs) as number)
    && Number.isFinite(Number(clientTiming?.startedAtMs) as number)
      ? Math.max(0, Math.floor(((clientTiming!.completedAtMs as number) - (clientTiming!.startedAtMs as number)) / 1000))
      : 0
  const finalCost = Math.max(clientEndToEnd, Number(detail || 0))

  const progressMessage = [
    '摄像头对齐',
    '实验切分',
    '材料生成',
  ]

  return (
    <div className="space-y-5">
      <PageHero
        eyebrow={<Link to="/experiments" className="hover:text-[color:var(--ui-text)]">实验</Link>}
        title={experimentName}
        description={stageHint(runStatus)}
        actions={
          <>
            {showAdvanced && (
              <button
                type="button"
                onClick={() => {
                  setShowAdvanced(value => !value)
                }}
                className={secondaryButtonClass('warning')}
              >
                <ShieldCheck className="h-4 w-4" />
                退出高级模式
              </button>
            )}
            <button type="button" onClick={() => void load(true)} className={secondaryButtonClass()}>
              <RefreshCw className="h-4 w-4" />
              刷新
            </button>
            <Link to={`/experiments/${experimentId}/materials`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'materials')} className={primaryButtonClass()}>
              <Boxes className="h-4 w-4" />
              关键素材库
            </Link>
          </>
        }
        tabs={<ExperimentTabs experimentId={experimentId} />}
      />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricTile label="窗口" value={windows} tone="primary" helper="实验窗口数量" />
        <MetricTile label="关键素材库" value={officialMaterials} tone="success" helper="可直接进入证据片段" />
        <MetricTile label="待确认" value={pendingCandidates} tone="warning" helper="待确认素材" />
        <MetricTile label="耗时" value={durationText} tone="primary" helper={`全链路 ${String(finalCost > 0 ? formatDurationSec(finalCost) : '-')}`} />
      </section>

      <ProgressStrip
        status={runStatus}
        progress={overview.run.progress}
        message={<span>{progressMessage.join(' → ')} · 当前阶段：{runStatus}{overview.run.message ? ` · ${cleanDisplayText(overview.run.message)}` : ''}</span>}
      />

      <div ref={resultRef} id="result-materials">
        <AnalysisTimingSummary
          run={overview.run}
          clientEndToEndSec={clientTiming && clientTiming.startedAtMs ? Number(finalCost) : undefined}
          statusLabel={cleanDisplayText(overview.run.message, runStatus)}
        />
      </div>

      <EvidenceCard className="px-0">
        <ExperimentSegments experimentId={experimentId} />
      </EvidenceCard>

      {showAdvanced && (
        <section className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/85 p-3 text-sm text-[color:var(--ui-text-muted)] shadow-[var(--ui-shadow-subtle)]">
          <p className="font-medium text-[color:var(--ui-text)]">高级模式</p>
          <p className="mt-1 text-xs">高级信息与诊断在高级模式展示，默认聚焦窗口与材料主线。</p>
        </section>
      )}
    </div>
  )
}

function ExperimentTabs({ experimentId }: { experimentId: string }) {
  const tabClass = 'rounded-md px-3 py-1.5 text-sm font-medium text-[color:var(--ui-text-muted)] transition hover:bg-[color:var(--ui-bg-muted)] hover:text-[color:var(--ui-text)]'
  const activeClass = 'rounded-md bg-[color:var(--ui-accent)] px-3 py-1.5 text-sm font-medium text-white'
  return (
    <nav className="flex flex-wrap gap-2">
      <Link to={`/experiments/${experimentId}/workspace`} className={activeClass}>实验片段</Link>
      <Link to={`/experiments/${experimentId}/materials`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'materials')} className={tabClass}>关键素材库</Link>
      <Link to={`/experiments/${experimentId}/report`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'report')} className={tabClass}>实验室日报</Link>
    </nav>
  )
}
