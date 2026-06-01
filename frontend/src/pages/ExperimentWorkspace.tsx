import { useEffect, useRef, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { Boxes, RefreshCw, ShieldCheck } from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import AnalysisTimingSummary from '../components/AnalysisTimingSummary'
import ExperimentPageShell from '../components/ExperimentSideNav'
import ExperimentSegments from '../components/ExperimentSegments'
import { EmptyEvidence, EvidenceCard, MetricTile, PageHero, ProgressStrip, primaryButtonClass, secondaryButtonClass } from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import {
  DEMO_TOTAL_ELAPSED_SEC,
  getDemoMaterialCount,
  getDemoSegmentCount,
  isDemoExperiment,
} from '../demo/weighingPipettingDemo'
import type { AnalysisOverview } from '../types'

const generatingStatuses = new Set(['analyzing', 'running', 'generating_outputs', 'writing_back', 'queued', 'capturing'])
const uploadTimingStoragePrefix = 'realityloop-upload-e2e:'
const MISSING_TEXT = '未记录'

type ClientUploadTiming = {
  startedAtMs?: number
  completedAtMs?: number
}

function parseClientEndToEndMs(value?: ClientUploadTiming | null) {
  const startedAtMs = Number(value?.startedAtMs)
  const completedAtMs = Number(value?.completedAtMs)
  if (!Number.isFinite(startedAtMs) || !Number.isFinite(completedAtMs)) return null
  const durationMs = completedAtMs - startedAtMs
  if (!Number.isFinite(durationMs) || durationMs < 0) return null
  return Math.max(0, Math.floor(durationMs / 1000))
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
    // ignore best effort storage errors in UI only
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
  const safe = Number(value)
  if (!Number.isFinite(safe) || safe <= 0) return null
  const seconds = Math.max(0, Math.floor(safe))
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remain = seconds % 60
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(remain).padStart(2, '0')}`
}

function parseCountMetric(value?: unknown) {
  if (value == null) return { available: false, value: 0 }
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed < 0) return { available: false, value: 0 }
  return { available: true, value: Math.floor(parsed) }
}

function formatMetricValue(metric: { available: boolean; value: number }) {
  return metric.available ? metric.value : MISSING_TEXT
}

function parseFinite(value: unknown) {
  const candidate = Number(value)
  return Number.isFinite(candidate) ? candidate : null
}

function safeExperimentId(overview: AnalysisOverview | null) {
  return cleanDisplayText(overview?.experiment?.experiment_id, '')
}

function stageHint(status: string) {
  if (status === 'running') return '当前状态：处理中，正在执行实验分析与素材生成。'
  if (status === 'completed') return '实验已完成，显示最终分析结果。'
  if (status === 'failed') return ''
  return '当前状态：准备就绪，等待执行分析任务。'
}

function collectFieldGaps(overview: AnalysisOverview) {
  const gaps: string[] = []
  if (!overview.summary || !parseCountMetric(overview.summary.official_step_count).available) {
    gaps.push('summary.official_step_count')
  }
  if (!overview.summary || !parseCountMetric(overview.summary.confirmed_step_count).available) {
    gaps.push('summary.confirmed_step_count')
  }
  if (!overview.summary || !parseCountMetric(overview.summary.candidate_step_count).available) {
    gaps.push('summary.candidate_step_count')
  }
  return gaps
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
    if (!overview || !id || !clientTiming?.startedAtMs || clientTiming.completedAtMs) return
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

  if (loading && !overview) return <EmptyEvidence title="加载中" description="正在读取实验结果" />
  if (error) return <EmptyEvidence title="实验加载失败" description={error} />
  if (!overview) return <EmptyEvidence title="实验数据不存在" description="当前实验数据不可用，请刷新后重试" />

  const experimentId = safeExperimentId(overview)
  const demoMode = isDemoExperiment(experimentId)
  const experimentName = cleanDisplayText(overview.experiment.experiment_name, `实验 ${experimentId}`)
  const runStatus = toRunStatusLabel(overview.run.status)
  const displayRunStatus = demoMode ? 'completed' : runStatus

  const scene = overview.scene_summary || {}
  const timing = overview.run.timing || {}
  const sceneSegments = parseCountMetric((scene as { segment_count?: unknown }).segment_count)
  const officialSegments = parseCountMetric(overview.summary?.official_step_count)
  const windows = sceneSegments.available ? sceneSegments : officialSegments
  const confirmedMaterials = parseCountMetric(overview.summary?.confirmed_step_count)
  const clientEndToEndSec = parseClientEndToEndMs(clientTiming)
  const serverEndToEndSec = parseFinite(overview.run.elapsed_sec)
    ?? parseFinite(timing.server_end_to_end_sec)
    ?? parseFinite(timing.elapsed_sec)
    ?? parseFinite((timing as { total_sec?: unknown }).total_sec)
  const durationSourceLabel = Number.isFinite(clientEndToEndSec)
    ? '客户端端到端耗时'
    : Number.isFinite(serverEndToEndSec)
      ? '服务端端到端耗时'
      : '字段缺失'
  const displayDurationSec = Number.isFinite(clientEndToEndSec) ? clientEndToEndSec : serverEndToEndSec
  const durationText = formatDurationSec(displayDurationSec) ?? MISSING_TEXT

  const runStatusMessage = cleanDisplayText(overview.run.message || '', '')
  const displayProgress = demoMode || displayRunStatus === 'completed' ? 1 : overview.run.progress
  const displayStatusMessage = demoMode
    ? '分析成功'
    : displayRunStatus === 'failed'
      ? `分析失败${runStatusMessage ? ` · ${runStatusMessage}` : ''}`
      : (
          <span>
            当前阶段：
            {displayRunStatus}
            {overview.run.message ? ` · ${cleanDisplayText(overview.run.message)}` : ''}
          </span>
        )
  const progressMessage = [
    '视频采集',
    '实验分析',
    '素材生成',
  ]
  const fieldGaps = collectFieldGaps(overview)

  return (
    <ExperimentPageShell experimentId={experimentId}>
    <div className="space-y-5">
      <PageHero
        eyebrow={<Link to="/experiments" className="hover:text-[color:var(--ui-text)]">实验列表</Link>}
        title={experimentName}
        description={stageHint(displayRunStatus)}
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
                开发者高级模式
              </button>
            )}
            <button type="button" onClick={() => void load(true)} className={secondaryButtonClass()}>
              <RefreshCw className="h-4 w-4" />
              刷新
            </button>
            <Link to={`/experiments/${experimentId}/materials`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'materials')} className={primaryButtonClass()}>
              <Boxes className="h-4 w-4" />
              关键素材
            </Link>
          </>
        }
      />

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        <MetricTile
          label="实验片段"
          value={demoMode ? String(getDemoSegmentCount()) : formatMetricValue(windows)}
          tone="primary"
          helper="实验片段数量"
        />
        <MetricTile
          label="关键素材"
          value={demoMode ? String(getDemoMaterialCount()) : formatMetricValue(confirmedMaterials)}
          tone="success"
          helper="当前可产出关键素材数量"
        />
        <MetricTile
          label="耗时"
          value={demoMode ? `${DEMO_TOTAL_ELAPSED_SEC} 秒` : durationText}
          tone="primary"
          helper={demoMode ? '实验总耗时' : `实验总耗时（${durationSourceLabel}）`}
        />
      </section>

      <ProgressStrip
        status={displayRunStatus}
        progress={displayProgress}
        variant="health"
        message={(
          <span>
            {progressMessage.join(' · ')}
            {' · '}
            {displayStatusMessage}
          </span>
        )}
      />

      {fieldGaps.length > 0 ? (
        <section className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/85 p-3 text-xs text-[color:var(--ui-text-muted)] shadow-[var(--ui-shadow-subtle)]">
          <p>当前返回字段存在缺失：{fieldGaps.join(', ')}</p>
        </section>
      ) : null}

      <div ref={resultRef} id="result-materials">
        <AnalysisTimingSummary
          run={overview.run}
          clientEndToEndSec={clientEndToEndSec ?? undefined}
          demo={demoMode}
        />
      </div>

      <EvidenceCard className="px-0">
        <ExperimentSegments experimentId={experimentId} />
      </EvidenceCard>

      {showAdvanced && (
        <details
          className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/85 p-3 text-sm text-[color:var(--ui-text-muted)] shadow-[var(--ui-shadow-subtle)]"
          open
        >
          <summary className="cursor-pointer font-medium text-[color:var(--ui-text)]">开发者高级信息</summary>
          <p className="mt-2 text-xs">run_id: {overview.run.run_id}</p>
          <p className="mt-1 text-xs">run.status: {overview.run.status}</p>
          <p className="mt-1 text-xs">run.message: {cleanDisplayText(overview.run.message || '无')}</p>
        </details>
      )}
    </div>
    </ExperimentPageShell>
  )
}
