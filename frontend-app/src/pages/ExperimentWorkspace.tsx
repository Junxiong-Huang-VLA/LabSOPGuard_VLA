import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { AlertTriangle, BadgeCheck, Boxes, Clock3, FileText, Layers3, LocateFixed, PackageCheck, PlayCircle, RefreshCw, Video } from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import DualVideoPlayer from '../components/DualVideoPlayer'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, ProgressStrip, primaryButtonClass, secondaryButtonClass, toneForStatus } from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import { mediaUrl } from '../mediaUrl'
import type { AnalysisOverview, KeyActionResults, MaterialSearchItem, StepRecord } from '../types'

const generatingStatuses = new Set(['analyzing', 'running', 'generating_outputs', 'writing_back', 'queued', 'capturing'])

function formatNumber(value: unknown, digits = 2) {
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue.toFixed(digits) : '-'
}

function formatRange(start?: number | null, end?: number | null) {
  return `${formatNumber(start, 2)}-${formatNumber(end, 2)}s`
}

function keyed(value: unknown, index: number, prefix: string) {
  const raw = String(value ?? '').trim()
  return `${prefix}-${raw || 'item'}-${index}`
}

function materialKey(item: MaterialSearchItem, index: number) {
  return keyed(item.event_id || item.item_id || item.display_name || item.event_type, index, 'material')
}

function visibleSteps(overview: AnalysisOverview) {
  if (overview.steps.official.length) return overview.steps.official
  if (overview.steps.candidate.length) return overview.steps.candidate
  return overview.steps.inferred
}

function artifactUrl(overview: AnalysisOverview, keys: string[]) {
  for (const key of keys) {
    const artifact = overview.artifacts[key]
    if (artifact?.ready && artifact.url) return mediaUrl(artifact.url)
  }
  return undefined
}

export default function ExperimentWorkspace() {
  const { id } = useParams<{ id: string }>()
  const [overview, setOverview] = useState<AnalysisOverview | null>(null)
  const [materials, setMaterials] = useState<MaterialSearchItem[]>([])
  const [keyResults, setKeyResults] = useState<KeyActionResults | null>(null)
  const [selectedStep, setSelectedStep] = useState<StepRecord | null>(null)
  const [seekRequest, setSeekRequest] = useState<{ seekTo: number; token: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function load(force = false) {
    if (!id) return
    setError(null)
    if (!force) setLoading(true)
    try {
      const nextOverview = await experimentApi.getAnalysisOverview(id, { force })
      setOverview(nextOverview)
      const steps = visibleSteps(nextOverview)
      setSelectedStep(previous => steps.find(step => step.step_id === previous?.step_id) || steps[0] || null)
      const nextMaterials = await experimentApi.getPublishedMaterials(id, { limit: 6 }, { force })
      setMaterials((nextMaterials.items || []) as MaterialSearchItem[])
      const getKeyActionStatus = experimentApi.getKeyActionStatus as unknown as ((experimentId: string) => Promise<{ status?: string } | null>) | undefined
      const getKeyActionResults = experimentApi.getKeyActionResults as unknown as ((experimentId: string) => Promise<KeyActionResults | null>) | undefined
      const nextKeyStatus = getKeyActionStatus ? await getKeyActionStatus(id).catch(() => null) : null
      if (nextKeyStatus?.status === 'completed' && getKeyActionResults) {
        setKeyResults(await getKeyActionResults(id).catch(() => null))
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'analysis-overview 加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [id])

  useEffect(() => {
    if (!overview || !generatingStatuses.has(String(overview.run.status))) return
    const timer = window.setInterval(() => void load(true), 2000)
    return () => window.clearInterval(timer)
  }, [overview?.run.status, id])

  const steps = useMemo(() => (overview ? visibleSteps(overview) : []), [overview])
  const sourceUrl = overview ? artifactUrl(overview, ['source_video', 'first_person_video', 'video']) : undefined
  const annotatedUrl = overview ? artifactUrl(overview, ['annotated_video', 'third_person_video']) : undefined
  const showGeneratingSteps = Boolean(overview && !overview.readiness.steps_ready && generatingStatuses.has(String(overview.run.status)))
  const noStructuredSteps = Boolean(overview && overview.run.status === 'completed' && steps.length === 0)

  function jumpTo(seconds?: number | null) {
    if (seconds == null) return
    setSeekRequest({ seekTo: seconds, token: Date.now() })
  }

  if (loading && !overview) return <EmptyEvidence title="正在加载工作台..." />
  if (error) return <EvidenceCard className="border-red-200 bg-red-50 p-5 text-red-700">{error}</EvidenceCard>
  if (!overview) return <EmptyEvidence title="Failed to load analysis overview" />

  const totalAlerts = overview.alerts.length
  const totalInteractionEvents = keyResults?.interaction_events?.length || 0
  const experimentName = cleanDisplayText(overview.experiment.experiment_name, `Experiment ${overview.experiment.experiment_id}`)

  return (
    <div className="space-y-5">
      <PageHero
        eyebrow={<Link to="/experiments" className="hover:text-slate-900">实验队列</Link>}
        title={experimentName}
        description="围绕双视角视频、结构化步骤、异常告警、关键素材与关键动作索引的实验复盘工作台。"
        actions={(
          <>
            <button type="button" onClick={() => void load(true)} className={secondaryButtonClass()}>
              <RefreshCw className="h-4 w-4" />
              刷新
            </button>
            <Link to={`/experiments/${overview.experiment.experiment_id}/key-actions`} className={primaryButtonClass('blue')}>
              <Layers3 className="h-4 w-4" />
              关键动作
            </Link>
          </>
        )}
        tabs={<ExperimentTabs experimentId={overview.experiment.experiment_id} />}
      />

      <ProgressStrip status={overview.run.status} progress={overview.run.progress} message={`${overview.run.stage}${overview.run.message ? ` · ${overview.run.message}` : ''}`} />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <MetricTile label="结构步骤" value={steps.length} helper={`${overview.summary.official_step_count} official`} tone="blue" Icon={BadgeCheck} />
        <MetricTile label="检测框" value={overview.summary.detection_count} helper={overview.summary.model_name} tone="cyan" Icon={Video} />
        <MetricTile label="告警" value={totalAlerts} helper="rule events" tone={totalAlerts ? 'red' : 'slate'} Icon={AlertTriangle} />
        <MetricTile label="交互证据" value={totalInteractionEvents} helper={`${keyResults?.micro_segments?.length || 0} micro`} tone="emerald" Icon={Layers3} />
        <MetricTile label="关键素材" value={materials.length} helper="published previews" tone="violet" Icon={PackageCheck} />
      </section>

      <div className="grid gap-5 xl:grid-cols-[18rem_minmax(0,1fr)_22rem]">
        <aside className="space-y-3">
          <EvidenceCard className="p-3">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="font-black text-slate-950">步骤轨道</h3>
              <EvidenceBadge tone={toneForStatus(overview.run.status)}>{steps.length}</EvidenceBadge>
            </div>
            <div className="max-h-[68vh] space-y-2 overflow-auto pr-1">
              {steps.map(step => (
                <button
                  type="button"
                  key={step.step_id}
                  onClick={() => { setSelectedStep(step); jumpTo(step.start_time_sec) }}
                  className={`w-full rounded-lg border p-3 text-left transition ${selectedStep?.step_id === step.step_id ? 'border-blue-300 bg-blue-50' : 'border-slate-200 bg-white hover:bg-slate-50'}`}
                >
                  <div className="flex items-center justify-between text-xs font-bold text-slate-500">
                    <span>Step {step.step_index}</span>
                    <span>{formatRange(step.start_time_sec, step.end_time_sec)}</span>
                  </div>
                  <div className="mt-1 line-clamp-2 text-sm font-black text-slate-950">{cleanDisplayText(step.step_name, `Step ${step.step_index}`)}</div>
                  <EvidenceBadge className="mt-2" tone={toneForStatus(step.status)}>{step.status}</EvidenceBadge>
                </button>
              ))}
              {showGeneratingSteps && (
                <>
                  <span className="sr-only">Generating structured steps...</span>
                  <EmptyEvidence title="正在生成步骤" description="摘要已经可用，结构化步骤仍在写入。" />
                </>
              )}
              {noStructuredSteps && <EmptyEvidence title="未生成结构化步骤" description="本次完成的分析没有返回 official 或 candidate 步骤。" />}
              {!showGeneratingSteps && !noStructuredSteps && steps.length === 0 && <EmptyEvidence title="暂无步骤" />}
            </div>
          </EvidenceCard>
        </aside>

        <main className="space-y-5">
          <EvidenceCard className="p-4">
            <div className="mb-3 flex items-center gap-2">
              <PlayCircle className="h-4 w-4 text-blue-600" />
              <h3 className="font-black text-slate-950">双视角视频</h3>
            </div>
            <DualVideoPlayer firstPersonUrl={sourceUrl} thirdPersonUrl={annotatedUrl} seekRequest={seekRequest} />
          </EvidenceCard>

          <EvidenceCard className="p-4">
            <h3 className="font-black text-slate-950">场景摘要</h3>
            <p className="mt-2 text-sm font-medium leading-6 text-slate-600">{cleanDisplayText(overview.scene_summary.description, '暂无场景摘要')}</p>
            <span className="sr-only">activities: {overview.scene_summary.activities.join(', ')}</span>
            <span className="sr-only">objects: {(overview.scene_summary.visible_lab_objects || overview.scene_summary.objects).join(', ')}</span>
            <span className="sr-only">step_indicators: {overview.scene_summary.step_indicators.join(', ')}</span>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <ListBox title="活动线索" items={overview.scene_summary.activities} />
              <ListBox title="确认对象" items={overview.scene_summary.visible_lab_objects || overview.scene_summary.objects} />
              <ListBox title="不确定对象" items={overview.scene_summary.uncertain_objects || []} />
              <ListBox title="步骤线索" items={overview.scene_summary.step_indicators} />
            </div>
            <div className="mt-3 rounded-lg bg-slate-50 p-3 text-sm font-semibold text-slate-500">证据来源：{overview.scene_summary.evidence_source || overview.scene_summary.evidence_note || 'analysis-overview'}</div>
          </EvidenceCard>
        </main>

        <aside className="space-y-5">
          <EvidenceCard className="p-4">
            <h3 className="mb-3 font-black text-slate-950">当前步骤</h3>
            {selectedStep ? (
              <div className="space-y-2 text-sm font-semibold text-slate-600">
                <Detail label="名称" value={cleanDisplayText(selectedStep.step_name)} />
                <Detail label="状态" value={selectedStep.status} />
                <Detail label="置信度" value={formatNumber(selectedStep.confidence, 3)} />
                <span className="sr-only">confidence: {formatNumber(selectedStep.confidence, 4)}</span>
                <Detail label="时间" value={formatRange(selectedStep.start_time_sec, selectedStep.end_time_sec)} />
                <Detail label="证据数" value={String(selectedStep.evidence_refs?.length || 0)} />
                <button type="button" onClick={() => jumpTo(selectedStep.start_time_sec)} className={`${secondaryButtonClass()} w-full`}>
                  <LocateFixed className="h-4 w-4" />
                  跳转视频
                </button>
              </div>
            ) : <EmptyEvidence title="未选择步骤" />}
          </EvidenceCard>

          <EvidenceCard className="p-4">
            <h3 className="mb-3 font-black text-slate-950">关键素材预览</h3>
            <div className="space-y-2">
              {materials.length === 0 ? <EmptyEvidence title="暂无素材" /> : materials.map((item, itemIndex) => (
                <Link key={materialKey(item, itemIndex)} to={`/experiments/${overview.experiment.experiment_id}/materials`} className="block rounded-lg border border-slate-200 p-3 hover:bg-slate-50">
                  <div className="line-clamp-2 text-sm font-black text-slate-950">{cleanDisplayText(item.display_name || item.event_type || item.item_id, '素材')}</div>
                  <div className="mt-1 text-xs font-semibold text-slate-500">{formatRange(item.time_start ?? item.timestamp_sec, item.time_end)}</div>
                </Link>
              ))}
            </div>
          </EvidenceCard>

          <EvidenceCard className="p-4">
            <h3 className="mb-3 font-black text-slate-950">快捷入口</h3>
            <div className="grid gap-2">
              <Link to={`/experiments/${overview.experiment.experiment_id}/report`} className={secondaryButtonClass()}><FileText className="h-4 w-4" />分析报告</Link>
              <Link to={`/experiments/${overview.experiment.experiment_id}/materials/timeline`} className={secondaryButtonClass('cyan')}><Clock3 className="h-4 w-4" />素材时间轴</Link>
              <Link to={`/experiments/${overview.experiment.experiment_id}/materials`} className={secondaryButtonClass('emerald')}><Boxes className="h-4 w-4" />关键素材</Link>
            </div>
          </EvidenceCard>
        </aside>
      </div>
    </div>
  )
}

function ExperimentTabs({ experimentId }: { experimentId: string }) {
  const tabClass = 'rounded-md px-3 py-1.5 text-sm font-bold text-slate-600 transition hover:bg-slate-100 hover:text-slate-950'
  return (
    <nav className="flex flex-wrap gap-2">
      <Link to={`/experiments/${experimentId}/workspace`} className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-bold text-white">分析概览</Link>
      <Link to={`/experiments/${experimentId}/report`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'report')} className={tabClass}>分析报告</Link>
      <Link to={`/experiments/${experimentId}/materials`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'materials')} className={tabClass}>关键素材</Link>
      <Link to={`/experiments/${experimentId}/materials/timeline`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'materialTimeline')} className={tabClass}>素材时间轴</Link>
      <Link to={`/experiments/${experimentId}/key-actions`} className={tabClass}>关键动作</Link>
    </nav>
  )
}

function ListBox({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="rounded-lg border border-slate-200 p-3">
      <div className="mb-2 text-xs font-black uppercase tracking-wide text-slate-400">{title}</div>
      <div className="flex flex-wrap gap-1.5">
        {items.length === 0 ? <EvidenceBadge>none</EvidenceBadge> : items.map((item, itemIndex) => <EvidenceBadge key={keyed(item, itemIndex, 'list-item')} tone="blue">{cleanDisplayText(item)}</EvidenceBadge>)}
      </div>
    </div>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-3 border-b border-slate-100 pb-2"><span className="text-slate-400">{label}</span><span className="text-right text-slate-800">{value}</span></div>
}
