import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, BadgeCheck, Boxes, Download, FileText, Layers3, Printer, ShieldAlert, Video } from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, ProgressStrip, primaryButtonClass, secondaryButtonClass, toneForStatus } from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import { mediaUrl } from '../mediaUrl'
import type { AnalysisOverview, KeyActionResults, MaterialSearchItem } from '../types'

function formatNumber(value: unknown, digits = 2) {
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue.toFixed(digits) : '-'
}

function keyed(value: unknown, index: number, prefix: string) {
  const raw = String(value ?? '').trim()
  return `${prefix}-${raw || 'item'}-${index}`
}

function materialKey(item: MaterialSearchItem, index: number) {
  return keyed(item.event_id || item.item_id || item.display_name || item.event_type, index, 'material')
}

export default function ExperimentReport() {
  const { id } = useParams<{ id: string }>()
  const [overview, setOverview] = useState<AnalysisOverview | null>(null)
  const [materials, setMaterials] = useState<MaterialSearchItem[]>([])
  const [keyResults, setKeyResults] = useState<KeyActionResults | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    setLoading(true)
    Promise.all([
      experimentApi.getAnalysisOverview(id, { force: true }),
      experimentApi.getPublishedMaterials(id, { limit: 12 }, { force: true }).catch(() => ({ items: [] })),
      experimentApi.getKeyActionResults(id).catch(() => null),
    ])
      .then(([nextOverview, nextMaterials, nextKeyResults]) => {
        setOverview(nextOverview)
        setMaterials((nextMaterials.items || []) as MaterialSearchItem[])
        setKeyResults(nextKeyResults)
      })
      .catch(exc => setError(exc instanceof Error ? exc.message : '报告加载失败'))
      .finally(() => setLoading(false))
  }, [id])

  const steps = useMemo(() => overview ? [...overview.steps.official, ...overview.steps.candidate, ...overview.steps.inferred] : [], [overview])
  const pendingMaterials = materials.filter(item => String(item.review_status || item.payload?.review_status || '').toLowerCase().includes('pending')).length
  const strongMaterials = materials.filter(item => String(item.evidence_level || item.payload?.evidence_level || '').toLowerCase().includes('strong')).length
  const professionalPdf = overview?.artifacts.professional_report_pdf
  const professionalHtml = overview?.artifacts.professional_report_html

  if (loading) return <EmptyEvidence title="正在生成报告视图..." />
  if (error) return <EvidenceCard className="border-red-200 bg-red-50 p-5 text-red-700">{error}</EvidenceCard>
  if (!overview) return <EmptyEvidence title="暂无报告数据" />

  return (
    <div className="space-y-5">
      <PageHero
        eyebrow={<Link to={`/experiments/${overview.experiment.experiment_id}/workspace`} className="hover:text-slate-900">分析概览</Link>}
        title="分析报告"
        description="将结构化步骤、风险告警、关键素材和关键动作证据汇总成可复核报告。"
        actions={(
          <>
            <Link to={`/experiments/${overview.experiment.experiment_id}/workspace`} className={secondaryButtonClass()}><ArrowLeft className="h-4 w-4" />工作台</Link>
            {professionalPdf?.ready && professionalPdf.url && (
              <a href={mediaUrl(professionalPdf.url)} target="_blank" rel="noreferrer" className={primaryButtonClass('emerald')}>
                <Download className="h-4 w-4" />
                专业PDF报告
              </a>
            )}
            {professionalHtml?.ready && professionalHtml.url && (
              <a href={mediaUrl(professionalHtml.url)} target="_blank" rel="noreferrer" className={secondaryButtonClass('cyan')}>
                <FileText className="h-4 w-4" />
                HTML报告
              </a>
            )}
            <button type="button" onClick={() => window.print()} className={secondaryButtonClass()}>
              <Printer className="h-4 w-4" />
              打印
            </button>
            <Link to={`/experiments/${overview.experiment.experiment_id}/key-actions`} className={primaryButtonClass('blue')}><Layers3 className="h-4 w-4" />关键动作</Link>
          </>
        )}
        tabs={<Tabs id={overview.experiment.experiment_id} />}
      />

      <ProgressStrip status={overview.run.status} progress={overview.run.progress} message={`${overview.run.stage} · ${overview.run.updated_at || ''}`} />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <MetricTile label="结构步骤" value={steps.length} helper={`${overview.summary.confirmed_step_count} confirmed`} tone="blue" Icon={BadgeCheck} />
        <MetricTile label="平均置信度" value={formatNumber(overview.summary.avg_confidence, 3)} helper={overview.summary.model_name} tone="emerald" Icon={FileText} />
        <MetricTile label="风险告警" value={overview.alerts.length} helper="rule matches" tone={overview.alerts.length ? 'red' : 'slate'} Icon={ShieldAlert} />
        <MetricTile label="强证据素材" value={strongMaterials} helper={`${pendingMaterials} pending`} tone="emerald" Icon={Boxes} />
        <MetricTile label="微片段" value={keyResults?.micro_segments?.length || 0} helper={`${keyResults?.interaction_events?.length || 0} interactions`} tone="violet" Icon={Layers3} />
      </section>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <main className="space-y-5">
          <EvidenceCard className="p-5">
            <h3 className="font-black text-slate-950">实验摘要</h3>
            <p className="mt-2 text-sm font-medium leading-6 text-slate-600">{cleanDisplayText(overview.scene_summary.description, '暂无摘要')}</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {overview.scene_summary.activities.map((item, itemIndex) => <EvidenceBadge key={keyed(item, itemIndex, 'activity')} tone="blue">{cleanDisplayText(item)}</EvidenceBadge>)}
              {(overview.scene_summary.visible_lab_objects || overview.scene_summary.objects).map((item, itemIndex) => <EvidenceBadge key={keyed(item, itemIndex, 'object')} tone="cyan">{cleanDisplayText(item)}</EvidenceBadge>)}
            </div>
          </EvidenceCard>

          <EvidenceCard className="p-5">
            <h3 className="mb-4 font-black text-slate-950">步骤证据</h3>
            <div className="space-y-3">
              {steps.length === 0 ? <EmptyEvidence title="暂无结构化步骤" /> : steps.map(step => (
                <div key={step.step_id} className="rounded-lg border border-slate-200 p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="font-black text-slate-950">{step.step_index}. {cleanDisplayText(step.step_name, '步骤')}</div>
                      <div className="mt-1 text-sm font-medium text-slate-500">{cleanDisplayText(step.step_description, '暂无描述')}</div>
                    </div>
                    <EvidenceBadge tone={toneForStatus(step.status)}>{step.status}</EvidenceBadge>
                  </div>
                  <div className="mt-2 text-xs font-bold text-slate-500">{formatNumber(step.start_time_sec, 2)}-{formatNumber(step.end_time_sec, 2)}s · confidence {formatNumber(step.confidence, 3)}</div>
                </div>
              ))}
            </div>
          </EvidenceCard>
        </main>

        <aside className="space-y-5">
          <EvidenceCard className="p-4">
            <h3 className="mb-3 font-black text-slate-950">关键动作摘要</h3>
            {keyResults ? (
              <div className="space-y-2 text-sm font-semibold text-slate-600">
                <Detail label="segments" value={String(keyResults.segments.length)} />
                <Detail label="micro segments" value={String(keyResults.micro_segments?.length || 0)} />
                <Detail label="interactions" value={String(keyResults.interaction_events?.length || 0)} />
                <Detail label="vector items" value={String((keyResults.vector_metadata?.length || 0) + (keyResults.micro_vector_metadata?.length || 0))} />
              </div>
            ) : <EmptyEvidence title="关键动作结果未就绪" />}
          </EvidenceCard>

          <EvidenceCard className="p-4">
            <h3 className="mb-3 font-black text-slate-950">关键素材摘要</h3>
            <div className="space-y-2">
              {materials.length === 0 ? <EmptyEvidence title="暂无素材" /> : materials.slice(0, 8).map((item, itemIndex) => (
                <Link key={materialKey(item, itemIndex)} to={`/experiments/${overview.experiment.experiment_id}/materials`} className="block rounded-lg border border-slate-200 p-3 hover:bg-slate-50">
                  <div className="line-clamp-2 text-sm font-black text-slate-950">{cleanDisplayText(item.display_name || item.event_type || item.item_id, '素材')}</div>
                  <div className="mt-1 flex flex-wrap gap-1.5">{[item.review_status, item.evidence_level, item.event_type].filter(Boolean).map((label, labelIndex) => <EvidenceBadge key={keyed(label, labelIndex, 'material-badge')}>{String(label)}</EvidenceBadge>)}</div>
                </Link>
              ))}
            </div>
          </EvidenceCard>

          <EvidenceCard className="p-4">
            <h3 className="mb-3 font-black text-slate-950">附件入口</h3>
            <div className="grid gap-2">
              <Link to={`/experiments/${overview.experiment.experiment_id}/materials`} className={secondaryButtonClass('cyan')}><Boxes className="h-4 w-4" />素材库</Link>
              <Link to={`/experiments/${overview.experiment.experiment_id}/workspace`} className={secondaryButtonClass('blue')}><Video className="h-4 w-4" />视频工作台</Link>
            </div>
          </EvidenceCard>
        </aside>
      </div>
    </div>
  )
}

function Tabs({ id }: { id: string }) {
  const cls = 'rounded-md px-3 py-1.5 text-sm font-bold text-slate-600 hover:bg-slate-100'
  return (
    <nav className="flex flex-wrap gap-2">
      <Link to={`/experiments/${id}/workspace`} onMouseEnter={() => prefetchExperimentRoute(id, 'workspace')} className={cls}>分析概览</Link>
      <Link to={`/experiments/${id}/report`} className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-bold text-white">分析报告</Link>
      <Link to={`/experiments/${id}/materials`} className={cls}>关键素材</Link>
      <Link to={`/experiments/${id}/materials/timeline`} className={cls}>素材时间轴</Link>
      <Link to={`/experiments/${id}/key-actions`} className={cls}>关键动作</Link>
    </nav>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-3 border-b border-slate-100 pb-2"><span className="text-slate-400">{label}</span><span className="text-right text-slate-800">{value}</span></div>
}
