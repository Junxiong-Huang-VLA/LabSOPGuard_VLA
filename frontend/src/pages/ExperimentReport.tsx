import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, BadgeCheck, Boxes, Download, FileText, Layers3, Printer, ShieldAlert, Video } from 'lucide-react'
import { experimentApi } from '../api'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, primaryButtonClass, secondaryButtonClass, toneForStatus } from '../components/EvidenceUI'
import ExperimentPageShell from '../components/ExperimentSideNav'
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

function materialPayload(item: MaterialSearchItem) {
  return item.payload && typeof item.payload === 'object' ? item.payload as Record<string, unknown> : {}
}

function pathText(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value
    if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  }
  return undefined
}

function materialFieldValue(item: MaterialSearchItem, key: string) {
  const record = item as Record<string, unknown>
  const payload = materialPayload(item)
  return record[key] ?? payload[key]
}

function materialFieldText(item: MaterialSearchItem, key: string) {
  return pathText(materialFieldValue(item, key))
}

function materialFieldList(item: MaterialSearchItem, key: string) {
  const value = materialFieldValue(item, key)
  if (Array.isArray(value)) return value.map(entry => String(entry)).filter(Boolean)
  if (typeof value === 'string' && value.trim()) return [value.trim()]
  return []
}

function uniqueTexts(values: Array<string | undefined>) {
  const seen = new Set<string>()
  for (const value of values) {
    const text = String(value || '').trim()
    if (text) seen.add(text)
  }
  return Array.from(seen)
}

function formatSummary(values: string[], fallback = '-') {
  if (!values.length) return fallback
  const visible = values.slice(0, 3).map(value => cleanDisplayText(value))
  return `${visible.join(' / ')}${values.length > 3 ? ` +${values.length - 3}` : ''}`
}

const MISSING_TARGET_REASON_CODES = ['no_target_box', 'no_hand_target_interaction_evidence', 'low_evidence', 'insufficient_evidence', 'evidence_insufficient', 'no_valid_evidence', '证据不足']

function materialMissingEvidenceReasons(item: MaterialSearchItem) {
  const fields = ['rerender_error', 'error', 'missing_reason', 'reason', 'quality_reasons', 'warnings', 'review_reason_codes']
  const reasons = fields.flatMap(field => [
    materialFieldText(item, field),
    ...materialFieldList(item, field),
  ])
  return uniqueTexts(reasons.filter(reason => MISSING_TARGET_REASON_CODES.some(code => String(reason || '').toLowerCase().includes(code))))
}

const MATERIAL_LABEL_ZH: Record<string, string> = {
  'hand-paper': '手部与称量纸操作',
  'hand-bottle': '手部与试剂瓶操作',
  'hand-balance': '天平设备面板操作',
  'hand-spatula': '手部与药匙操作',
  'hand-container': '手部与容器操作',
  weighing_paper_operation: '手部与称量纸操作',
  balance_operation: '天平设备面板操作',
  pipetting_operation: '移液操作',
  bottle_operation: '试剂瓶操作',
  liquid_transfer: '液体转移操作',
  hand_object_contact: '手物接触',
  object_movement_detected: '物体移动',
  container_state_change: '容器状态变化',
  paper: '称量纸',
  reagent_bottle: '试剂瓶',
  balance: '天平',
  spatula: '药匙',
  container: '容器',
  beaker: '烧杯',
  pipette: '移液器',
  tube: '试管',
  sample_bottle: '样品瓶',
}

function zhLabel(value?: unknown) {
  const text = pathText(value)
  if (!text) return ''
  return MATERIAL_LABEL_ZH[text] || MATERIAL_LABEL_ZH[text.toLowerCase()] || cleanDisplayText(text)
}

function statusLabel(value?: unknown) {
  const raw = String(value || '').toLowerCase()
  return {
    completed: '已完成',
    analyzed: '已分析',
    partial_failed: '部分待复核',
    failed: '待复核',
    running: '处理中',
    accepted: '已接收',
    approved: '已批准',
    confirmed: '已确认',
    needs_review: '待确认',
    pending: '待确认',
    unreviewed: '未审核',
  }[raw] || cleanDisplayText(value, '待确认')
}

function reportMaterialTitle(item: MaterialSearchItem) {
  const display = pathText(item.display_title, item.display_name)
  if (display) return cleanDisplayText(display)
  const action = zhLabel(materialFieldValue(item, 'canonical_action_type') || materialFieldValue(item, 'event_type'))
  const object = zhLabel(materialFieldValue(item, 'canonical_object') || materialFieldValue(item, 'primary_object'))
  return [action, object].filter(Boolean).join(' / ') || cleanDisplayText(item.item_id || item.event_id, '关键素材')
}

function reportMaterialTime(item: MaterialSearchItem) {
  const start = Number(materialFieldValue(item, 'time_start') ?? materialFieldValue(item, 'timestamp_sec'))
  const end = Number(materialFieldValue(item, 'time_end'))
  if (!Number.isFinite(start)) return '时间戳待补充'
  const endText = Number.isFinite(end) ? `${formatNumber(end, 2)}s` : '-'
  return `${formatNumber(start, 2)}s 至 ${endText}`
}

function reportMaterialSource(item: MaterialSearchItem) {
  const raw = pathText(item.experiment_window_id, item.segment_id, item.parent_segment_id)
  if (!raw) return '来源片段待补充'
  const match = raw.match(/(?:episode|formal_window|seg)_(\d+)/i) || raw.match(/(\d+)$/)
  return match ? `实验片段 ${Number(match[1])}` : cleanDisplayText(raw)
}

function reportMaterialPreview(item: MaterialSearchItem) {
  const raw = pathText(
    materialFieldValue(item, 'preview_url'),
    materialFieldValue(item, 'material_url'),
    materialFieldValue(item, 'first_keyframe'),
    materialFieldValue(item, 'third_keyframe'),
    materialFieldValue(item, 'frame_path'),
  )
  return raw ? mediaUrl(raw) : ''
}

function reportMaterialAction(item: MaterialSearchItem) {
  return zhLabel(
    materialFieldValue(item, 'canonical_action_type')
      || materialFieldValue(item, 'event_type')
      || materialFieldValue(item, 'action_name'),
  ) || '关键动作'
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
  const pendingMaterials = materials.filter(item => {
    const status = String(item.review_status || item.payload?.review_status || '').toLowerCase()
    return status.includes('pending') || status.includes('needs_review') || status.includes('unreviewed')
  }).length
  const flaggedMaterials = materials.filter(item => materialMissingEvidenceReasons(item).length > 0 || Boolean(materialFieldText(item, 'rerender_error'))).length
  const materialActions = uniqueTexts(materials.map(reportMaterialAction)).filter(Boolean)
  const microSegmentCount = keyResults?.micro_segments?.length || 0
  const interactionCount = keyResults?.interaction_events?.length || 0
  const reviewNotes = [
    pendingMaterials > 0 ? `${pendingMaterials} 个关键素材需要确认` : '',
    flaggedMaterials > 0 ? `${flaggedMaterials} 个素材证据需要复核` : '',
    overview?.alerts.length ? `${overview.alerts.length} 条风险告警需要查看` : '',
  ].filter(Boolean)
  const professionalPdf = overview?.artifacts.professional_report_pdf
  const professionalHtml = overview?.artifacts.professional_report_html

  if (loading) return <EmptyEvidence title="正在生成报告视图..." />
  if (error) return <EvidenceCard className="border-red-200 bg-red-50 p-5 text-red-700">{error}</EvidenceCard>
  if (!overview) return <EmptyEvidence title="暂无报告数据" />

  return (
    <ExperimentPageShell experimentId={overview.experiment.experiment_id}>
    <div className="space-y-5">
      <PageHero
        eyebrow={<Link to={`/experiments/${overview.experiment.experiment_id}/workspace`} className="hover:text-slate-900">实验工作台</Link>}
        title="实验日报"
        description="把本次实验整理成一份可读日报：做了什么、留下了哪些关键素材、还有哪些需要确认。"
        actions={(
          <>
            <Link to={`/experiments/${overview.experiment.experiment_id}/workspace`} className={secondaryButtonClass()}><ArrowLeft className="h-4 w-4" />返回工作台</Link>
            {professionalPdf?.ready && professionalPdf.url && (
              <a href={mediaUrl(professionalPdf.url)} target="_blank" rel="noreferrer" className={primaryButtonClass('emerald')}>
                <Download className="h-4 w-4" />
                下载日报PDF
              </a>
            )}
            {professionalHtml?.ready && professionalHtml.url && (
              <a href={mediaUrl(professionalHtml.url)} target="_blank" rel="noreferrer" className={secondaryButtonClass('cyan')}>
                <FileText className="h-4 w-4" />
                打开HTML日报
              </a>
            )}
            <button type="button" onClick={() => window.print()} className={secondaryButtonClass()}>
              <Printer className="h-4 w-4" />
              打印
            </button>
            <Link to={`/experiments/${overview.experiment.experiment_id}/materials`} className={primaryButtonClass('blue')}><Boxes className="h-4 w-4" />关键素材</Link>
          </>
        )}
      />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <MetricTile label="实验步骤" value={steps.length} helper={`${overview.summary.confirmed_step_count} 个已确认`} tone="blue" Icon={BadgeCheck} />
        <MetricTile label="关键动作" value={materialActions.length || interactionCount} helper={formatSummary(materialActions, '动作类型待确认')} tone="emerald" Icon={FileText} />
        <MetricTile label="需要确认" value={overview.alerts.length + pendingMaterials + flaggedMaterials} helper="建议人工看一眼" tone={(overview.alerts.length + pendingMaterials + flaggedMaterials) ? 'amber' : 'slate'} Icon={ShieldAlert} />
        <MetricTile label="关键素材" value={materials.length} helper={`${pendingMaterials} 个待确认 / ${flaggedMaterials} 个需复核`} tone="emerald" Icon={Boxes} />
        <MetricTile label="视频片段" value={microSegmentCount} helper={`${interactionCount} 条关键操作记录`} tone="violet" Icon={Layers3} />
      </section>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <main className="space-y-5">
          <EvidenceCard className="p-5">
            <h3 className="mb-3 font-black text-slate-950">这次实验说明什么</h3>
            <p className="text-sm leading-7 text-slate-700">
              这次实验已经整理出 {steps.length} 个实验步骤、{materials.length} 个关键素材和 {microSegmentCount} 个可回看的视频片段。
              {materialActions.length ? ` 主要实验操作包括：${formatSummary(materialActions)}。` : ' 主要实验操作仍待进一步确认。'}
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <EvidenceBadge tone={toneForStatus(overview.run.status)}>{statusLabel(overview.run.status)}</EvidenceBadge>
              {overview.run.updated_at ? <EvidenceBadge tone="slate">更新时间 {cleanDisplayText(overview.run.updated_at)}</EvidenceBadge> : null}
            </div>
          </EvidenceCard>

          <EvidenceCard className="p-5">
            <h3 className="mb-4 font-black text-slate-950">实验过程时间线</h3>
            <div className="space-y-3">
              {steps.length === 0 ? <EmptyEvidence title="暂无实验过程记录" description="实验步骤整理完成后会在这里形成时间线。" /> : steps.map(step => (
                <div key={step.step_id} className="rounded-lg border border-slate-200 p-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="font-black text-slate-950">{step.step_index}. {cleanDisplayText(step.step_name, '步骤')}</div>
                      <div className="mt-1 text-sm font-medium text-slate-500">{cleanDisplayText(step.step_description, '暂无描述')}</div>
                    </div>
                    <EvidenceBadge tone={toneForStatus(step.status)}>{statusLabel(step.status)}</EvidenceBadge>
                  </div>
                  <div className="mt-2 text-xs font-bold text-slate-500">时间戳 {formatNumber(step.start_time_sec, 2)}s 至 {formatNumber(step.end_time_sec, 2)}s</div>
                </div>
              ))}
            </div>
          </EvidenceCard>

          <EvidenceCard className="p-5">
            <h3 className="mb-4 font-black text-slate-950">值得回看的关键素材</h3>
            {materials.length === 0 ? (
              <EmptyEvidence title="暂无关键素材" description="关键帧和关键片段确认后会在这里展示。" />
            ) : (
              <div className="grid gap-3 md:grid-cols-2">
                {materials.slice(0, 6).map((item, itemIndex) => {
                  const preview = reportMaterialPreview(item)
                  return (
                    <Link key={materialKey(item, itemIndex)} to={`/experiments/${overview.experiment.experiment_id}/materials`} className="overflow-hidden rounded-xl border border-slate-200 bg-white transition hover:-translate-y-0.5 hover:shadow-sm">
                      {preview ? (
                        <div className="aspect-video overflow-hidden bg-slate-100">
                          <img src={preview} alt={reportMaterialTitle(item)} className="h-full w-full object-cover" />
                        </div>
                      ) : null}
                      <div className="p-3">
                        <div className="line-clamp-2 text-sm font-black text-slate-950">{reportMaterialTitle(item)}</div>
                        <div className="mt-2 space-y-1 text-xs font-semibold text-slate-500">
                          <div>{reportMaterialSource(item)}</div>
                          <div>时间戳 {reportMaterialTime(item)}</div>
                          <div>动作类型 {reportMaterialAction(item)}</div>
                        </div>
                      </div>
                    </Link>
                  )
                })}
              </div>
            )}
          </EvidenceCard>
        </main>

        <aside className="space-y-5">
          <EvidenceCard className="p-4">
            <h3 className="mb-3 font-black text-slate-950">需要你确认的事项</h3>
            {reviewNotes.length === 0 ? (
              <p className="text-sm font-semibold text-slate-600">暂无需要优先处理的事项。</p>
            ) : (
              <div className="space-y-2">
                {reviewNotes.map((note, index) => (
                  <div key={keyed(note, index, 'review-note')} className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm font-semibold text-amber-900">
                    {note}
                  </div>
                ))}
              </div>
            )}
          </EvidenceCard>

          <EvidenceCard className="p-4">
            <h3 className="mb-3 font-black text-slate-950">素材与实验沉淀</h3>
            {keyResults ? (
              <div className="space-y-2 text-sm font-semibold text-slate-600">
                <Detail label="实验片段" value={String(keyResults.segments.length)} />
                <Detail label="视频片段" value={String(microSegmentCount)} />
                <Detail label="关键操作" value={String(interactionCount)} />
                <Detail label="关键素材" value={String(materials.length)} />
                <Detail label="操作类型" value={formatSummary(materialActions, '待确认')} />
              </div>
            ) : <EmptyEvidence title="素材整理结果未就绪" />}
          </EvidenceCard>

          <EvidenceCard className="p-4">
            <h3 className="mb-3 font-black text-slate-950">建议下一步</h3>
            <div className="space-y-2">
              <p className="text-sm leading-6 text-slate-600">
                优先确认待处理关键素材；如果某些实验操作缺少第一人称或第三人称画面，下一次采集时可以补齐对应视角或适当延长关键片段。
              </p>
            </div>
          </EvidenceCard>

          <EvidenceCard className="p-4">
            <h3 className="mb-3 font-black text-slate-950">附件入口</h3>
            <div className="grid gap-2">
              <Link to={`/experiments/${overview.experiment.experiment_id}/materials`} className={secondaryButtonClass('cyan')}><Boxes className="h-4 w-4" />关键素材</Link>
              <Link to={`/experiments/${overview.experiment.experiment_id}/workspace`} className={secondaryButtonClass('blue')}><Video className="h-4 w-4" />视频工作台</Link>
            </div>
          </EvidenceCard>
        </aside>
      </div>
    </div>
    </ExperimentPageShell>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return <div className="flex justify-between gap-3 border-b border-slate-100 pb-2"><span className="shrink-0 text-slate-400">{label}</span><span className="min-w-0 break-words text-right text-slate-800">{value}</span></div>
}
