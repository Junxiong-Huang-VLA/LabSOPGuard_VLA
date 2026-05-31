import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  AlertCircle,
  ArrowLeft,
  BadgeCheck,
  Boxes,
  DatabaseZap,
  FileText,
  Gauge,
  Layers3,
  ListChecks,
  Loader2,
  PlayCircle,
  Search,
  SlidersHorizontal,
  Video,
  Zap,
} from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import {
  EmptyEvidence,
  EvidenceBadge,
  EvidenceCard,
  MetricTile,
  PageHero,
  ProgressStrip,
  primaryButtonClass,
  secondaryButtonClass,
  toneForStatus,
} from '../components/EvidenceUI'
import type { Tone } from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import { experimentFileUrl } from '../mediaUrl'
import type {
  KeyActionClipRef,
  KeyActionDetectorConfig,
  KeyActionInteractionEvent,
  KeyActionInteractionKeyframe,
  KeyActionMicroSegment,
  KeyActionResults,
  KeyActionSegment,
  KeyActionStatus,
} from '../types'

type QueryIndexLevel = 'all' | 'segment' | 'micro_segment'
type QueryRow = Record<string, unknown>

const POLLING_STATUSES = new Set(['queued', 'running'])
const DEFAULT_QUERY = '查找使用移液枪加样、手部接触容器或关键试剂转移的片段'

const ACTION_LABELS: Record<string, string> = {
  complete_experiment_episode: '完整实验片段',
  liquid_transfer: '液体转移',
  container_state_change: '容器状态变化',
  panel_operation: '设备操作',
  weighing: '称量操作',
  hand_object_interaction: '手物交互',
}

function formatNumber(value: unknown, digits = 2) {
  if (value == null || value === '') return '-'
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue.toFixed(digits) : '-'
}

function formatCount(value: unknown) {
  const numberValue = Number(value ?? 0)
  return Number.isFinite(numberValue) ? String(numberValue) : '0'
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}

function recordString(record: Record<string, unknown> | null | undefined, key: string, fallback = '') {
  const value = record?.[key]
  return value == null || value === '' ? fallback : String(value)
}

function recordNumber(record: Record<string, unknown> | null | undefined, key: string) {
  const value = record?.[key]
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

function uniqueCleanList(values: unknown[], limit = 8) {
  const seen = new Set<string>()
  const items: string[] = []
  for (const value of values) {
    const text = cleanDisplayText(String(value || '').replace(/_/g, ' '), '').trim()
    if (!text || text.length > 42 || /[\[\]{}]/.test(text)) continue
    const key = text.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    items.push(text)
    if (items.length >= limit) break
  }
  return items
}

function statusLabel(status?: string | null) {
  const value = String(status || '').toLowerCase()
  if (value === 'completed') return '已完成'
  if (value === 'running') return '运行中'
  if (value === 'queued') return '排队中'
  if (value === 'failed') return '失败'
  if (value === 'not_started') return '未开始'
  return status || '未知'
}

function evidenceTone(level?: unknown): Tone {
  const value = String(level || '').toLowerCase()
  if (value.includes('strong') || value.includes('trusted') || value.includes('confirmed')) return 'emerald'
  if (value.includes('moderate') || value.includes('review')) return 'blue'
  if (value.includes('insufficient') || value.includes('weak')) return 'red'
  return 'amber'
}

function confidenceTone(confidence?: string | null, score?: unknown): Tone {
  const normalized = String(confidence || '').toLowerCase()
  const numeric = Number(score)
  if (normalized.includes('high') || numeric >= 0.75) return 'emerald'
  if (normalized.includes('low') || (Number.isFinite(numeric) && numeric < 0.4)) return 'red'
  return 'amber'
}

function friendlyActionName(value?: string | null) {
  const raw = String(value || '').toLowerCase()
  return ACTION_LABELS[raw] || cleanDisplayText(raw.replace(/_/g, ' '), '关键动作')
}

function segmentTitle(segment: KeyActionSegment, index: number) {
  const summary = cleanDisplayText(segment.text_description?.summary, '')
  if (summary && summary.length <= 90 && !/unknown_operation|complete_experiment_episode|[\[\]{}]/.test(summary)) {
    return summary
  }
  const objects = uniqueCleanList([
    ...(segment.text_description?.objects || []),
    ...(segment.visual_keywords || []),
    ...Object.keys(segment.yolo_label_counts || {}),
    ...(segment.yolo_labels || []),
  ], 4)
  const action = friendlyActionName(segment.text_description?.action_type)
  return objects.length ? `${action} · ${objects.join(' / ')}` : `${action} ${index + 1}`
}

function microTitle(micro: KeyActionMicroSegment, index: number) {
  const summary = cleanDisplayText(micro.text_description?.summary, '')
  if (summary) return summary
  const objectName = cleanDisplayText(micro.primary_object || micro.interaction?.primary_object || '', '')
  const interaction = cleanDisplayText(micro.interaction_type || micro.interaction?.interaction_type || '', '')
  return [interaction, objectName].filter(Boolean).join(' · ') || micro.display_id || `Micro ${index + 1}`
}

function clipUrl(clip?: KeyActionClipRef | null, experimentId?: string | null) {
  return experimentFileUrl(clip?.annotated_clip_url || clip?.clip_url || undefined, experimentId || undefined)
}

function clipMode(clip?: KeyActionClipRef | null) {
  if (!clip) return '无片段'
  return clip.annotated_clip_url ? 'YOLO 标注片段' : '原始片段'
}

function segmentTimeRange(segment: KeyActionSegment) {
  const start = segment.third_person?.local_start_sec ?? segment.first_person?.local_start_sec
  const end = segment.third_person?.local_end_sec ?? segment.first_person?.local_end_sec
  if (start != null || end != null) return `${formatNumber(start, 2)}-${formatNumber(end, 2)}s`
  return `${segment.global_start_time || '-'} - ${segment.global_end_time || '-'}`
}

function microTimeRange(micro: KeyActionMicroSegment) {
  const start = micro.start_sec ?? micro.first_person?.local_start_sec ?? micro.third_person?.local_start_sec
  const end = micro.end_sec ?? micro.first_person?.local_end_sec ?? micro.third_person?.local_end_sec
  if (start != null || end != null) return `${formatNumber(start, 2)}-${formatNumber(end, 2)}s`
  return `${micro.global_start_time || '-'} - ${micro.global_end_time || '-'}`
}

function yoloLabelEntries(segment: KeyActionSegment) {
  const counts = Object.entries(segment.yolo_label_counts || {})
  if (counts.length) return counts.map(([label, count]) => [label, count] as [string, number | null])
  return (segment.yolo_labels || []).map(label => [label, null] as [string, number | null])
}

function eventsForSegment(segment: KeyActionSegment | null, allEvents: KeyActionInteractionEvent[]) {
  if (!segment) return []
  const local = segment.interaction_events || []
  if (local.length) return local
  return allEvents.filter(event => event.segment_id === segment.segment_id)
}

function keyframesForSegment(segment: KeyActionSegment | null, allKeyframes: KeyActionInteractionKeyframe[]) {
  if (!segment) return []
  const local = segment.interaction_keyframes || []
  if (local.length) return local
  return allKeyframes.filter(frame => frame.segment_id === segment.segment_id)
}

function microSegmentsForSegment(segment: KeyActionSegment | null, allMicroSegments: KeyActionMicroSegment[]) {
  if (!segment) return []
  const local = segment.micro_segments || []
  if (local.length) return local
  return allMicroSegments.filter(item => item.parent_segment_id === segment.segment_id)
}

function interactionLabel(event: KeyActionInteractionEvent | KeyActionInteractionKeyframe) {
  const raw = event.display_name
    || ('stable_name' in event ? event.stable_name : null)
    || event.interaction
    || ('object_name' in event ? event.object_name : null)
    || ('object_label' in event ? event.object_label : null)
    || event.event_type
    || event.event_id
  return cleanDisplayText(String(raw || 'interaction').replace(/_/g, ' '), '交互证据')
}

function interactionObjectText(event: KeyActionInteractionEvent) {
  return uniqueCleanList([
    event.object_name,
    event.object_label,
    ...(event.involved_objects || []),
    ...(event.related_detection_classes || []),
  ], 4).join(' / ') || '对象待确认'
}

function keyframeUrl(frame: KeyActionInteractionKeyframe, experimentId?: string | null) {
  return experimentFileUrl(frame.url || frame.preview_url || frame.path || undefined, experimentId || undefined)
}

function microKeyframeUrls(micro: KeyActionMicroSegment, experimentId?: string | null) {
  const rawUrls = [
    ...(micro.keyframes?.urls || []),
    micro.keyframes?.contact_frame_url,
    micro.keyframes?.peak_frame_url,
    micro.keyframes?.release_frame_url,
    micro.keyframes?.contact_frame,
    micro.keyframes?.peak_frame,
    micro.keyframes?.release_frame,
    micro.peak_keyframe,
  ]
  return Array.from(new Set(rawUrls.map(item => experimentFileUrl(item || undefined, experimentId || undefined)).filter(Boolean) as string[]))
}

function detectorConfig(results: KeyActionResults | null): KeyActionDetectorConfig | null {
  if (!results) return null
  return results.detection_config || results.detector_config || results.debug?.detector_config || null
}

function readError(error: unknown) {
  const record = asRecord(error)
  const response = asRecord(record?.response)
  const data = asRecord(response?.data)
  return recordString(data, 'detail') || recordString(record, 'message', '请求失败')
}

function queryResultTitle(row: QueryRow, index: number) {
  const metadata = asRecord(row.metadata)
  return cleanDisplayText(
    recordString(row, 'title')
      || recordString(row, 'display_name')
      || recordString(metadata, 'display_name')
      || recordString(row, 'segment_id')
      || recordString(row, 'micro_segment_id')
      || `结果 ${index + 1}`,
  )
}

function queryResultText(row: QueryRow) {
  const metadata = asRecord(row.metadata)
  return cleanDisplayText(
    recordString(row, 'text')
      || recordString(row, 'index_text')
      || recordString(row, 'index_text_preview')
      || recordString(row, 'content')
      || recordString(metadata, 'index_text')
      || recordString(metadata, 'summary')
      || '该命中缺少索引文本。',
  )
}

function queryResultLevel(row: QueryRow) {
  const metadata = asRecord(row.metadata)
  const level = recordString(row, 'index_level') || recordString(metadata, 'index_level')
  if (level === 'micro_segment') return '微片段'
  if (level === 'segment') return '片段'
  return level || '混合'
}

function queryResultScore(row: QueryRow) {
  const score = recordNumber(row, 'score') ?? recordNumber(row, 'similarity') ?? recordNumber(asRecord(row.metadata), 'score')
  return score == null ? '-' : formatNumber(score, 3)
}

function queryResultSegmentId(row: QueryRow) {
  const metadata = asRecord(row.metadata)
  return recordString(row, 'segment_id') || recordString(metadata, 'segment_id') || recordString(row, 'parent_segment_id') || recordString(metadata, 'parent_segment_id')
}

function queryResultMicroId(row: QueryRow) {
  const metadata = asRecord(row.metadata)
  return recordString(row, 'micro_segment_id') || recordString(metadata, 'micro_segment_id')
}

export default function KeyActionIndex() {
  const { id } = useParams<{ id: string }>()
  const [status, setStatus] = useState<KeyActionStatus | null>(null)
  const [results, setResults] = useState<KeyActionResults | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null)
  const [selectedMicroId, setSelectedMicroId] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'simple' | 'advanced'>('simple')
  const [query, setQuery] = useState(DEFAULT_QUERY)
  const [indexLevel, setIndexLevel] = useState<QueryIndexLevel>('all')
  const [primaryObject, setPrimaryObject] = useState('')
  const [interactionType, setInteractionType] = useState('')
  const [queryResults, setQueryResults] = useState<QueryRow[]>([])
  const [queryValidation, setQueryValidation] = useState<Record<string, unknown> | null>(null)
  const [queryLoading, setQueryLoading] = useState(false)
  const [queryError, setQueryError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    const experimentId = id
    let cancelled = false
    let timer: number | undefined

    async function refresh() {
      try {
        const nextStatus = await experimentApi.getKeyActionStatus(experimentId)
        if (cancelled) return
        setStatus(nextStatus)
        const currentStatus = String(nextStatus.status || '').toLowerCase()
        if (currentStatus === 'completed') {
          const nextResults = await experimentApi.getKeyActionResults(experimentId)
          if (cancelled) return
          setResults(nextResults)
          setSelectedSegmentId(previous => (
            nextResults.segments?.some(segment => segment.segment_id === previous)
              ? previous
              : nextResults.segments?.[0]?.segment_id || null
          ))
        } else {
          setResults(null)
          if (POLLING_STATUSES.has(currentStatus)) timer = window.setTimeout(refresh, 2000)
        }
        setError(null)
      } catch (exc) {
        if (!cancelled) setError(readError(exc))
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void refresh()
    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [id])

  const segments = results?.segments ?? []
  const microSegments = results?.micro_segments ?? []
  const interactionEvents = results?.interaction_events ?? []
  const interactionKeyframes = results?.interaction_keyframes ?? []
  const selectedSegment = useMemo(() => {
    return segments.find(segment => segment.segment_id === selectedSegmentId) || segments[0] || null
  }, [segments, selectedSegmentId])
  const selectedMicroSegments = useMemo(() => {
    return microSegmentsForSegment(selectedSegment, microSegments)
  }, [microSegments, selectedSegment])
  const selectedMicro = useMemo(() => {
    return selectedMicroSegments.find(item => item.micro_segment_id === selectedMicroId) || selectedMicroSegments[0] || null
  }, [selectedMicroId, selectedMicroSegments])
  const activeEvents = useMemo(() => eventsForSegment(selectedSegment, interactionEvents), [interactionEvents, selectedSegment])
  const activeKeyframes = useMemo(() => keyframesForSegment(selectedSegment, interactionKeyframes), [interactionKeyframes, selectedSegment])
  const availableObjects = useMemo(() => uniqueCleanList([
    ...microSegments.map(item => item.primary_object || item.interaction?.primary_object),
    ...segments.flatMap(segment => segment.text_description?.objects || []),
    ...interactionEvents.map(event => event.object_name || event.object_label),
  ].filter(Boolean), 20), [interactionEvents, microSegments, segments])
  const availableInteractions = useMemo(() => uniqueCleanList([
    ...microSegments.map(item => item.interaction_type || item.interaction?.interaction_type),
    ...interactionEvents.map(event => event.interaction || event.event_type),
  ].filter(Boolean), 20), [interactionEvents, microSegments])

  useEffect(() => {
    if (selectedMicroId && selectedMicroSegments.some(item => item.micro_segment_id === selectedMicroId)) return
    setSelectedMicroId(selectedMicroSegments[0]?.micro_segment_id || null)
  }, [selectedMicroId, selectedMicroSegments])

  async function runQuery(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!id || !query.trim()) return
    setQueryLoading(true)
    setQueryError(null)
    try {
      const data = await experimentApi.queryKeyActions(id, query.trim(), 5, {
        indexLevel,
        primaryObject: primaryObject.trim() || undefined,
        interactionType: interactionType.trim() || undefined,
      })
      const response = data as { results?: QueryRow[]; validation_summary?: Record<string, unknown> }
      setQueryResults(response.results || [])
      setQueryValidation(response.validation_summary || null)
    } catch (exc) {
      setQueryError(readError(exc))
    } finally {
      setQueryLoading(false)
    }
  }

  function focusQueryResult(row: QueryRow) {
    const microId = queryResultMicroId(row)
    const micro = microId ? microSegments.find(item => item.micro_segment_id === microId) : null
    const segmentId = queryResultSegmentId(row) || micro?.parent_segment_id || null
    if (segmentId) setSelectedSegmentId(segmentId)
    if (microId) setSelectedMicroId(microId)
  }

  const totalDuration = segments.reduce((sum, segment) => sum + Number(segment.duration_sec || 0), 0)
  const strongEvidenceCount = microSegments.filter(item => evidenceTone(item.evidence_level) === 'emerald').length
  const pendingEvidenceCount = microSegments.filter(item => String(item.evidence_level || '').toLowerCase().includes('insufficient')).length
  const config = detectorConfig(results)
  const formalReport = asRecord(results?.formal_report)
  const reportUrl = experimentFileUrl(String(results?.formal_report_url || formalReport?.url || results?.debug?.formal_report_url || results?.debug?.report || '') || undefined, id)
  const currentStatus = statusLabel(status?.status)

  return (
    <div className="space-y-5">
      <PageHero
        eyebrow={(
          <span>
            <Link to="/experiments" className="hover:text-slate-900">实验队列</Link>
            <span className="mx-2 text-slate-300">/</span>
            <span>关键动作索引</span>
          </span>
        )}
        title="关键动作证据检索"
        description="把 YOLO 检出的动作片段、手物交互、微片段和双视角剪辑组织成可审阅、可检索的证据工作台。"
        actions={(
          <>
            {id && (
              <Link to={`/experiments/${id}/workspace`} className={secondaryButtonClass()}>
                <ArrowLeft className="h-4 w-4" />
                工作台
              </Link>
            )}
            {id && reportUrl && (
              <a href={reportUrl} target="_blank" rel="noreferrer" className={primaryButtonClass('blue')}>
                <FileText className="h-4 w-4" />
                正式报告
              </a>
            )}
          </>
        )}
        tabs={id ? <ExperimentTabs experimentId={id} /> : null}
      />

      <ProgressStrip status={status?.status || 'not_started'} progress={Number(status?.progress ?? 0)} message={`${currentStatus}${status?.message ? ` · ${status.message}` : ''}`} />

      {error && (
        <EvidenceCard className="border-red-200 bg-red-50 p-4 text-sm text-red-700">
          <div className="flex items-center gap-2 font-bold">
            <AlertCircle className="h-4 w-4" />
            加载关键动作结果失败
          </div>
          <div className="mt-1">{error}</div>
        </EvidenceCard>
      )}

      {loading && !results && !error && (
        <EvidenceCard className="flex items-center gap-3 p-5 text-sm font-semibold text-slate-600">
          <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
          正在读取关键动作状态...
        </EvidenceCard>
      )}

      {!loading && !results && !error && (
        <StatusPanel status={status} experimentId={id || ''} />
      )}

      {results && (
        <>
          <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
            <MetricTile label="关键片段" value={segments.length} helper={`${formatNumber(totalDuration, 1)}s total`} tone="blue" Icon={ListChecks} />
            <MetricTile label="微片段" value={microSegments.length} helper={`${pendingEvidenceCount} pending review`} tone="violet" Icon={Layers3} />
            <MetricTile label="手物交互" value={interactionEvents.length} helper={`${interactionKeyframes.length} keyframes`} tone="emerald" Icon={Zap} />
            <MetricTile label="向量条目" value={(results.vector_metadata?.length || 0) + (results.micro_vector_metadata?.length || 0)} helper="segment + micro" tone="cyan" Icon={DatabaseZap} />
            <MetricTile label="强证据" value={strongEvidenceCount} helper={String(results.metric_mode || 'evidence mode')} tone="emerald" Icon={BadgeCheck} />
          </section>

          <EvidenceCard className="p-4">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex items-center gap-3">
                <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-900 text-white">
                  <SlidersHorizontal className="h-4 w-4" />
                </span>
                <div>
                  <div className="font-black text-slate-950">显示密度</div>
                  <div className="text-sm font-medium text-slate-500">默认聚焦证据，高级模式保留检测阈值、诊断资产和索引文本。</div>
                </div>
              </div>
              <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-1">
                <button type="button" onClick={() => setViewMode('simple')} className={`rounded-md px-4 py-2 text-sm font-bold ${viewMode === 'simple' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-600 hover:bg-white'}`}>简洁</button>
                <button type="button" onClick={() => setViewMode('advanced')} className={`rounded-md px-4 py-2 text-sm font-bold ${viewMode === 'advanced' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-600 hover:bg-white'}`}>高级</button>
              </div>
            </div>
          </EvidenceCard>

          {segments.length === 0 ? (
            <EmptyEvidence title="未检出关键动作片段" description="当前结果没有 segment 数据。请先确认双视角视频、ROI 和 YOLO 帧扫描是否进入关键动作流程。" />
          ) : (
            <div className="grid gap-5 xl:grid-cols-[20rem_minmax(0,1fr)_24rem]">
              <SegmentRail
                segments={segments}
                microSegments={microSegments}
                selectedSegmentId={selectedSegment?.segment_id || null}
                selectedMicroId={selectedMicro?.micro_segment_id || null}
                onSelectSegment={setSelectedSegmentId}
                onSelectMicro={setSelectedMicroId}
              />

              <main className="space-y-5">
                {selectedSegment && (
                  <>
                    <EvidenceCard className="overflow-hidden">
                      <div className="border-b border-slate-100 p-4">
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                          <div>
                            <div className="text-xs font-bold uppercase tracking-wide text-slate-500">Selected Segment</div>
                            <h3 className="mt-1 text-xl font-black text-slate-950">{segmentTitle(selectedSegment, segments.indexOf(selectedSegment))}</h3>
                          </div>
                          <EvidenceBadge tone={toneForStatus(status?.status)}>{segmentTimeRange(selectedSegment)}</EvidenceBadge>
                        </div>
                      </div>
                      <div className="grid gap-4 p-4 lg:grid-cols-2">
                        <VideoPanel title="第三人称（俯视桌面）" clip={selectedSegment.third_person} experimentId={id} />
                        <VideoPanel title="第一人称（操作者视角）" clip={selectedSegment.first_person} experimentId={id} />
                      </div>
                    </EvidenceCard>

                    <EvidenceCard className="p-4">
                      <div className="mb-4 flex items-center justify-between gap-3">
                        <div>
                          <h3 className="font-black text-slate-950">物理交互证据</h3>
                          <p className="mt-1 text-sm font-medium text-slate-500">YOLO 标签、手物交互事件和关键帧用于支撑片段检索，不把框图当最终交付物。</p>
                        </div>
                        <EvidenceBadge tone={activeEvents.length ? 'emerald' : 'amber'}>{activeEvents.length} events</EvidenceBadge>
                      </div>
                      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_17rem]">
                        <div className="space-y-4">
                          <YoloLabels entries={yoloLabelEntries(selectedSegment)} />
                          <InteractionList events={activeEvents} />
                        </div>
                        <KeyframeGrid keyframes={activeKeyframes} maxItems={viewMode === 'advanced' ? 24 : 9} experimentId={id} />
                      </div>
                    </EvidenceCard>

                    <MicroSegmentPanel
                      microSegments={selectedMicroSegments}
                      selectedMicroId={selectedMicro?.micro_segment_id || null}
                      onSelectMicro={setSelectedMicroId}
                      advanced={viewMode === 'advanced'}
                      experimentId={id}
                    />
                  </>
                )}
              </main>

              <aside className="space-y-5">
                <QueryPanel
                  query={query}
                  setQuery={setQuery}
                  indexLevel={indexLevel}
                  setIndexLevel={setIndexLevel}
                  primaryObject={primaryObject}
                  setPrimaryObject={setPrimaryObject}
                  interactionType={interactionType}
                  setInteractionType={setInteractionType}
                  availableObjects={availableObjects}
                  availableInteractions={availableInteractions}
                  queryResults={queryResults}
                  queryValidation={queryValidation || asRecord(results.query_validation_summary)}
                  loading={queryLoading}
                  error={queryError}
                  onSubmit={runQuery}
                  onFocusResult={focusQueryResult}
                />
                <SelectionDetail segment={selectedSegment} micro={selectedMicro} advanced={viewMode === 'advanced'} />
                {viewMode === 'advanced' && <DetectorPanel results={results} config={config} experimentId={id} />}
              </aside>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function ExperimentTabs({ experimentId }: { experimentId: string }) {
  const tabClass = 'rounded-md px-3 py-1.5 text-sm font-bold text-slate-600 transition hover:bg-slate-100 hover:text-slate-950'
  return (
    <nav className="flex flex-wrap gap-2">
      <Link to={`/experiments/${experimentId}/workspace`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'workspace')} onFocus={() => prefetchExperimentRoute(experimentId, 'workspace')} className={tabClass}>分析概览</Link>
      <Link to={`/experiments/${experimentId}/report`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'report')} onFocus={() => prefetchExperimentRoute(experimentId, 'report')} className={tabClass}>分析报告</Link>
      <Link to={`/experiments/${experimentId}/materials`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'materials')} onFocus={() => prefetchExperimentRoute(experimentId, 'materials')} className={tabClass}>关键素材库</Link>
      <Link to={`/experiments/${experimentId}/key-actions`} className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-bold text-white">关键动作</Link>
      <Link to={`/experiments/${experimentId}/key-actions/review`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'reviewQueue')} onFocus={() => prefetchExperimentRoute(experimentId, 'reviewQueue')} className={tabClass}>Review Queue</Link>
    </nav>
  )
}

function StatusPanel({ status, experimentId }: { status: KeyActionStatus | null; experimentId: string }) {
  const value = String(status?.status || 'not_started').toLowerCase()
  const isFailed = value === 'failed'
  return (
    <EvidenceCard className="p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <EvidenceBadge tone={isFailed ? 'red' : 'amber'}>{statusLabel(status?.status)}</EvidenceBadge>
          <h3 className="mt-3 text-xl font-black text-slate-950">{isFailed ? '关键动作流程失败' : '关键动作索引尚未就绪'}</h3>
          <p className="mt-2 max-w-2xl text-sm font-medium leading-6 text-slate-500">
            关键动作页面依赖双视角视频、YOLO 片段、手物交互证据和微片段索引。完成上传或重新触发关键动作流程后，这里会自动轮询并显示检索工作台。
          </p>
          {status?.error && <div className="mt-3 rounded-lg bg-red-50 p-3 text-sm font-semibold text-red-700">{status.error}</div>}
        </div>
        {experimentId && (
          <Link to={`/experiments/${experimentId}/workspace`} className={primaryButtonClass(isFailed ? 'red' : 'blue')}>
            <Video className="h-4 w-4" />
            返回工作台
          </Link>
        )}
      </div>
    </EvidenceCard>
  )
}

function SegmentRail({
  segments,
  microSegments,
  selectedSegmentId,
  selectedMicroId,
  onSelectSegment,
  onSelectMicro,
}: {
  segments: KeyActionSegment[]
  microSegments: KeyActionMicroSegment[]
  selectedSegmentId: string | null
  selectedMicroId: string | null
  onSelectSegment: (id: string) => void
  onSelectMicro: (id: string) => void
}) {
  return (
    <aside className="space-y-3">
      <EvidenceCard className="p-3">
        <div className="mb-3 flex items-center justify-between px-1">
          <div className="font-black text-slate-950">片段队列</div>
          <EvidenceBadge tone="blue">{segments.length}</EvidenceBadge>
        </div>
        <div className="max-h-[72vh] space-y-2 overflow-auto pr-1">
          {segments.map((segment, index) => {
            const active = segment.segment_id === selectedSegmentId
            const micros = microSegmentsForSegment(segment, microSegments)
            return (
              <div key={segment.segment_id} className="space-y-1">
                <button
                  type="button"
                  onClick={() => onSelectSegment(segment.segment_id)}
                  className={`w-full rounded-lg border p-3 text-left transition ${active ? 'border-blue-300 bg-blue-50 shadow-sm' : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-black text-slate-500">S{index + 1}</span>
                    <span className="font-mono text-xs font-bold text-slate-500">{formatNumber(segment.duration_sec, 1)}s</span>
                  </div>
                  <div className="mt-1 line-clamp-2 text-sm font-black text-slate-950">{segmentTitle(segment, index)}</div>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {uniqueCleanList([...(segment.text_description?.objects || []), ...Object.keys(segment.yolo_label_counts || {})], 3).map(item => (
                      <span key={item} className="rounded bg-white px-1.5 py-0.5 text-xs font-semibold text-slate-600 ring-1 ring-slate-200">{item}</span>
                    ))}
                  </div>
                </button>
                {active && micros.length > 0 && (
                  <div className="ml-3 space-y-1 border-l border-slate-200 pl-2">
                    {micros.slice(0, 10).map((micro, microIndex) => (
                      <button
                        type="button"
                        key={micro.micro_segment_id}
                        onClick={() => onSelectMicro(micro.micro_segment_id)}
                        className={`w-full rounded-md px-2 py-1.5 text-left text-xs font-semibold transition ${micro.micro_segment_id === selectedMicroId ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'}`}
                      >
                        <span className="mr-1 font-mono">M{microIndex + 1}</span>
                        {microTitle(micro, microIndex)}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </EvidenceCard>
    </aside>
  )
}

function VideoPanel({ title, clip, experimentId }: { title: string; clip?: KeyActionClipRef | null; experimentId?: string | null }) {
  const url = clipUrl(clip, experimentId)
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-black text-slate-800">
          <PlayCircle className="h-4 w-4 text-blue-600" />
          {title}
        </div>
        <EvidenceBadge tone={clip?.annotated_clip_url ? 'emerald' : 'slate'}>{clipMode(clip)}</EvidenceBadge>
      </div>
      {url ? (
        <video src={url} className="aspect-video w-full rounded-lg bg-slate-950 object-contain" controls playsInline preload="metadata" />
      ) : (
        <div className="flex aspect-video items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 text-sm font-semibold text-slate-400">
          暂无可播放剪辑
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 text-xs font-semibold text-slate-500">
        <div className="rounded-md bg-slate-50 px-2 py-1">local {formatNumber(clip?.local_start_sec, 2)}-{formatNumber(clip?.local_end_sec, 2)}s</div>
        <div className="rounded-md bg-slate-50 px-2 py-1">YOLO {formatCount(clip?.yolo_detection_count)} boxes</div>
      </div>
    </div>
  )
}

function YoloLabels({ entries }: { entries: Array<[string, number | null]> }) {
  if (!entries.length) return <EmptyEvidence title="暂无 YOLO 标签" description="该片段没有可展示的检测标签。" />
  return (
    <div>
      <div className="mb-2 text-xs font-black uppercase tracking-wide text-slate-500">YOLO labels</div>
      <div className="flex flex-wrap gap-2">
        {entries.slice(0, 18).map(([label, count]) => (
          <span key={label} className="rounded-md bg-slate-100 px-2 py-1 text-xs font-bold text-slate-700">
            {cleanDisplayText(label.replace(/_/g, ' '))}
            {count != null && <span className="ml-1 font-mono text-slate-500">{count}</span>}
          </span>
        ))}
      </div>
    </div>
  )
}

function InteractionList({ events }: { events: KeyActionInteractionEvent[] }) {
  if (!events.length) return <EmptyEvidence title="暂无手物交互事件" description="当前片段未生成接触、拿取、释放等物理交互证据。" />
  return (
    <div className="space-y-2">
      <div className="text-xs font-black uppercase tracking-wide text-slate-500">Interaction events</div>
      {events.slice(0, 8).map(event => (
        <div key={event.event_id} className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="truncate text-sm font-black text-slate-950">{interactionLabel(event)}</div>
              <div className="mt-1 text-xs font-semibold text-slate-500">{interactionObjectText(event)}</div>
            </div>
            <EvidenceBadge tone={evidenceTone(event.evidence_grade || event.review_status)}>{formatNumber(event.confidence, 2)}</EvidenceBadge>
          </div>
          <div className="mt-2 flex flex-wrap gap-2 text-xs font-semibold text-slate-500">
            <span className="rounded bg-slate-50 px-2 py-1">{formatNumber(event.start_time_sec ?? event.local_time_sec, 2)}s</span>
            {event.view && <span className="rounded bg-slate-50 px-2 py-1">{event.view}</span>}
            {event.review_status && <span className="rounded bg-slate-50 px-2 py-1">{event.review_status}</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

function KeyframeGrid({ keyframes, maxItems, experimentId }: { keyframes: KeyActionInteractionKeyframe[]; maxItems: number; experimentId?: string | null }) {
  if (!keyframes.length) return <EmptyEvidence title="暂无关键帧" description="没有手物交互关键帧缩略图。" />
  const visible = keyframes.slice(0, maxItems)
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs font-black uppercase tracking-wide text-slate-500">Keyframes</div>
        <span className="text-xs font-bold text-slate-400">{visible.length}/{keyframes.length}</span>
      </div>
        <div className="grid grid-cols-2 gap-2">
          {visible.map((frame, index) => {
          const url = keyframeUrl(frame, experimentId)
          return (
            <a key={`${frame.event_id}-${index}`} href={url || undefined} target="_blank" rel="noreferrer" className="group overflow-hidden rounded-lg border border-slate-200 bg-slate-50">
              {url ? (
                <img src={url} alt={interactionLabel(frame)} className="aspect-video w-full object-cover transition group-hover:scale-[1.02]" />
              ) : (
                <div className="flex aspect-video items-center justify-center text-xs text-slate-400">no image</div>
              )}
              <div className="truncate px-2 py-1 text-xs font-semibold text-slate-500">{formatNumber(frame.timestamp_sec ?? frame.local_time_sec, 2)}s</div>
            </a>
          )
        })}
      </div>
    </div>
  )
}

function MicroSegmentPanel({
  microSegments,
  selectedMicroId,
  onSelectMicro,
  advanced,
  experimentId,
}: {
  microSegments: KeyActionMicroSegment[]
  selectedMicroId: string | null
  onSelectMicro: (id: string) => void
  advanced: boolean
  experimentId?: string | null
}) {
  if (!microSegments.length) return <EvidenceCard className="p-4"><EmptyEvidence title="该片段尚无微片段" description="微片段是后续物体级检索和审阅的最小证据单位。" /></EvidenceCard>
  return (
    <EvidenceCard className="p-4">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h3 className="font-black text-slate-950">微片段证据</h3>
          <p className="mt-1 text-sm font-medium text-slate-500">按物体、交互类型、置信度和关键帧组织，支持 segment 与 micro-level 检索。</p>
        </div>
        <EvidenceBadge tone="violet">{microSegments.length} items</EvidenceBadge>
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        {microSegments.slice(0, advanced ? 24 : 8).map((micro, index) => {
          const selected = micro.micro_segment_id === selectedMicroId
          const urls = microKeyframeUrls(micro, experimentId).slice(0, 3)
          return (
            <button
              type="button"
              key={micro.micro_segment_id}
              onClick={() => onSelectMicro(micro.micro_segment_id)}
              className={`rounded-lg border p-3 text-left transition ${selected ? 'border-blue-300 bg-blue-50 shadow-sm' : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'}`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-xs font-black text-slate-500">M{index + 1} · {microTimeRange(micro)}</div>
                  <div className="mt-1 line-clamp-2 text-sm font-black text-slate-950">{microTitle(micro, index)}</div>
                </div>
                <EvidenceBadge tone={confidenceTone(micro.confidence, micro.max_interaction_score)}>{micro.confidence || formatNumber(micro.max_interaction_score, 2)}</EvidenceBadge>
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {[micro.primary_object_family, micro.primary_object, micro.interaction_type].filter(Boolean).map(item => (
                  <span key={String(item)} className="rounded bg-white px-1.5 py-0.5 text-xs font-semibold text-slate-600 ring-1 ring-slate-200">{cleanDisplayText(String(item).replace(/_/g, ' '))}</span>
                ))}
              </div>
              {urls.length > 0 && (
                <div className="mt-3 grid grid-cols-3 gap-1.5">
                  {urls.map(url => (
                    <img key={url} src={url} alt="micro evidence" className="aspect-video rounded-md bg-slate-100 object-cover" />
                  ))}
                </div>
              )}
              {advanced && micro.evidence_reasons?.length ? (
                <div className="mt-2 line-clamp-2 text-xs font-medium text-slate-500">{micro.evidence_reasons.join(' · ')}</div>
              ) : null}
            </button>
          )
        })}
      </div>
    </EvidenceCard>
  )
}

function QueryPanel({
  query,
  setQuery,
  indexLevel,
  setIndexLevel,
  primaryObject,
  setPrimaryObject,
  interactionType,
  setInteractionType,
  availableObjects,
  availableInteractions,
  queryResults,
  queryValidation,
  loading,
  error,
  onSubmit,
  onFocusResult,
}: {
  query: string
  setQuery: (value: string) => void
  indexLevel: QueryIndexLevel
  setIndexLevel: (value: QueryIndexLevel) => void
  primaryObject: string
  setPrimaryObject: (value: string) => void
  interactionType: string
  setInteractionType: (value: string) => void
  availableObjects: string[]
  availableInteractions: string[]
  queryResults: QueryRow[]
  queryValidation: Record<string, unknown> | null
  loading: boolean
  error: string | null
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  onFocusResult: (row: QueryRow) => void
}) {
  return (
    <EvidenceCard className="p-4">
      <div className="mb-4 flex items-center gap-2">
        <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-50 text-blue-700 ring-1 ring-blue-200">
          <Search className="h-4 w-4" />
        </span>
        <div>
          <h3 className="font-black text-slate-950">证据检索</h3>
          <div className="text-xs font-semibold text-slate-500">segment / micro-segment 混合查询</div>
        </div>
      </div>
      <form onSubmit={onSubmit} className="space-y-3">
        <textarea
          value={query}
          onChange={event => setQuery(event.target.value)}
          rows={4}
          className="w-full resize-none rounded-lg border border-slate-200 bg-white p-3 text-sm font-medium text-slate-800 outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
        />
        <div className="grid gap-2 sm:grid-cols-3">
          <select value={indexLevel} onChange={event => setIndexLevel(event.target.value as QueryIndexLevel)} className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-bold text-slate-700">
            <option value="all">全部索引</option>
            <option value="segment">片段</option>
            <option value="micro_segment">微片段</option>
          </select>
          <select value={primaryObject} onChange={event => setPrimaryObject(event.target.value)} className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-bold text-slate-700">
            <option value="">任意物体</option>
            {availableObjects.map(item => <option key={item} value={item}>{item}</option>)}
          </select>
          <select value={interactionType} onChange={event => setInteractionType(event.target.value)} className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-bold text-slate-700">
            <option value="">任意交互</option>
            {availableInteractions.map(item => <option key={item} value={item}>{item}</option>)}
          </select>
        </div>
        <button type="submit" disabled={loading || !query.trim()} className={`${primaryButtonClass('blue')} w-full disabled:cursor-not-allowed disabled:opacity-60`}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
          检索证据
        </button>
      </form>
      {error && <div className="mt-3 rounded-lg bg-red-50 p-3 text-sm font-semibold text-red-700">{error}</div>}
      {queryValidation && (
        <div className="mt-3 rounded-lg bg-slate-50 p-3 text-xs font-semibold text-slate-500">
          validation: {recordString(queryValidation, 'status', 'available')}
        </div>
      )}
      <div className="mt-4 space-y-2">
        {queryResults.length === 0 ? (
          <EmptyEvidence title="暂无检索命中" description="输入自然语言问题后，会返回最相关的片段或微片段。" />
        ) : queryResults.map((row, index) => (
          <div key={`${queryResultTitle(row, index)}-${index}`} className="rounded-lg border border-slate-200 bg-white p-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="truncate text-sm font-black text-slate-950">{queryResultTitle(row, index)}</div>
                <div className="mt-1 line-clamp-3 text-xs font-medium leading-5 text-slate-500">{queryResultText(row)}</div>
              </div>
              <EvidenceBadge tone={index === 0 ? 'emerald' : 'blue'}>{queryResultScore(row)}</EvidenceBadge>
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5 text-xs font-semibold text-slate-500">
              <span className="rounded bg-slate-50 px-2 py-1">{queryResultLevel(row)}</span>
              {queryResultSegmentId(row) && <span className="rounded bg-slate-50 px-2 py-1">{queryResultSegmentId(row)}</span>}
              {queryResultMicroId(row) && <span className="rounded bg-slate-50 px-2 py-1">{queryResultMicroId(row)}</span>}
              {(queryResultSegmentId(row) || queryResultMicroId(row)) && (
                <button type="button" onClick={() => onFocusResult(row)} className="rounded bg-blue-50 px-2 py-1 font-black text-blue-700 hover:bg-blue-100">
                  定位证据
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </EvidenceCard>
  )
}

function SelectionDetail({ segment, micro, advanced }: { segment: KeyActionSegment | null; micro: KeyActionMicroSegment | null; advanced: boolean }) {
  return (
    <EvidenceCard className="p-4">
      <div className="mb-3 flex items-center gap-2">
        <Boxes className="h-4 w-4 text-slate-500" />
        <h3 className="font-black text-slate-950">当前证据</h3>
      </div>
      {!segment ? (
        <EmptyEvidence title="未选择片段" />
      ) : (
        <div className="space-y-3 text-sm">
          <DetailRow label="segment_id" value={segment.segment_id} />
          <DetailRow label="时间" value={segmentTimeRange(segment)} />
          <DetailRow label="动作" value={friendlyActionName(segment.text_description?.action_type)} />
          <DetailRow label="motion" value={formatNumber(segment.cv_detection?.avg_motion_score, 3)} />
          <DetailRow label="active" value={formatNumber(segment.cv_detection?.avg_active_score, 3)} />
          {micro && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="mb-2 text-xs font-black uppercase tracking-wide text-slate-500">Selected micro</div>
              <DetailRow label="micro_id" value={micro.micro_segment_id} />
              <DetailRow label="时间" value={microTimeRange(micro)} />
              <DetailRow label="物体" value={cleanDisplayText(micro.primary_object || micro.interaction?.primary_object || '-', '-')} />
              <DetailRow label="交互" value={cleanDisplayText(micro.interaction_type || micro.interaction?.interaction_type || '-', '-')} />
              <DetailRow label="证据级别" value={String(micro.evidence_level || micro.confidence || '-')} />
            </div>
          )}
          {advanced && (
            <div className="rounded-lg bg-slate-950 p-3 text-xs font-medium leading-5 text-slate-200">
              {segment.index?.index_text || micro?.text_description?.index_text || '暂无索引文本。'}
            </div>
          )}
        </div>
      )}
    </EvidenceCard>
  )
}

function DetectorPanel({ results, config, experimentId }: { results: KeyActionResults; config: KeyActionDetectorConfig | null; experimentId?: string | null }) {
  const scan = results.yolo_frame_scan
  const debugLinks = [
    ['ROI preview', results.debug?.roi_preview],
    ['Frame scores', results.debug?.frame_score_plot || results.debug?.frame_scores],
    ['Contact sheet', results.debug?.segments_contact_sheet],
    ['Debug report', results.debug?.report],
  ].filter(([, url]) => Boolean(url))
  return (
    <EvidenceCard className="p-4">
      <div className="mb-3 flex items-center gap-2">
        <Gauge className="h-4 w-4 text-slate-500" />
        <h3 className="font-black text-slate-950">检测诊断</h3>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs font-semibold text-slate-600">
        <DetailBox label="sample_fps" value={formatNumber(config?.sample_fps, 2)} />
        <DetailBox label="start" value={formatNumber(config?.start_threshold, 2)} />
        <DetailBox label="end" value={formatNumber(config?.end_threshold, 2)} />
        <DetailBox label="merge_gap" value={`${formatNumber(config?.merge_gap_sec, 1)}s`} />
        <DetailBox label="frames" value={formatCount(scan?.detection_frame_count)} />
        <DetailBox label="events" value={formatCount(scan?.physical_event_count)} />
      </div>
      {debugLinks.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {debugLinks.map(([label, url]) => (
            <a key={label} href={experimentFileUrl(String(url), experimentId)} target="_blank" rel="noreferrer" className={secondaryButtonClass('slate')}>
              {label}
            </a>
          ))}
        </div>
      )}
    </EvidenceCard>
  )
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2 last:border-0 last:pb-0">
      <span className="shrink-0 text-xs font-black uppercase tracking-wide text-slate-400">{label}</span>
      <span className="min-w-0 break-words text-right font-bold text-slate-700">{value}</span>
    </div>
  )
}

function DetailBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-50 p-2">
      <div className="text-slate-400">{label}</div>
      <div className="mt-1 font-mono font-black text-slate-800">{value}</div>
    </div>
  )
}
