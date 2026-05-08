import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  AlertTriangle,
  ArrowLeft,
  BadgeCheck,
  CheckCircle2,
  ClipboardCheck,
  Download,
  FileJson,
  Gauge,
  Keyboard,
  Layers3,
  Loader2,
  RefreshCw,
  RotateCcw,
  SearchCheck,
  SlidersHorizontal,
  SplitSquareHorizontal,
  ZoomIn,
  ZoomOut,
  XCircle,
} from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import {
  EmptyEvidence,
  EvidenceBadge,
  EvidenceCard,
  MetricTile,
  PageHero,
  primaryButtonClass,
  secondaryButtonClass,
  toneForStatus,
} from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import type { KeyActionEvidenceAdapters, KeyActionReviewItem, KeyActionReviewQueue } from '../types'

type ReviewFilter = 'all' | 'pending' | 'approved' | 'rejected' | 'needs_review'
type TypeFilter = 'all' | 'qa_warning' | 'evidence_semantic' | 'segment' | 'micro_segment' | 'material_candidate'

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}

function recordString(record: Record<string, unknown> | null | undefined, key: string, fallback = '') {
  const value = record?.[key]
  return value == null || value === '' ? fallback : String(value)
}

function readError(error: unknown) {
  const record = asRecord(error)
  const response = asRecord(record?.response)
  const data = asRecord(response?.data)
  return recordString(data, 'detail') || recordString(record, 'message', 'request failed')
}

function formatNumber(value: unknown, digits = 2) {
  if (value === null || value === undefined || value === '') return '-'
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue.toFixed(digits) : '-'
}

function numberValue(value: unknown, fallback = 0) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function percent(value: unknown) {
  const numberValue = Number(value)
  if (!Number.isFinite(numberValue)) return '-'
  return `${Math.round(numberValue * 100)}%`
}

function statusTone(status?: string | null) {
  const value = String(status || '').toLowerCase()
  if (value === 'approved') return 'emerald'
  if (value === 'rejected') return 'red'
  if (value === 'needs_review' || value === 'pending') return 'amber'
  return toneForStatus(value)
}

function typeLabel(value?: string | null) {
  const key = String(value || '')
  if (key === 'qa_warning') return 'QA Warning'
  if (key === 'evidence_semantic') return 'Evidence'
  if (key === 'segment') return 'Segment'
  if (key === 'micro_segment') return 'Micro'
  if (key === 'material_candidate') return 'Material'
  return key || 'Item'
}

function severityTone(value?: string | null) {
  if (value === 'error') return 'red'
  if (value === 'warning') return 'amber'
  return 'slate'
}

function itemTime(item: KeyActionReviewItem) {
  if (item.start_sec == null && item.end_sec == null) return '-'
  return `${formatNumber(item.adjusted_start_sec ?? item.start_sec, 2)}-${formatNumber(item.adjusted_end_sec ?? item.end_sec, 2)}s`
}

function downloadJson(payload: unknown, name: string) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = name
  anchor.click()
  URL.revokeObjectURL(url)
}

export default function KeyActionReviewQueue() {
  const { id } = useParams<{ id: string }>()
  const [queue, setQueue] = useState<KeyActionReviewQueue | null>(null)
  const [adapters, setAdapters] = useState<KeyActionEvidenceAdapters | null>(null)
  const [retrievalEval, setRetrievalEval] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)
  const [savingId, setSavingId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<ReviewFilter>('pending')
  const [typeFilter, setTypeFilter] = useState<TypeFilter>('all')
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set())
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [boundaries, setBoundaries] = useState<Record<string, { start?: string; end?: string }>>({})

  async function load(force = false) {
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const [nextQueue, nextAdapters] = await Promise.all([
        experimentApi.getKeyActionReviewQueue(id, { force }),
        experimentApi.getKeyActionEvidenceAdapters(id, { force }),
      ])
      setQueue(nextQueue)
      setAdapters(nextAdapters)
      setSelectedIds(previous => {
        const available = new Set((nextQueue.items || []).map(item => item.item_id))
        return new Set(Array.from(previous).filter(item => available.has(item)))
      })
    } catch (exc) {
      setError(readError(exc))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load(true)
  }, [id])

  const items = queue?.items || []
  const filteredItems = useMemo(() => {
    return items.filter(item => {
      const status = String(item.review_status || 'pending') as ReviewFilter
      const type = String(item.item_type || 'all') as TypeFilter
      return (statusFilter === 'all' || status === statusFilter) && (typeFilter === 'all' || type === typeFilter)
    })
  }, [items, statusFilter, typeFilter])

  const selectedItems = items.filter(item => selectedIds.has(item.item_id))
  const quality = queue?.quality
  const metrics = quality?.core_metrics || {}
  const adapterCounts = adapters?.counts || {}

  function toggleSelected(itemId: string) {
    setSelectedIds(previous => {
      const next = new Set(previous)
      if (next.has(itemId)) next.delete(itemId)
      else next.add(itemId)
      return next
    })
  }

  function replaceSelected(itemIds: string[]) {
    setSelectedIds(new Set(itemIds))
  }

  async function decide(item: KeyActionReviewItem, decision: string) {
    if (!id) return
    setSavingId(item.item_id)
    try {
      const boundary = boundaries[item.item_id] || {}
      const response = await experimentApi.decideKeyActionReviewItem(id, item.item_id, {
        decision,
        reviewer: 'frontend_reviewer',
        note: notes[item.item_id] || '',
        boundary_start_sec: boundary.start ? Number(boundary.start) : undefined,
        boundary_end_sec: boundary.end ? Number(boundary.end) : undefined,
      })
      setQueue(response.queue)
    } catch (exc) {
      setError(readError(exc))
    } finally {
      setSavingId(null)
    }
  }

  async function bulkDecision(decision: string, onlySelected: boolean) {
    if (!id) return
    setSavingId('bulk')
    try {
      const response = await experimentApi.bulkDecideKeyActionReviewItems(id, {
        item_ids: onlySelected ? selectedItems.map(item => item.item_id) : undefined,
        decision,
        reviewer: 'frontend_reviewer',
        note: onlySelected ? 'bulk decision from Review Queue selection' : 'bulk decision for pending Review Queue items',
      })
      setQueue(response.queue)
      setSelectedIds(new Set())
    } catch (exc) {
      setError(readError(exc))
    } finally {
      setSavingId(null)
    }
  }

  async function exportQueue() {
    if (!id) return
    setSavingId('export')
    try {
      const payload = await experimentApi.exportKeyActionReview(id)
      downloadJson(payload, `${id}-key-action-review-export.json`)
    } catch (exc) {
      setError(readError(exc))
    } finally {
      setSavingId(null)
    }
  }

  async function freezeReviewedDataset() {
    if (!id) return
    setSavingId('freeze')
    try {
      const payload = await experimentApi.freezeKeyActionReviewedDataset(id)
      setQueue(payload.queue)
      downloadJson(payload.reviewed_export || payload.manifest, `${id}-reviewed-dataset.json`)
    } catch (exc) {
      setError(readError(exc))
    } finally {
      setSavingId(null)
    }
  }

  async function rollbackRelease() {
    if (!id) return
    setSavingId('rollback')
    try {
      const payload = await experimentApi.rollbackKeyActionReviewedRelease(id)
      setQueue(payload.queue)
      downloadJson(payload.reviewed_export || payload.rollback, `${id}-reviewed-release-rollback.json`)
    } catch (exc) {
      setError(readError(exc))
    } finally {
      setSavingId(null)
    }
  }

  async function promoteReviewedRelease() {
    if (!id) return
    const reviewer = window.prompt('Reviewer identity for promotion audit')
    if (!reviewer?.trim()) return
    const note = window.prompt('Promotion note')?.trim() || 'Promotion requested from Review Queue after backend gates pass.'
    setSavingId('promote')
    try {
      const payload = await experimentApi.promoteKeyActionReviewedRelease(id, {
        reviewer: reviewer.trim(),
        note,
        query_count: 50,
      })
      setQueue(payload.queue)
      downloadJson(payload.promotion, `${id}-promoted-reviewed-release.json`)
    } catch (exc) {
      setError(readError(exc))
    } finally {
      setSavingId(null)
    }
  }

  async function runRetrievalEval() {
    if (!id) return
    setSavingId('retrieval')
    try {
      const payload = await experimentApi.evaluateKeyActionRetrieval(id, 50)
      setRetrievalEval(payload.evaluation)
    } catch (exc) {
      setError(readError(exc))
    } finally {
      setSavingId(null)
    }
  }

  return (
    <div className="space-y-5">
      <PageHero
        eyebrow={(
          <span>
            <Link to="/experiments" className="hover:text-slate-900">Experiments</Link>
            <span className="mx-2 text-slate-300">/</span>
            <span>Key Action Review Queue</span>
          </span>
        )}
        title="Key Action 审核闭环"
        description="把 QA warning、低置信 segment、候选 micro、未确认素材和检索评测集中到一个可确认、可驳回、可导出的审核工作台。"
        actions={(
          <>
            {id && (
              <Link to={`/experiments/${id}/key-actions`} className={secondaryButtonClass()}>
                <ArrowLeft className="h-4 w-4" />
                Key Action
              </Link>
            )}
            <button type="button" onClick={() => void load(true)} className={secondaryButtonClass('blue')}>
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
            <button type="button" onClick={() => void exportQueue()} className={primaryButtonClass('emerald')}>
              {savingId === 'export' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
              Export
            </button>
            <button type="button" onClick={() => void freezeReviewedDataset()} className={primaryButtonClass('blue')}>
              {savingId === 'freeze' ? <Loader2 className="h-4 w-4 animate-spin" /> : <ClipboardCheck className="h-4 w-4" />}
              Freeze
            </button>
            <button type="button" onClick={() => void promoteReviewedRelease()} className={primaryButtonClass('emerald')}>
              {savingId === 'promote' ? <Loader2 className="h-4 w-4 animate-spin" /> : <BadgeCheck className="h-4 w-4" />}
              Promote
            </button>
            <button type="button" onClick={() => void rollbackRelease()} className={secondaryButtonClass('amber')}>
              {savingId === 'rollback' ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCcw className="h-4 w-4" />}
              Rollback
            </button>
          </>
        )}
        tabs={id ? <ExperimentTabs experimentId={id} /> : null}
      />

      {error && (
        <EvidenceCard className="border-red-200 bg-red-50 p-4 text-sm font-semibold text-red-700">
          {error}
        </EvidenceCard>
      )}

      {loading && !queue ? (
        <EvidenceCard className="flex items-center gap-3 p-5 text-sm font-semibold text-slate-600">
          <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
          Loading review queue...
        </EvidenceCard>
      ) : queue ? (
        <>
          <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-6">
            <MetricTile label="Health Score" value={quality?.health_score ?? queue.summary.quality_score ?? '-'} helper={quality?.status || 'quality'} tone={(quality?.health_score || 0) >= 82 ? 'emerald' : 'amber'} Icon={Gauge} />
            <MetricTile label="Pending" value={queue.summary.pending} helper={`${queue.summary.total} total`} tone="amber" Icon={ClipboardCheck} />
            <MetricTile label="Segments" value={metrics.segment_count ?? queue.summary.segment_count ?? 0} helper={`${formatNumber(metrics.longest_segment_sec, 1)}s longest`} tone="blue" Icon={SplitSquareHorizontal} />
            <MetricTile label="Micro" value={metrics.micro_segment_count ?? queue.summary.micro_segment_count ?? 0} helper={`${metrics.unreviewed_count ?? 0} unreviewed`} tone="violet" Icon={Layers3} />
            <MetricTile label="Coverage" value={percent(metrics.total_action_coverage_ratio)} helper={`limit ${percent(asRecord(quality?.coverage_check)?.threshold ?? 0.65)}`} tone={String(asRecord(quality?.coverage_check)?.status) === 'warning' ? 'amber' : 'emerald'} Icon={SlidersHorizontal} />
            <MetricTile label="Vectors" value={metrics.vector_count ?? 0} helper="segment + micro" tone="cyan" Icon={FileJson} />
          </section>

          <EvidenceCard className="p-4">
            <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_24rem]">
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  {(['all', 'pending', 'needs_review', 'approved', 'rejected'] as ReviewFilter[]).map(status => (
                    <button
                      key={status}
                      type="button"
                      onClick={() => setStatusFilter(status)}
                      className={`rounded-lg px-3 py-2 text-sm font-bold ${statusFilter === status ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-700 hover:bg-white'}`}
                    >
                      {status}
                    </button>
                  ))}
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  {(['all', 'qa_warning', 'evidence_semantic', 'segment', 'micro_segment', 'material_candidate'] as TypeFilter[]).map(type => (
                    <button
                      key={type}
                      type="button"
                      onClick={() => setTypeFilter(type)}
                      className={`rounded-lg px-3 py-2 text-sm font-bold ${typeFilter === type ? 'bg-blue-600 text-white' : 'bg-blue-50 text-blue-700 hover:bg-white'}`}
                    >
                      {typeLabel(type)}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex flex-wrap items-start justify-end gap-2">
                <button type="button" onClick={() => void bulkDecision('approved', true)} disabled={!selectedItems.length || savingId === 'bulk'} className={`${secondaryButtonClass('emerald')} disabled:opacity-50`}>
                  <CheckCircle2 className="h-4 w-4" />
                  Approve selected
                </button>
                <button type="button" onClick={() => void bulkDecision('needs_review', true)} disabled={!selectedItems.length || savingId === 'bulk'} className={`${secondaryButtonClass('amber')} disabled:opacity-50`}>
                  <AlertTriangle className="h-4 w-4" />
                  Need review
                </button>
                <button type="button" onClick={() => void bulkDecision('approved', false)} disabled={savingId === 'bulk'} className={primaryButtonClass('blue')}>
                  {savingId === 'bulk' ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
                  Approve pending
                </button>
              </div>
            </div>
          </EvidenceCard>

          <CompactTimeline
            items={items}
            adapters={adapters}
            selectedIds={selectedIds}
            boundaries={boundaries}
            onSelect={toggleSelected}
            onSelectMany={replaceSelected}
            onClearSelection={() => setSelectedIds(new Set())}
            onBoundary={(itemId, value) => setBoundaries(previous => ({ ...previous, [itemId]: { ...(previous[itemId] || {}), ...value } }))}
            onApplySplit={() => void freezeReviewedDataset()}
          />

          <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_25rem]">
            <main className="space-y-3">
              {filteredItems.length === 0 ? (
                <EmptyEvidence title="No review items" description="当前筛选条件下没有待审证据。" />
              ) : filteredItems.map(item => (
                <ReviewItemCard
                  key={item.item_id}
                  item={item}
                  selected={selectedIds.has(item.item_id)}
                  note={notes[item.item_id] || ''}
                  boundary={boundaries[item.item_id] || {}}
                  saving={savingId === item.item_id}
                  onToggle={() => toggleSelected(item.item_id)}
                  onNote={value => setNotes(previous => ({ ...previous, [item.item_id]: value }))}
                  onBoundary={(value) => setBoundaries(previous => ({ ...previous, [item.item_id]: { ...(previous[item.item_id] || {}), ...value } }))}
                  onDecision={decision => void decide(item, decision)}
                />
              ))}
            </main>

            <aside className="space-y-5">
              <QualityPanel queue={queue} />
              <AdapterPanel adapters={adapters} counts={adapterCounts} />
              <RetrievalPanel evaluation={retrievalEval} saving={savingId === 'retrieval'} onRun={() => void runRetrievalEval()} />
            </aside>
          </div>
        </>
      ) : (
        <EmptyEvidence title="Review queue is not ready" description="请先运行 key action 分析。" />
      )}
    </div>
  )
}

function CompactTimeline({
  items,
  adapters,
  selectedIds,
  boundaries,
  onSelect,
  onSelectMany,
  onClearSelection,
  onBoundary,
  onApplySplit,
}: {
  items: KeyActionReviewItem[]
  adapters: KeyActionEvidenceAdapters | null
  selectedIds: Set<string>
  boundaries: Record<string, { start?: string; end?: string }>
  onSelect: (itemId: string) => void
  onSelectMany: (itemIds: string[]) => void
  onClearSelection: () => void
  onBoundary: (itemId: string, value: { start?: string; end?: string }) => void
  onApplySplit: () => void
}) {
  const [zoom, setZoom] = useState(1)
  const timedItems = items.filter(item => item.start_sec != null || item.end_sec != null)
  const adapterRows = Object.entries(adapters?.adapters || {}).map(([name, adapter]) => {
    const coverage = asRecord(adapter.coverage)
    const semanticIssueCount = Number(adapter.semantic_issue_count ?? 0)
    return {
      name,
      status: adapter.status || 'missing',
      start: coverage?.start_sec,
      end: coverage?.end_sec,
      rowCount: adapter.row_count ?? 0,
      issueCount: (adapter.error_count ?? 0) + (adapter.warning_count ?? 0),
      semanticIssueCount,
    }
  }).filter(row => row.start != null || row.end != null)
  const warningItems = timedItems.filter(item => item.item_type === 'qa_warning' || item.item_type === 'evidence_semantic')
  const segmentItems = timedItems.filter(item => item.item_type === 'segment')
  const microItems = timedItems.filter(item => item.item_type === 'micro_segment')
  const conflictItems = timedItems.filter(item => timelineConflict(item))
  const starts = [
    ...timedItems.map(item => numberValue(item.adjusted_start_sec ?? item.start_sec, 0)),
    ...adapterRows.map(row => numberValue(row.start, 0)),
  ]
  const ends = [
    ...timedItems.map(item => numberValue(item.adjusted_end_sec ?? item.end_sec, item.start_sec ?? 0)),
    ...adapterRows.map(row => numberValue(row.end, numberValue(row.start, 0))),
  ]
  const min = Math.min(0, ...starts)
  const max = Math.max(1, ...ends)
  const span = Math.max(1, max - min)
  const selectedItem = items.find(item => selectedIds.has(item.item_id) && (item.item_type === 'segment' || item.item_type === 'micro_segment'))
  const boundary = selectedItem ? boundaries[selectedItem.item_id] || {} : {}
  const selectedStart = selectedItem ? numberValue(boundary.start ?? selectedItem.adjusted_start_sec ?? selectedItem.start_sec, min) : min
  const selectedEnd = selectedItem ? numberValue(boundary.end ?? selectedItem.adjusted_end_sec ?? selectedItem.end_sec, max) : max
  const selectedPreview = selectedItem ? [...(selectedItem.preview_urls || []), ...(selectedItem.clip_urls || [])][0] : null
  const splitCount = items.filter(item => item.item_type === 'segment' && item.reasons?.includes('coarse_long_segment')).length
  const selectedTimelineCount = timedItems.filter(item => selectedIds.has(item.item_id)).length
  const zoomWidth = `${Math.round(zoom * 100)}%`

  function left(start: unknown) {
    return `${Math.max(0, Math.min(100, ((numberValue(start, min) - min) / span) * 100))}%`
  }

  function width(start: unknown, end: unknown) {
    const startValue = numberValue(start, min)
    const value = ((numberValue(end, startValue) - startValue) / span) * 100
    return `${Math.max(0.6, Math.min(100, value))}%`
  }

  function clamp(value: number, lower: number, upper: number) {
    return Math.max(lower, Math.min(upper, value))
  }

  function nudgeSelected(delta: number, mode: 'move' | 'start' | 'end') {
    if (!selectedItem) return
    const duration = Math.max(0.05, selectedEnd - selectedStart)
    if (mode === 'move') {
      const nextStart = clamp(selectedStart + delta, min, Math.max(min, max - duration))
      onBoundary(selectedItem.item_id, { start: nextStart.toFixed(2), end: (nextStart + duration).toFixed(2) })
      return
    }
    if (mode === 'start') {
      onBoundary(selectedItem.item_id, { start: clamp(selectedStart + delta, min, selectedEnd - 0.05).toFixed(2) })
      return
    }
    onBoundary(selectedItem.item_id, { end: clamp(selectedEnd + delta, selectedStart + 0.05, max).toFixed(2) })
  }

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (!selectedItem || (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight')) return
      const target = event.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(target.tagName)) return
      event.preventDefault()
      const direction = event.key === 'ArrowRight' ? 1 : -1
      const step = event.ctrlKey || event.metaKey ? 0.25 : event.altKey ? 1 : 0.05
      nudgeSelected(direction * step, event.shiftKey ? 'end' : event.altKey ? 'start' : 'move')
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedItem, selectedStart, selectedEnd, min, max, onBoundary])

  function itemTone(item: KeyActionReviewItem) {
    if (item.item_type === 'segment') return 'bg-blue-500'
    if (item.item_type === 'micro_segment') return 'bg-emerald-500'
    if (item.item_type === 'evidence_semantic') return 'bg-amber-500'
    if (item.item_type === 'qa_warning') return 'bg-red-500'
    return 'bg-slate-400'
  }

  function renderItemBar(item: KeyActionReviewItem) {
    const start = item.adjusted_start_sec ?? item.start_sec
    const end = item.adjusted_end_sec ?? item.end_sec ?? start
    const selected = selectedIds.has(item.item_id)
    const conflict = timelineConflict(item)
    const ring = selected ? 'ring-2 ring-slate-950' : conflict ? 'ring-2 ring-red-300' : ''
    return (
      <button
        key={item.item_id}
        type="button"
        title={`${typeLabel(item.item_type)} ${itemTime(item)} ${cleanDisplayText(item.title || '')}`}
        onClick={() => onSelect(item.item_id)}
        className={`absolute top-2 h-6 rounded ${itemTone(item)} ${ring} ${selected ? '' : 'opacity-80 hover:opacity-100'}`}
        style={{ left: left(start), width: width(start, end) }}
      />
    )
  }

  function renderItemLane(label: string, laneItems: KeyActionReviewItem[]) {
    return (
      <div className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-2">
        <div className="pt-2 text-[11px] font-black uppercase tracking-wide text-slate-400">{label}</div>
        <div className="relative h-10 rounded-lg bg-slate-50 ring-1 ring-slate-100">
          {laneItems.length ? laneItems.map(renderItemBar) : <span className="absolute left-3 top-3 text-[11px] font-semibold text-slate-300">empty</span>}
        </div>
      </div>
    )
  }

  return (
    <EvidenceCard className="p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="h-4 w-4 text-slate-500" />
          <h3 className="font-black text-slate-950">Review Timeline</h3>
          <EvidenceBadge tone="slate">{formatNumber(min, 1)}-{formatNumber(max, 1)}s</EvidenceBadge>
          <EvidenceBadge tone={conflictItems.length ? 'red' : 'emerald'}>{conflictItems.length} conflicts</EvidenceBadge>
          <EvidenceBadge tone="blue">{selectedTimelineCount} selected</EvidenceBadge>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" onClick={() => setZoom(value => clamp(Number((value - 0.25).toFixed(2)), 1, 4))} className={secondaryButtonClass()}>
            <ZoomOut className="h-4 w-4" />
          </button>
          <button type="button" onClick={() => setZoom(value => clamp(Number((value + 0.25).toFixed(2)), 1, 4))} className={secondaryButtonClass()}>
            <ZoomIn className="h-4 w-4" />
          </button>
          <button type="button" onClick={() => onSelectMany(timedItems.map(item => item.item_id))} className={secondaryButtonClass('blue')}>
            <Layers3 className="h-4 w-4" />
            Select timed
          </button>
          <button type="button" onClick={() => onSelectMany(conflictItems.map(item => item.item_id))} className={secondaryButtonClass('red')}>
            <AlertTriangle className="h-4 w-4" />
            Select conflicts
          </button>
          <button type="button" onClick={onClearSelection} className={secondaryButtonClass()}>
            Clear
          </button>
          <button type="button" onClick={onApplySplit} className={secondaryButtonClass('blue')}>
            <SplitSquareHorizontal className="h-4 w-4" />
            Apply split {splitCount ? `(${splitCount})` : ''}
          </button>
        </div>
      </div>

      <div className="overflow-x-auto rounded-lg border border-slate-100 bg-white">
        <div className="space-y-2 p-3" style={{ width: zoomWidth, minWidth: '100%' }}>
          {renderItemLane('Warnings', warningItems)}
          {renderItemLane('Segments', segmentItems)}
          {renderItemLane('Micros', microItems)}
          <div className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-2">
            <div className="pt-2 text-[11px] font-black uppercase tracking-wide text-slate-400">Adapters</div>
            <div className="relative h-10 rounded-lg bg-slate-50 ring-1 ring-slate-100">
              {adapterRows.length ? adapterRows.map(row => {
                const conflict = row.status !== 'pass' || row.semanticIssueCount > 0 || row.issueCount > 0
                return (
                  <button
                    key={row.name}
                    type="button"
                    title={`${row.name}: ${row.rowCount} rows, ${row.issueCount} issues, ${row.semanticIssueCount} semantic`}
                    className={`absolute top-2 h-6 rounded ${row.status === 'pass' ? 'bg-cyan-500' : row.status === 'fail' ? 'bg-red-500' : 'bg-amber-400'} ${conflict ? 'ring-2 ring-red-300' : 'opacity-80'}`}
                    style={{ left: left(row.start), width: width(row.start, row.end) }}
                  />
                )
              }) : <span className="absolute left-3 top-3 text-[11px] font-semibold text-slate-300">empty</span>}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_16rem]">
        <div className="grid gap-2 sm:grid-cols-2">
          {selectedItem ? (
            <>
              <div className="rounded-lg bg-slate-50 p-3">
                <div className="mb-2 flex items-center justify-between gap-2 text-xs font-bold text-slate-500">
                  <span>Start {formatNumber(selectedStart, 2)}s</span>
                  <Keyboard className="h-3.5 w-3.5" />
                </div>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={0.05}
                  value={Math.min(selectedStart, selectedEnd)}
                  onChange={event => onBoundary(selectedItem.item_id, { start: event.target.value })}
                  className="w-full accent-blue-600"
                />
              </div>
              <div className="rounded-lg bg-slate-50 p-3">
                <div className="mb-2 text-xs font-bold text-slate-500">End {formatNumber(selectedEnd, 2)}s</div>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={0.05}
                  value={Math.max(selectedStart, selectedEnd)}
                  onChange={event => onBoundary(selectedItem.item_id, { end: event.target.value })}
                  className="w-full accent-blue-600"
                />
              </div>
            </>
          ) : (
            <div className="rounded-lg bg-slate-50 p-3 text-xs font-semibold text-slate-500 sm:col-span-2">Select a segment or micro bar to adjust boundaries.</div>
          )}
        </div>
        <div className="min-h-20 overflow-hidden rounded-lg bg-slate-50 ring-1 ring-slate-100">
          {selectedPreview ? (
            selectedPreview.includes('.mp4') || selectedPreview.includes('.mov') ? (
              <a href={selectedPreview} target="_blank" rel="noreferrer" className="flex h-full min-h-20 items-center justify-center text-xs font-bold text-slate-600">Open clip</a>
            ) : (
              <a href={selectedPreview} target="_blank" rel="noreferrer">
                <img src={selectedPreview} alt="evidence preview" className="h-24 w-full object-cover" />
              </a>
            )
          ) : (
            <div className="flex h-full min-h-20 items-center justify-center text-xs font-semibold text-slate-400">No preview</div>
          )}
        </div>
      </div>
    </EvidenceCard>
  )
}

function timelineConflict(item: KeyActionReviewItem) {
  const severity = String(item.severity || '').toLowerCase()
  const reasons = (item.reasons || []).map(reason => String(reason).toLowerCase())
  const status = String(item.review_status || '').toLowerCase()
  return (
    severity === 'error' ||
    item.item_type === 'evidence_semantic' ||
    status === 'needs_review' ||
    reasons.some(reason => (
      reason.includes('semantic') ||
      reason.includes('conflict') ||
      reason.includes('mismatch') ||
      reason.includes('low') ||
      reason.includes('coarse') ||
      reason.includes('missing') ||
      reason.includes('unconfirmed')
    ))
  )
}

function ReviewItemCard({
  item,
  selected,
  note,
  boundary,
  saving,
  onToggle,
  onNote,
  onBoundary,
  onDecision,
}: {
  item: KeyActionReviewItem
  selected: boolean
  note: string
  boundary: { start?: string; end?: string }
  saving: boolean
  onToggle: () => void
  onNote: (value: string) => void
  onBoundary: (value: { start?: string; end?: string }) => void
  onDecision: (decision: string) => void
}) {
  const canAdjustBoundary = item.item_type === 'segment' || item.item_type === 'micro_segment'
  return (
    <EvidenceCard className={`p-4 ${selected ? 'ring-2 ring-blue-200' : ''}`}>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <input type="checkbox" checked={selected} onChange={onToggle} className="h-4 w-4 rounded border-slate-300" />
            <EvidenceBadge tone={severityTone(item.severity)}>{item.severity || 'info'}</EvidenceBadge>
            <EvidenceBadge tone={statusTone(item.review_status)}>{item.review_status || 'pending'}</EvidenceBadge>
            <EvidenceBadge tone="slate">{typeLabel(item.item_type)}</EvidenceBadge>
            <span className="font-mono text-xs font-bold text-slate-400">{item.item_id}</span>
          </div>
          <h3 className="mt-2 text-lg font-black text-slate-950">{cleanDisplayText(item.title || item.source_id || item.item_id)}</h3>
          {item.summary && <p className="mt-1 line-clamp-3 text-sm font-medium leading-6 text-slate-500">{cleanDisplayText(item.summary)}</p>}
          <div className="mt-3 flex flex-wrap gap-2 text-xs font-semibold text-slate-500">
            <span className="rounded bg-slate-50 px-2 py-1">{itemTime(item)}</span>
            {item.segment_id && <span className="rounded bg-slate-50 px-2 py-1">{item.segment_id}</span>}
            {item.micro_segment_id && <span className="rounded bg-slate-50 px-2 py-1">{item.micro_segment_id}</span>}
            {item.confidence != null && <span className="rounded bg-slate-50 px-2 py-1">confidence {formatNumber(item.confidence, 3)}</span>}
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          <button type="button" onClick={() => onDecision('approved')} disabled={saving} className={secondaryButtonClass('emerald')}>
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
            Approve
          </button>
          <button type="button" onClick={() => onDecision('needs_review')} disabled={saving} className={secondaryButtonClass('amber')}>
            <AlertTriangle className="h-4 w-4" />
            Hold
          </button>
          <button type="button" onClick={() => onDecision('rejected')} disabled={saving} className={secondaryButtonClass('red')}>
            <XCircle className="h-4 w-4" />
            Reject
          </button>
        </div>
      </div>

      <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div className="space-y-3">
          {item.reasons?.length ? (
            <div className="flex flex-wrap gap-1.5">
              {item.reasons.slice(0, 8).map(reason => (
                <span key={reason} className="rounded bg-amber-50 px-2 py-1 text-xs font-bold text-amber-700 ring-1 ring-amber-200">{reason}</span>
              ))}
            </div>
          ) : null}
          <textarea
            value={note}
            onChange={event => onNote(event.target.value)}
            rows={2}
            placeholder="审核备注"
            className="w-full resize-none rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
          />
          {canAdjustBoundary && (
            <div className="grid gap-2 sm:grid-cols-2">
              <input value={boundary.start ?? ''} onChange={event => onBoundary({ start: event.target.value })} placeholder={`start ${formatNumber(item.start_sec, 2)}s`} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold outline-none focus:border-blue-400" />
              <input value={boundary.end ?? ''} onChange={event => onBoundary({ end: event.target.value })} placeholder={`end ${formatNumber(item.end_sec, 2)}s`} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold outline-none focus:border-blue-400" />
            </div>
          )}
        </div>
        <div className="grid grid-cols-3 gap-2 lg:grid-cols-2">
          {(item.preview_urls || []).slice(0, 4).map(url => (
            <a key={url} href={url} target="_blank" rel="noreferrer" className="overflow-hidden rounded-lg border border-slate-200 bg-slate-50">
              <img src={url} alt="preview" className="aspect-video w-full object-cover" />
            </a>
          ))}
          {(item.clip_urls || []).slice(0, 2).map(url => (
            <a key={url} href={url} target="_blank" rel="noreferrer" className="flex aspect-video items-center justify-center rounded-lg border border-slate-200 bg-slate-900 text-xs font-bold text-white">
              clip
            </a>
          ))}
        </div>
      </div>
    </EvidenceCard>
  )
}

function QualityPanel({ queue }: { queue: KeyActionReviewQueue }) {
  const quality = queue.quality
  const recommendations = quality?.recommendations || []
  const splitCount = quality?.long_segment_split_candidates?.length || 0
  const boundaryCount = quality?.boundary_refinement_candidates?.length || 0
  const gate = asRecord(quality?.quality_gate)
  const gateSummary = asRecord(gate?.summary)
  const blocking = Array.isArray(gate?.blocking_checks) ? gate.blocking_checks as Record<string, unknown>[] : []
  return (
    <EvidenceCard className="p-4">
      <div className="mb-3 flex items-center gap-2">
        <Gauge className="h-4 w-4 text-slate-500" />
        <h3 className="font-black text-slate-950">Quality Convergence</h3>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs font-semibold text-slate-600">
        <DetailBox label="score" value={quality?.health_score ?? '-'} />
        <DetailBox label="status" value={quality?.status || '-'} />
        <DetailBox label="split candidates" value={splitCount} />
        <DetailBox label="boundary candidates" value={boundaryCount} />
        <DetailBox label="gate" value={recordString(gate, 'status', '-')} />
        <DetailBox label="blocking" value={recordString(gateSummary, 'blocking_count', '0')} />
      </div>
      {blocking.length > 0 && (
        <div className="mt-3 space-y-2">
          {blocking.slice(0, 4).map((item, index) => (
            <div key={index} className="rounded-lg bg-red-50 p-3 text-xs font-semibold text-red-700">
              <span className="font-black">{recordString(item, 'name')}</span>
              <div className="mt-1">{recordString(item, 'message')}</div>
            </div>
          ))}
        </div>
      )}
      <div className="mt-3 space-y-2">
        {recommendations.length === 0 ? (
          <EmptyEvidence title="No recommendations" />
        ) : recommendations.slice(0, 5).map((item, index) => (
          <div key={index} className="rounded-lg bg-slate-50 p-3 text-xs font-semibold text-slate-600">
            <span className="font-black text-slate-900">{recordString(item, 'priority')}</span>
            <span className="mx-1">/</span>
            {recordString(item, 'action')}
            <div className="mt-1 text-slate-500">{recordString(item, 'reason')}</div>
          </div>
        ))}
      </div>
    </EvidenceCard>
  )
}

function AdapterPanel({ adapters, counts }: { adapters: KeyActionEvidenceAdapters | null; counts: Record<string, number> }) {
  const inputs = adapters?.input_contracts || {}
  const validationAdapters = adapters?.adapters || {}
  return (
    <EvidenceCard className="p-4">
      <div className="mb-3 flex items-center gap-2">
        <FileJson className="h-4 w-4 text-slate-500" />
        <h3 className="font-black text-slate-950">Evidence Adapters</h3>
      </div>
      <div className="space-y-2">
        {Object.entries(inputs).map(([name, description]) => (
          (() => {
            const key = name.replace('.jsonl', '')
            const validation = validationAdapters[key]
            const rowCount = validation?.row_count ?? counts[key] ?? 0
            const status = validation?.status || ((rowCount || 0) > 0 ? 'pass' : 'missing')
            return (
          <div key={name} className="rounded-lg border border-slate-200 bg-white p-3">
            <div className="flex items-center justify-between gap-2">
              <div className="font-mono text-xs font-black text-slate-800">{name}</div>
              <EvidenceBadge tone={status === 'pass' ? 'emerald' : status === 'fail' ? 'red' : 'amber'}>
                {rowCount}
              </EvidenceBadge>
            </div>
            <div className="mt-1 text-xs font-medium leading-5 text-slate-500">{description}</div>
            <div className="mt-2 grid grid-cols-4 gap-1 text-[11px] font-bold text-slate-500">
              <span>{status}</span>
              <span>{validation?.error_count ?? 0} errors</span>
              <span>{validation?.warning_count ?? 0} warn</span>
              <span>{validation?.semantic_issue_count ?? 0} semantic</span>
            </div>
            {validation?.views?.length ? (
              <div className="mt-1 text-[11px] font-semibold text-slate-400">{validation.views.join(', ')}</div>
            ) : null}
          </div>
            )
          })()
        ))}
      </div>
    </EvidenceCard>
  )
}

function RetrievalPanel({ evaluation, saving, onRun }: { evaluation: Record<string, unknown> | null; saving: boolean; onRun: () => void }) {
  return (
    <EvidenceCard className="p-4">
      <div className="mb-3 flex items-center gap-2">
        <SearchCheck className="h-4 w-4 text-slate-500" />
        <h3 className="font-black text-slate-950">Retrieval Eval</h3>
      </div>
      <button type="button" onClick={onRun} disabled={saving} className={`${primaryButtonClass('blue')} w-full`}>
        {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <SearchCheck className="h-4 w-4" />}
        Run 50 Chinese Queries
      </button>
      {evaluation && (
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs font-semibold text-slate-600">
          <DetailBox label="status" value={recordString(evaluation, 'status', '-')} />
          <DetailBox label="queries" value={recordString(evaluation, 'query_count', '0')} />
          <DetailBox label="top1" value={percent(evaluation.top1_hit_rate)} />
          <DetailBox label="top3" value={percent(evaluation.topk_hit_rate)} />
          <DetailBox label="quality" value={percent(evaluation.quality_hit_rate)} />
          <DetailBox label="id hit" value={percent(evaluation.expected_id_hit_rate)} />
        </div>
      )}
    </EvidenceCard>
  )
}

function DetailBox({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2">
      <div className="text-[11px] font-black uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 break-words font-bold text-slate-700">{String(value ?? '-')}</div>
    </div>
  )
}

function ExperimentTabs({ experimentId }: { experimentId: string }) {
  const tabClass = 'rounded-md px-3 py-1.5 text-sm font-bold text-slate-600 transition hover:bg-slate-100 hover:text-slate-950'
  return (
    <nav className="flex flex-wrap gap-2">
      <Link to={`/experiments/${experimentId}/workspace`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'workspace')} onFocus={() => prefetchExperimentRoute(experimentId, 'workspace')} className={tabClass}>Analysis</Link>
      <Link to={`/experiments/${experimentId}/materials`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'materials')} onFocus={() => prefetchExperimentRoute(experimentId, 'materials')} className={tabClass}>Materials</Link>
      <Link to={`/experiments/${experimentId}/key-actions`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'keyActions')} onFocus={() => prefetchExperimentRoute(experimentId, 'keyActions')} className={tabClass}>Key Action</Link>
      <Link to={`/experiments/${experimentId}/key-actions/review`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'reviewQueue')} onFocus={() => prefetchExperimentRoute(experimentId, 'reviewQueue')} className="rounded-md bg-slate-900 px-3 py-1.5 text-sm font-bold text-white">Review Queue</Link>
    </nav>
  )
}
