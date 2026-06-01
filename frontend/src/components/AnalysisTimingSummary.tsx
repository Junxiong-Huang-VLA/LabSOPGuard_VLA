import type { AnalysisOverview } from '../types'
import { getDemoTiming } from '../demo/weighingPipettingDemo'

type Run = AnalysisOverview['run']

type TimingStage = {
  stage: string
  label_zh?: string
  duration_sec?: number | null
  available?: boolean
}

type StageTile = {
  id: string
  label: string
  keys: string[]
  fallback?: string[]
  includeTotal?: boolean
}

type TimingMap = Record<string, number | null>

const STAGE_TILES: StageTile[] = [
  {
    id: 'total_elapsed',
    label: '分析总耗时',
    keys: ['total_elapsed', 'total_runtime', 'total', 'server_end_to_end_sec', 'elapsed_sec', 'elapsed', 'total_sec'],
    includeTotal: true,
  },
  {
    id: 'time_alignment',
    label: '时间对齐',
    keys: ['time_alignment', 'time_axis', 'timestamp_alignment', 'video_time_sync'],
    fallback: ['alignment', 'sync'],
  },
  {
    id: 'coarse_scan',
    label: '粗筛',
    keys: ['coarse_scan', 'parallel_scan', 'window_scan', 'coarse'],
    fallback: ['coarse_scan', 'coarse'],
  },
  {
    id: 'fine_scan',
    label: '细筛',
    keys: ['fine_scan', 'segment_generation', 'micro_segment', 'segment_scan', 'fine'],
    fallback: ['fine_scan', 'segment_generation', 'segment'],
  },
  {
    id: 'material_publish',
    label: '关键素材生成',
    keys: ['material_publish', 'material_generation', 'material_build', 'publish'],
    fallback: ['material_publish', 'material', 'publish'],
  },
  {
    id: 'memory_write',
    label: '记忆写入',
    keys: ['memory_write', 'video_memory', 'write_video_memory', 'memory'],
    fallback: ['memory_write', 'video_memory', 'memory'],
  },
]

function normalizeStage(value: unknown) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_')
}

function parseFinite(value: unknown) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function formatDuration(value?: number | null) {
  if (!Number.isFinite(Number(value))) return '未记录'
  return `${parseFloat(Number(value).toFixed(1))} 秒`
}

function buildTimingMap(timing: Run['timing']): TimingMap {
  if (!timing) return {}
  const map: TimingMap = {}
  const simpleMap = (timing as { stages?: Record<string, unknown> }).stages
  if (simpleMap && typeof simpleMap === 'object' && !Array.isArray(simpleMap)) {
    for (const [key, value] of Object.entries(simpleMap)) {
      const duration = parseFinite(value)
      if (duration == null) continue
      map[normalizeStage(key)] = duration
    }
  }

  const stageRows = Array.isArray((timing as { stages?: unknown }).stages)
    ? ((timing as { stages?: unknown }).stages as unknown[])
    : []
  const namedStageRows = Array.isArray((timing as { stage_rows?: unknown }).stage_rows)
    ? ((timing as { stage_rows?: unknown }).stage_rows as unknown[])
    : []
  const displayRows = Array.isArray((timing as { display_stages?: unknown }).display_stages)
    ? ((timing as { display_stages?: unknown }).display_stages as unknown[])
    : []
  const rows = [...stageRows, ...namedStageRows, ...displayRows]

  for (const row of rows) {
    if (!row || typeof row !== 'object') continue
    const entry = row as Record<string, unknown>
    const stage = normalizeStage(entry.stage)
    const duration = parseFinite(entry.duration_sec)
    if (!stage || duration == null) continue
    if (map[stage] == null) map[stage] = duration
  }

  return map
}

function findDisplayText(raw: unknown) {
  return String(raw || '').trim() || null
}

function pickDurationFromMap(stageMap: TimingMap, keys: string[], fallback: string[]) {
  const direct = keys.map(key => stageMap[normalizeStage(key)]).find(value => value != null)
  if (direct != null) return direct
  const fallbackValue = Object.entries(stageMap).find(([stageKey]) => fallback.some(item => stageKey.includes(normalizeStage(item))))?.[1]
  return fallbackValue ?? null
}

type Props = {
  run: Run
  clientEndToEndSec?: number
  demo?: boolean
}

function DemoTimingSummary() {
  const { totalSec, stages } = getDemoTiming()
  const tiles = [{ id: 'total_elapsed', label: '分析总耗时', durationSec: totalSec }, ...stages]
  return (
    <section className="space-y-3" data-smoke="analysis-timing-summary">
      <h3 className="text-base font-semibold text-[color:var(--ui-text)]">分析耗时</h3>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {tiles.map(tile => (
          <article
            key={tile.id}
            className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/85 p-3 shadow-[var(--ui-shadow-subtle)]"
            data-smoke={`timing-tile-${tile.id}`}
          >
            <p className="text-xs font-medium text-[color:var(--ui-text-muted)]">{tile.label}</p>
            <p className="mt-2 text-lg font-semibold text-[color:var(--ui-text)]">{tile.durationSec} 秒</p>
          </article>
        ))}
      </div>
    </section>
  )
}

export default function AnalysisTimingSummary({ run, clientEndToEndSec, demo }: Props) {
  if (demo) {
    return <DemoTimingSummary />
  }
  const timing = run.timing || {}
  const displayStages = Array.isArray((timing as { display_stages?: unknown }).display_stages)
    ? (timing as { display_stages?: unknown }).display_stages as TimingStage[]
    : []

  const displayStageMap = new Map(
    displayStages.map(item => [normalizeStage(item.stage), {
      duration: parseFinite(item.duration_sec),
      available: item.available,
      label: findDisplayText(item.label_zh),
    }]),
  )

  const stageMap = buildTimingMap(timing)
  const orderedTiles = (() => {
    const configuredOrder = Array.isArray((timing as { display_stage_order?: unknown }).display_stage_order)
      ? (timing as { display_stage_order?: unknown }).display_stage_order
      : []
    if (!Array.isArray(configuredOrder) || configuredOrder.length === 0) return STAGE_TILES

    const configuredIds = (configuredOrder as unknown[]).map(item => normalizeStage(item)).filter(Boolean)
    const knownIds = new Set(STAGE_TILES.map(stage => stage.id))
    const seen = new Set<string>()
    const uniqueConfigured = configuredIds.filter(id => {
      if (!knownIds.has(id) || seen.has(id)) return false
      seen.add(id)
      return true
    })
    const missingIds = STAGE_TILES.map(stage => stage.id).filter(id => !uniqueConfigured.includes(id))

    return [...uniqueConfigured, ...missingIds].map(id => STAGE_TILES.find(stage => stage.id === id)!).filter(Boolean)
  })()

  const totalElapsed = parseFinite(clientEndToEndSec)
    ?? parseFinite(timing.server_end_to_end_sec)
    ?? parseFinite(timing.elapsed_sec)
    ?? parseFinite((timing as { total_sec?: unknown }).total_sec)

  return (
    <section className="space-y-3" data-smoke="analysis-timing-summary">
      <h3 className="text-base font-semibold text-[color:var(--ui-text)]">分析耗时</h3>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {orderedTiles.map(tile => {
          const direct = displayStageMap.get(normalizeStage(tile.id))
          const matched = direct?.duration != null
            ? direct.duration
            : pickDurationFromMap(stageMap, tile.keys, tile.fallback || [])
          const isTotal = tile.id === 'total_elapsed'
          const available = direct?.available !== undefined ? Boolean(direct.available) : matched != null
          const value = isTotal ? totalElapsed ?? matched : matched
          const showLabel = direct?.label || tile.label

          return (
            <article
              key={tile.id}
              className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/85 p-3 shadow-[var(--ui-shadow-subtle)]"
              data-smoke={`timing-tile-${tile.id}`}
            >
              <p className="text-xs font-medium text-[color:var(--ui-text-muted)]">{showLabel}</p>
              <p className="mt-2 text-lg font-semibold text-[color:var(--ui-text)]">{formatDuration(available && Number.isFinite(value as number) ? value : null)}</p>
              {!available && <p className="mt-1 text-xs text-[color:var(--ui-text-muted)]">未记录</p>}
            </article>
          )
        })}
      </div>
    </section>
  )
}
