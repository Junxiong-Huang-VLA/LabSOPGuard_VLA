import type { AnalysisOverview } from '../types'

function finiteNumber(value: unknown) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function formatHHMMSS(value: unknown) {
  const seconds = finiteNumber(value)
  if (seconds == null) return '-'
  const safe = Math.max(0, Math.floor(seconds))
  const h = Math.floor(safe / 3600)
  const m = Math.floor((safe % 3600) / 60)
  const s = safe % 60
  if (h > 0) return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

type Run = AnalysisOverview['run']

type StageDef = {
  id: string
  label: string
  hint: string
  keys: string[]
  fallbackLabel: string
}

const STAGE_DEFS: StageDef[] = [
  {
    id: 'camera_alignment',
    label: '摄像头对齐',
    hint: '先完成时间轴匹配与双视角起始点对齐，保证后续切分参照同一时间基线。',
    keys: ['time_alignment', 'timestamp_alignment', 'video_time_sync', 'time_axis'],
    fallbackLabel: '对齐耗时',
  },
  {
    id: 'window_split',
    label: '实验切分',
    hint: '基于活动与停顿判定生成实验窗口，保留可复核片段入口。',
    keys: ['coarse_scan', 'fine_scan', 'segment_generation', 'window_generation', 'experiment_split', 'segment_detection'],
    fallbackLabel: '切分耗时',
  },
  {
    id: 'material_publish',
    label: '材料生成',
    hint: '输出窗口级材料、关键帧和关键片段，并同步到官方素材入口。',
    keys: ['material_generation', 'material_publish', 'material_generation_wall', 'material_build', 'publish'],
    fallbackLabel: '生成耗时',
  },
]

function stageFromRows(timing: Run['timing']): Record<string, number | null> {
  if (!timing) return {}
  const stageSource = [
    ...(Array.isArray((timing as { stages?: unknown }).stages) ? (timing as { stages?: unknown }).stages : []),
    ...(Array.isArray((timing as { stage_rows?: unknown }).stage_rows) ? (timing as { stage_rows?: unknown }).stage_rows : []),
    ...(Array.isArray((timing as { display_stages?: unknown }).display_stages) ? (timing as { display_stages?: unknown }).display_stages : []),
  ] as unknown[]

  const map = new Map<string, number>()
  for (const row of stageSource) {
    if (!row || typeof row !== 'object') continue
    const entry = row as Record<string, unknown>
    const stage = String(entry.stage || entry.label || '').trim()
    if (!stage) continue
    const value = finiteNumber(entry.duration_sec)
    if (value == null) continue
    const normalized = stage.toLowerCase().replace(/\s+/g, '_')
    if (!map.has(normalized)) map.set(normalized, value)
  }
  return Object.fromEntries(map)
}

function pickDuration(rowByStage: Record<string, number | null>, keys: string[]) {
  const direct = keys.map(key => rowByStage[key]).find(finiteNumber)
  if (direct != null) return direct
  const fallback = Object.entries(rowByStage)
    .find(([key]) => keys.some(prefix => key.includes(prefix)) && finiteNumber(rowByStage[key]) != null)
  return fallback ? finiteNumber(fallback[1]) : null
}

function getProgressState(status: string, progress: number, index: number) {
  const cleaned = String(status || '').toLowerCase()
  if (['failed', 'error', 'blocked'].some(item => cleaned.includes(item))) {
    return index < 2 ? '已完成' : '已停止'
  }
  if (['completed', 'partial_completed'].some(item => cleaned.includes(item))) return '已完成'
  if (index === 0) {
    if (progress <= 0.05) return '排队中'
    return '进行中'
  }
  if (index === 1) return progress >= 0.45 ? '进行中' : '待启动'
  return progress >= 0.85 ? '进行中' : '待启动'
}

type Props = {
  run: Run
  statusLabel?: string
  clientEndToEndSec?: number
}

export default function AnalysisTimingSummary({ run, statusLabel, clientEndToEndSec }: Props) {
  const timing = run.timing || {}
  const stageRows = stageFromRows(timing)
  const runStatus = String(statusLabel || run.status || '').toLowerCase()
  const progressValue = Number.isFinite(run.progress) ? Number(run.progress) : 0
  const endToEnd = finiteNumber(clientEndToEndSec) || finiteNumber(timing.server_end_to_end_sec) || finiteNumber(timing.elapsed_sec) || null

  const stageEntries = STAGE_DEFS.map(stage => {
    const duration = pickDuration(stageRows, stage.keys)
    return {
      ...stage,
      status: getProgressState(runStatus, progressValue, STAGE_DEFS.indexOf(stage)),
      durationSec: duration,
      durationText: duration != null ? formatHHMMSS(duration) : '-',
    }
  })

  const advanced = {
    stage_count: timing.stage_count,
    total_sec: timing.server_end_to_end_sec || timing.elapsed_sec,
    queue_wait_sec: timing.queue_wait_sec,
    algorithm_elapsed_sec: timing.algorithm_elapsed_sec,
    upload_save_sec: timing.upload_save_sec,
    coarse_sampled_frame_count: (timing as { coarse_sampled_frame_count?: unknown }).coarse_sampled_frame_count,
    fine_sampled_frame_count: (timing as { fine_sampled_frame_count?: unknown }).fine_sampled_frame_count,
    coarse_wall_sec: (timing as { coarse_wall_sec?: unknown }).coarse_wall_sec,
    fine_wall_sec: (timing as { fine_wall_sec?: unknown }).fine_wall_sec,
    gpu_device: (timing as { gpu_device?: unknown }).gpu_device,
    batch_size: (timing as { batch_size?: unknown }).batch_size,
  }

  return (
    <section className="space-y-3">
      <div className="grid gap-3 xl:grid-cols-3">
        {stageEntries.map(item => (
          <article key={item.id} className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/85 p-4 shadow-[var(--ui-shadow-subtle)]">
            <div className="flex items-start justify-between gap-3">
              <p className="text-sm font-semibold text-[color:var(--ui-text)]">{item.label}</p>
              <span className="rounded-full border border-[color:var(--ui-accent-soft)] bg-[color:var(--ui-accent-soft)] px-2 py-1 text-xs font-medium text-[color:var(--ui-accent)]">{item.status}</span>
            </div>
            <p className="mt-1 text-xs text-[color:var(--ui-text-muted)]">{item.hint}</p>
            <p className="mt-3 text-lg font-semibold text-[color:var(--ui-text)]">{item.durationText}</p>
            <p className="text-xs text-[color:var(--ui-text-muted)]">{item.fallbackLabel}</p>
          </article>
        ))}
      </div>

      <article className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/85 p-4 shadow-[var(--ui-shadow-subtle)]">
        <p className="text-sm font-semibold text-[color:var(--ui-text)]">总耗时</p>
        <p className="mt-2 text-lg font-semibold text-[color:var(--ui-text)]">{formatHHMMSS(endToEnd)} {endToEnd ? '' : '(待刷新)'}</p>
        {run.message && <p className="mt-2 text-xs text-[color:var(--ui-text-muted)]">{run.message}</p>}
      </article>

      <details className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/85 p-4 shadow-[var(--ui-shadow-subtle)]">
        <summary className="cursor-pointer text-sm font-semibold text-[color:var(--ui-text)]">高级信息（指标细节）</summary>
        <div className="mt-3 grid gap-2 text-xs text-[color:var(--ui-text-muted)]">
          {Object.entries(advanced)
            .filter(([, value]) => value != null && value !== '')
            .map(([key, value]) => (
              <p key={key}>
                <span className="mr-2 font-medium text-[color:var(--ui-text)]">{key}</span>
                {String(value)}
              </p>
            ))}
          {Object.keys(advanced).every(key => !Object.values(advanced).some(value => value != null && value !== '')) && (
            <p className="text-sm">当前阶段仅保留核心指标，后台指标可稍后刷新后查看。</p>
          )}
        </div>
      </details>
    </section>
  )
}
