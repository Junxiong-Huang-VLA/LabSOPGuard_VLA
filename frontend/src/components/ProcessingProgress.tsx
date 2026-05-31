import { useEffect, useState } from 'react'
import { experimentApi, type ExperimentTimingResponse, type ProcessingQueueStatusResponse } from '../api'
import type { DisplayTimingStageSummary } from '../types'

const DISPLAY_STAGE_DEFS: Array<{ stage: string; label_zh: string; missing_reason_zh: string }> = [
  { stage: 'total_elapsed', label_zh: '总耗时', missing_reason_zh: '未记录端到端总耗时' },
  { stage: 'upload_register', label_zh: '上传入库', missing_reason_zh: '未记录上传入库耗时' },
  { stage: 'time_alignment', label_zh: '时间轴对齐', missing_reason_zh: '未记录时间轴对齐耗时' },
  { stage: 'coarse_scan', label_zh: '长视频粗扫', missing_reason_zh: '未记录长视频粗扫耗时' },
  { stage: 'fine_scan', label_zh: '片段精扫', missing_reason_zh: '未记录片段精扫耗时' },
  { stage: 'action_alignment', label_zh: '动作对齐', missing_reason_zh: '未记录动作对齐耗时' },
  { stage: 'micro_gate', label_zh: 'Micro/Gate', missing_reason_zh: '未记录 Micro/Gate 耗时' },
  { stage: 'material_generation', label_zh: '素材生成', missing_reason_zh: '未记录素材生成耗时' },
  { stage: 'material_publish', label_zh: '素材发布', missing_reason_zh: '未记录素材发布耗时' },
  { stage: 'memory_write', label_zh: 'Memory 写入', missing_reason_zh: '未记录 Memory 写入耗时' },
  { stage: 'result_loading', label_zh: '前端加载', missing_reason_zh: '未记录前端加载耗时' },
]

function finiteNumber(value: unknown) {
  if (value == null || value === '') return null
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

function formatSec(value: unknown) {
  const seconds = finiteNumber(value)
  if (seconds == null) return '-'
  if (seconds < 60) return `${seconds.toFixed(1)} 秒`
  const whole = Math.round(seconds)
  const minutes = Math.floor(whole / 60)
  const remain = whole % 60
  if (minutes < 60) return `${minutes} 分 ${remain.toString().padStart(2, '0')} 秒`
  const hours = Math.floor(minutes / 60)
  const mins = minutes % 60
  return `${hours} 小时 ${mins.toString().padStart(2, '0')} 分`
}

function normalizedDisplayRows(timing: ExperimentTimingResponse | null): DisplayTimingStageSummary[] {
  const rows = Array.isArray(timing?.display_stages) ? timing.display_stages : []
  const byStage = new Map(rows.map(row => [row.stage, row]))
  return DISPLAY_STAGE_DEFS.map(def => {
    const row = byStage.get(def.stage)
    const history = timing?.timing_history_summary?.stages?.[def.stage]
    const duration = finiteNumber(row?.duration_sec)
    return {
      ...row,
      stage: def.stage,
      label_zh: row?.label_zh || def.label_zh,
      duration_sec: duration,
      available: duration != null && row?.available !== false,
      missing_reason_zh: duration == null ? row?.missing_reason_zh || def.missing_reason_zh : undefined,
      p50_sec: finiteNumber(row?.p50_sec) ?? finiteNumber(history?.p50_sec),
      p90_sec: finiteNumber(row?.p90_sec) ?? finiteNumber(history?.p90_sec),
      history_sample_count: Number(row?.history_sample_count ?? history?.sample_count ?? 0),
    }
  })
}

export default function ProcessingProgress({ experimentId }: { experimentId: string }) {
  const [timing, setTiming] = useState<ExperimentTimingResponse | null>(null)
  const [queue, setQueue] = useState<ProcessingQueueStatusResponse | null>(null)

  useEffect(() => {
    const fetch = () => {
      experimentApi.getTiming(experimentId, { force: true })
        .then(setTiming)
        .catch(() => undefined)
      experimentApi.getProcessingQueueStatus({ force: true })
        .then(setQueue)
        .catch(() => undefined)
    }
    fetch()
    const interval = setInterval(fetch, 3000)
    return () => clearInterval(interval)
  }, [experimentId])

  const displayRows = normalizedDisplayRows(timing)
  const hasTiming = displayRows.some(row => row.available)

  if (!hasTiming && !queue?.queue_size && !queue?.processing_count) {
    return null
  }

  return (
    <div className="mb-4 rounded-lg border bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-gray-700">处理进度</h3>
      {queue && (queue.processing_count > 0 || queue.queue_size > 0) && (
        <div className="mb-3 rounded-md bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-600">
          GPU 队列：处理中 {queue.processing_count}/{queue.max_concurrent}，等待 {queue.queue_size}
        </div>
      )}
      <div className="space-y-2">
        {displayRows.map(row => {
          const available = row.available !== false && finiteNumber(row.duration_sec) != null
          return (
            <div key={row.stage} className="flex items-center gap-2 text-sm">
              <span className="w-5 text-center text-emerald-600">{available ? '✓' : '○'}</span>
              <span className="min-w-0 flex-1 text-gray-700">{row.label_zh}</span>
              <span className="text-xs text-gray-500">
                {available ? formatSec(row.duration_sec) : '未记录'}
              </span>
            </div>
          )
        })}
      </div>
      {(timing?.total_sec || 0) > 0 && (
        <div className="mt-3 border-t pt-2 text-xs text-gray-500">
          总耗时：{formatSec(timing?.total_sec)}
          {timing?.core_analysis_sec ? `；算法分析 ${formatSec(timing.core_analysis_sec)}` : ''}
        </div>
      )}
    </div>
  )
}
