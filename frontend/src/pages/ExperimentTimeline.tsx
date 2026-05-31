import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { experimentApi } from '../api'
import { cleanDisplayText } from '../displayText'
import type { AnalysisOverview, StepRecord } from '../types'

function EmptyTimeline() {
  return (
    <div className="rounded-lg border border-dashed border-gray-300 bg-gray-50 p-8 text-center text-sm text-gray-600">
      <div className="font-medium text-gray-800 mb-2">暂无时间轴 No timeline</div>
      <div>尚未生成可对齐视频的步骤切片 No aligned step slices generated yet.</div>
    </div>
  )
}

function visibleSteps(overview: AnalysisOverview) {
  const { official, candidate, inferred } = overview.steps
  const all = [...official, ...candidate, ...(inferred ?? [])]
  const layer = official.length > 0 ? 'official' : candidate.length > 0 ? 'candidate' : 'none'
  return { all, layer }
}

export default function ExperimentTimelinePage() {
  const { id } = useParams<{ id: string }>()
  const [overview, setOverview] = useState<AnalysisOverview | null>(null)
  const [filter, setFilter] = useState<string>('all')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (id) void loadData()
  }, [id])

  const loadData = async () => {
    try {
      const data = await experimentApi.getAnalysisOverview(id!)
      setOverview(data)
    } catch (error) {
      console.error('Failed to load overview:', error)
    } finally {
      setLoading(false)
    }
  }

  const steps = useMemo(() => {
    if (!overview) return { all: [] as StepRecord[], layer: 'none' }
    return visibleSteps(overview)
  }, [overview])

  if (loading) return <div className="text-center py-12">加载中 Loading...</div>
  if (!overview) return <div className="text-center py-12 text-red-600">时间轴加载失败 Failed to load</div>

  const { summary } = overview
  const experimentName = cleanDisplayText(overview.experiment.experiment_name, `Experiment ${overview.experiment.experiment_id}`)
  const totalSteps = steps.all.length
  const confirmedCount = steps.all.filter(s => s.status === 'confirmed').length
  const candidateCount = steps.all.filter(s => s.status === 'candidate' || s.status === 'needs_review').length
  const inferredCount = steps.all.filter(s => s.status === 'inferred').length

  const filteredSteps = steps.all.filter(step => {
    if (filter === 'all') return true
    if (filter === 'candidate') return step.status === 'candidate' || step.status === 'needs_review'
    return step.status === filter
  })

  const maxTime = Math.max(
    ...steps.all.map(s => s.end_time_sec ?? s.start_time_sec ?? 0),
    1,
  )

  const getStepStatusColor = (status: string) => {
    switch (status) {
      case 'confirmed': return 'bg-green-500'
      case 'candidate': case 'needs_review': return 'bg-yellow-500'
      case 'inferred': return 'bg-orange-500'
      default: return 'bg-gray-500'
    }
  }

  const getStepStatusBadge = (status: string) => {
    switch (status) {
      case 'confirmed': return 'bg-green-100 text-green-800'
      case 'candidate': case 'needs_review': return 'bg-yellow-100 text-yellow-800'
      case 'inferred': return 'bg-orange-100 text-orange-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div>
      <div className="mb-6">
        <div className="flex items-center space-x-2 text-sm text-gray-500 mb-2">
          <Link to="/experiments" className="hover:text-gray-700">实验列表 Experiments</Link>
          <span>/</span>
          <Link to={`/experiments/${id}/workspace`} className="hover:text-gray-700">{experimentName}</Link>
          <span>/</span>
          <span className="text-gray-900">步骤时间轴 Timeline</span>
        </div>
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-900">步骤时间轴 Step Timeline</h2>
          <span className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-500">{steps.layer}</span>
        </div>
      </div>

      <div className="grid grid-cols-5 gap-4 mb-6">
        <StatCard label="总步骤 Total" value={String(totalSteps)} />
        <StatCard label="已确认 Confirmed" value={String(confirmedCount)} />
        <StatCard label="候选 Candidate" value={String(candidateCount)} />
        <StatCard label="推断 Inferred" value={String(inferredCount)} />
        <StatCard label="平均置信度 Avg Conf" value={summary.avg_confidence == null ? '-' : Number(summary.avg_confidence).toFixed(2)} />
      </div>

      <div className="mb-6 flex space-x-2">
        {(['all', 'confirmed', 'candidate', 'inferred'] as const).map(status => (
          <button
            key={status}
            onClick={() => setFilter(status)}
            className={`px-4 py-2 rounded-md text-sm font-medium ${filter === status ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'}`}
          >
            {{all: '全部 All', confirmed: '已确认 Confirmed', candidate: '候选 Candidate', inferred: '推断 Inferred'}[status]}
          </button>
        ))}
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">时间轴可视化 Timeline</h3>
        {steps.all.length === 0 ? (
          <EmptyTimeline />
        ) : (
          <>
            <div className="relative h-36 bg-gray-100 rounded overflow-hidden">
              {steps.all.map((step, idx) => {
                const startSec = step.start_time_sec ?? 0
                const endSec = step.end_time_sec ?? startSec
                const startPct = (startSec / maxTime) * 100
                const widthPct = ((endSec - startSec) / maxTime) * 100
                return (
                  <div
                    key={step.step_id || idx}
                    className={`absolute h-full ${getStepStatusColor(step.status)} opacity-60 hover:opacity-100 transition-opacity flex items-center justify-center text-white text-xs font-medium cursor-default border-r border-white/30`}
                    style={{ left: `${startPct}%`, width: `${Math.max(widthPct, 1.5)}%`, top: 0 }}
                    title={`${cleanDisplayText(step.step_name, `Step ${step.step_index ?? idx}`)}\n${startSec.toFixed(1)}s - ${endSec.toFixed(1)}s\nconfidence: ${(step.confidence ?? 0).toFixed(2)}`}
                  >
                    {widthPct > 6 && <span className="truncate px-1">{step.step_index ?? idx}</span>}
                  </div>
                )
              })}
            </div>
            <div className="mt-2 flex justify-between text-xs text-gray-500">
              <span>0s</span>
              <span>{maxTime.toFixed(1)}s</span>
            </div>
          </>
        )}
      </div>

      <div className="space-y-4">
        {filteredSteps.map((step, idx) => (
          <div key={step.step_id || idx} className="bg-white rounded-lg shadow p-6">
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <span className="text-sm font-medium text-gray-500">Step {step.step_index ?? idx}</span>
                  <h3 className="text-lg font-semibold text-gray-900">{cleanDisplayText(step.step_name, `Step ${step.step_index ?? idx}`)}</h3>
                  <span className={`px-2 py-0.5 text-xs font-medium rounded ${getStepStatusBadge(step.status)}`}>{step.status}</span>
                </div>
                {step.step_description && <p className="text-sm text-gray-600">{cleanDisplayText(step.step_description, '')}</p>}
              </div>
              <div className="ml-4 text-right">
                <div className="text-sm font-medium text-gray-900">{(step.start_time_sec ?? 0).toFixed(1)}s - {(step.end_time_sec ?? 0).toFixed(1)}s</div>
                <div className="text-xs text-gray-500 mt-1">置信度 Confidence {(step.confidence ?? 0).toFixed(2)}</div>
              </div>
            </div>
            <div className="flex items-center space-x-4 text-sm text-gray-600">
              {step.evidence_refs && <span>证据 Evidence: {step.evidence_refs.length}</span>}
            </div>
          </div>
        ))}
      </div>

      {filteredSteps.length === 0 && <EmptyTimeline />}
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="text-sm text-gray-600">{label}</div>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
    </div>
  )
}
