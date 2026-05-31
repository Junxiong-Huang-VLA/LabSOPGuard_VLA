import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { experimentApi } from '../api'
import type { StepRecord, UpdateStepRequest } from '../types'

function displayValue(value: unknown, fallback = '-'): string {
  if (value === null || value === undefined || value === '') return fallback
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

function seconds(value: unknown): string {
  const numeric = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(numeric) ? `${numeric.toFixed(1)}s` : '-'
}

export default function StepDetail() {
  const { id, stepId } = useParams<{ id: string; stepId: string }>()
  const [step, setStep] = useState<StepRecord | null>(null)
  const [loading, setLoading] = useState(true)
  const [editing, setEditing] = useState(false)
  const [editForm, setEditForm] = useState<UpdateStepRequest>({
    step_name: '',
    step_description: '',
    status: 'candidate',
    notes: '',
  })

  const evidenceRefs = useMemo(() => step?.evidence_refs ?? [], [step])
  const parameters = useMemo(() => step?.parameters ?? [], [step])
  const provenance = step?.provenance

  useEffect(() => {
    if (!id || !stepId) return
    void loadStep()
  }, [id, stepId])

  const loadStep = async () => {
    if (!id || !stepId) return
    try {
      setLoading(true)
      const data = await experimentApi.getStep(id, stepId)
      setStep(data)
      setEditForm({
        step_name: data.step_name,
        step_description: data.step_description,
        status: data.status,
        notes: data.notes || '',
      })
    } catch (error) {
      console.error('Failed to load step:', error)
      setStep(null)
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    if (!id || !stepId) return
    try {
      await experimentApi.updateStep(id, stepId, editForm)
      await loadStep()
      setEditing(false)
    } catch (error) {
      console.error('Failed to update step:', error)
      alert('保存失败')
    }
  }

  if (loading) return <div className="py-12 text-center">加载中...</div>
  if (!step) return <div className="py-12 text-center text-red-600">步骤加载失败</div>

  return (
    <div>
      <div className="mb-6">
        <div className="mb-2 flex items-center space-x-2 text-sm text-gray-500">
          <Link to="/experiments" className="hover:text-gray-700">实验列表</Link>
          <span>/</span>
          <Link to={`/experiments/${id}/workspace`} className="hover:text-gray-700">实验详情</Link>
          <span>/</span>
          <span className="text-gray-900">步骤 {step.step_index}</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-2xl font-bold text-gray-900">{step.step_name}</h2>
          <button
            onClick={() => setEditing(value => !value)}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700"
          >
            {editing ? '取消编辑' : '编辑步骤'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <section className="rounded-lg bg-white p-6 shadow">
            <h3 className="mb-4 text-lg font-semibold">基本信息</h3>
            {editing ? (
              <div className="space-y-4">
                <label className="block text-sm font-medium text-gray-700">
                  步骤名称
                  <input
                    type="text"
                    value={editForm.step_name || ''}
                    onChange={event => setEditForm({ ...editForm, step_name: event.target.value })}
                    className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2"
                  />
                </label>
                <label className="block text-sm font-medium text-gray-700">
                  步骤描述
                  <textarea
                    value={editForm.step_description || ''}
                    onChange={event => setEditForm({ ...editForm, step_description: event.target.value })}
                    rows={3}
                    className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2"
                  />
                </label>
                <label className="block text-sm font-medium text-gray-700">
                  状态
                  <select
                    value={editForm.status || 'candidate'}
                    onChange={event => setEditForm({ ...editForm, status: event.target.value })}
                    className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2"
                  >
                    <option value="confirmed">confirmed</option>
                    <option value="candidate">candidate</option>
                    <option value="inferred">inferred</option>
                    <option value="skipped">skipped</option>
                    <option value="needs_review">needs_review</option>
                  </select>
                </label>
                <label className="block text-sm font-medium text-gray-700">
                  备注
                  <textarea
                    value={editForm.notes || ''}
                    onChange={event => setEditForm({ ...editForm, notes: event.target.value })}
                    rows={2}
                    className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2"
                  />
                </label>
                <button onClick={handleSave} className="rounded-md bg-green-600 px-4 py-2 text-sm text-white hover:bg-green-700">
                  保存修改
                </button>
              </div>
            ) : (
              <dl className="grid grid-cols-1 gap-4 text-sm sm:grid-cols-2">
                <div><dt className="font-medium text-gray-700">步骤索引</dt><dd className="text-gray-900">{step.step_index}</dd></div>
                <div><dt className="font-medium text-gray-700">状态</dt><dd className="text-gray-900">{step.status}</dd></div>
                <div className="sm:col-span-2"><dt className="font-medium text-gray-700">描述</dt><dd className="text-gray-900">{displayValue(step.step_description)}</dd></div>
                <div><dt className="font-medium text-gray-700">时间范围</dt><dd className="text-gray-900">{seconds(step.start_time_sec)} - {seconds(step.end_time_sec)}</dd></div>
                <div><dt className="font-medium text-gray-700">置信度</dt><dd className="text-gray-900">{Number(step.confidence ?? 0).toFixed(4)} {step.step_confidence ? `(${step.step_confidence})` : ''}</dd></div>
                {step.notes && <div className="sm:col-span-2"><dt className="font-medium text-gray-700">备注</dt><dd className="text-gray-900">{step.notes}</dd></div>}
              </dl>
            )}
          </section>

          {step.completed_by_inference && (
            <section className="rounded-lg border border-orange-200 bg-orange-50 p-6">
              <h3 className="mb-4 text-lg font-semibold text-orange-900">推断信息</h3>
              <div className="space-y-2 text-sm text-orange-900">
                <div>推断方法: {displayValue(step.inference_method)}</div>
                <div>使用模型: {displayValue(step.inference_model)}</div>
                <div>置信度: {Number(step.confidence ?? 0).toFixed(4)}</div>
              </div>
            </section>
          )}

          <section className="rounded-lg bg-white p-6 shadow">
            <h3 className="mb-4 text-lg font-semibold">证据引用 ({evidenceRefs.length})</h3>
            {evidenceRefs.length > 0 ? (
              <div className="space-y-3">
                {evidenceRefs.map(ref => (
                  <div key={ref.evidence_id} className="rounded-lg bg-gray-50 p-4">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <span className="text-sm font-medium text-gray-900">{ref.evidence_type}</span>
                      <span className="text-xs text-gray-500">置信度 {Number(ref.confidence ?? 0).toFixed(2)}</span>
                    </div>
                    <div className="space-y-1 text-sm text-gray-600">
                      <div>来源: {displayValue(ref.source)}</div>
                      {ref.frame_id !== undefined && <div>帧 ID: {displayValue(ref.frame_id)}</div>}
                      {ref.timestamp_sec !== undefined && <div>时间戳: {seconds(ref.timestamp_sec)}</div>}
                      {ref.description && <div>描述: {ref.description}</div>}
                    </div>
                    {ref.provenance && (
                      <div className="mt-2 border-t border-gray-200 pt-2 text-xs text-gray-500">
                        Provenance: {displayValue(ref.provenance.source)}, inferred={ref.provenance.is_inferred ? '是' : '否'}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-gray-500">暂无证据引用</div>
            )}
          </section>

          {parameters.length > 0 && (
            <section className="rounded-lg bg-white p-6 shadow">
              <h3 className="mb-4 text-lg font-semibold">操作参数 ({parameters.length})</h3>
              <div className="space-y-2">
                {parameters.map((param, index) => (
                  <div key={`${param.name}-${index}`} className="flex items-center justify-between rounded bg-gray-50 p-3 text-sm">
                    <span className="font-medium text-gray-900">{param.name}</span>
                    <span className="text-gray-600">{displayValue(param.value)} {param.unit || ''}</span>
                    <span className="text-xs text-gray-500">{displayValue(param.source)}</span>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>

        <aside className="space-y-6">
          {provenance && (
            <section className="rounded-lg bg-white p-6 shadow">
              <h3 className="mb-4 text-lg font-semibold">来源追踪</h3>
              <dl className="space-y-3 text-sm">
                <div><dt className="font-medium text-gray-700">来源</dt><dd className="text-gray-900">{displayValue(provenance.source)}</dd></div>
                <div><dt className="font-medium text-gray-700">是否推断</dt><dd className="text-gray-900">{provenance.is_inferred ? '是' : '否'}</dd></div>
                <div><dt className="font-medium text-gray-700">置信度</dt><dd className="text-gray-900">{Number(provenance.confidence ?? 0).toFixed(4)}</dd></div>
                {provenance.inference_method && <div><dt className="font-medium text-gray-700">推断方法</dt><dd className="text-gray-900">{provenance.inference_method}</dd></div>}
                {provenance.model_used && <div><dt className="font-medium text-gray-700">使用模型</dt><dd className="text-gray-900">{provenance.model_used}</dd></div>}
                <div><dt className="font-medium text-gray-700">时间戳</dt><dd className="text-xs text-gray-900">{displayValue(provenance.timestamp)}</dd></div>
              </dl>
            </section>
          )}

          <section className="rounded-lg bg-white p-6 shadow">
            <h3 className="mb-4 text-lg font-semibold">元数据</h3>
            <dl className="space-y-2 text-sm">
              <div><dt className="font-medium text-gray-700">创建时间</dt><dd className="text-xs text-gray-900">{displayValue(step.created_at)}</dd></div>
              <div><dt className="font-medium text-gray-700">更新时间</dt><dd className="text-xs text-gray-900">{displayValue(step.updated_at)}</dd></div>
            </dl>
          </section>
        </aside>
      </div>
    </div>
  )
}
