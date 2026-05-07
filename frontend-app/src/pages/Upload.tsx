import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowLeft, CheckCircle2, FileVideo2, Loader2, UploadCloud } from 'lucide-react'
import { experimentApi } from '../api'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, PageHero, primaryButtonClass, secondaryButtonClass } from '../components/EvidenceUI'

type Slot = 'first' | 'third' | 'supplemental'

export default function Upload() {
  const navigate = useNavigate()
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [files, setFiles] = useState<Record<Slot, File[]>>({ first: [], third: [], supplemental: [] })
  const [progress, setProgress] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const allFiles = useMemo(() => [...files.first, ...files.third, ...files.supplemental], [files])
  const canSubmit = title.trim() && allFiles.length > 0 && !submitting

  function updateFiles(slot: Slot, nextFiles: FileList | null) {
    setFiles(previous => ({ ...previous, [slot]: Array.from(nextFiles || []) }))
  }

  async function submit() {
    if (!canSubmit) return
    setSubmitting(true)
    setError(null)
    setProgress(0)
    try {
      const created = await experimentApi.create({ title: title.trim(), description: description.trim() })
      await experimentApi.uploadAndAnalyze(
        created.experiment_id,
        {
          firstPersonVideo: files.first[0] || null,
          thirdPersonVideo: files.third[0] || null,
          topVideo: files.supplemental[0] || null,
        },
        setProgress,
      )
      navigate(`/experiments/${created.experiment_id}/workspace`)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '上传失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-5">
      <PageHero
        eyebrow={<Link to="/experiments" className="hover:text-slate-900">实验队列</Link>}
        title="采集导入"
        description="面向第一视角、第三视角和补充视角的视频导入。上传后立即进入分析与关键动作证据抽取流程。"
        actions={(
          <Link to="/experiments" className={secondaryButtonClass()}>
            <ArrowLeft className="h-4 w-4" />
            返回队列
          </Link>
        )}
      />

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_22rem]">
        <EvidenceCard className="p-5">
          <div className="grid gap-4 lg:grid-cols-2">
            <label className="block">
              <span className="mb-2 block text-sm font-black text-slate-700">实验名称</span>
              <input value={title} onChange={event => setTitle(event.target.value)} className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100" placeholder="例如：ELISA 加样过程" />
            </label>
            <label className="block">
              <span className="mb-2 block text-sm font-black text-slate-700">实验描述</span>
              <input value={description} onChange={event => setDescription(event.target.value)} className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100" placeholder="批次、样本或 SOP 摘要" />
            </label>
          </div>

          <div className="mt-5 grid gap-4 lg:grid-cols-3">
            <UploadSlot title="第一视角" helper="操作者佩戴/近距视角" files={files.first} onChange={list => updateFiles('first', list)} />
            <UploadSlot title="第三视角" helper="台面/环境固定视角" files={files.third} onChange={list => updateFiles('third', list)} />
            <UploadSlot title="补充视角" helper="可选视频，同步进入分析" files={files.supplemental} onChange={list => updateFiles('supplemental', list)} multiple />
          </div>

          {error && <div className="mt-4 rounded-lg bg-red-50 p-3 text-sm font-semibold text-red-700">{error}</div>}
          {submitting && (
            <div className="mt-4">
              <div className="mb-2 flex justify-between text-sm font-bold text-slate-600"><span>上传并分析</span><span>{progress}%</span></div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-100"><div className="h-full rounded-full bg-blue-600 transition-all" style={{ width: `${progress}%` }} /></div>
            </div>
          )}
        </EvidenceCard>

        <EvidenceCard className="p-5">
          <div className="mb-4 flex items-center gap-2">
            <UploadCloud className="h-4 w-4 text-blue-600" />
            <h3 className="font-black text-slate-950">导入检查</h3>
          </div>
          <div className="space-y-3 text-sm font-semibold text-slate-600">
            <CheckRow ok={Boolean(title.trim())} label="实验名称已填写" />
            <CheckRow ok={files.first.length > 0} label="第一视角视频" />
            <CheckRow ok={files.third.length > 0} label="第三视角视频" />
            <CheckRow ok={allFiles.length > 0} label={`${allFiles.length} 个视频待上传`} />
          </div>
          <button type="button" disabled={!canSubmit} onClick={() => void submit()} className={`${primaryButtonClass('blue')} mt-5 w-full disabled:cursor-not-allowed disabled:opacity-50`}>
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <UploadCloud className="h-4 w-4" />}
            创建并分析
          </button>
          <EmptyEvidence title="Dry-run 兼容" description="没有真实视频或 ffmpeg 时，后端 dry-run 仍可返回可审阅的占位状态。" />
        </EvidenceCard>
      </div>
    </div>
  )
}

function UploadSlot({ title, helper, files, onChange, multiple = false }: { title: string; helper: string; files: File[]; onChange: (files: FileList | null) => void; multiple?: boolean }) {
  return (
    <label className="block rounded-lg border border-dashed border-slate-300 bg-slate-50 p-4 transition hover:border-blue-300 hover:bg-blue-50/40">
      <span className="mb-3 flex items-center justify-between gap-2">
        <span>
          <span className="block font-black text-slate-900">{title}</span>
          <span className="block text-xs font-semibold text-slate-500">{helper}</span>
        </span>
        <FileVideo2 className="h-5 w-5 text-slate-400" />
      </span>
      <input type="file" accept="video/*" multiple={multiple} onChange={event => onChange(event.target.files)} className="block w-full text-sm font-semibold text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-slate-900 file:px-3 file:py-2 file:text-sm file:font-bold file:text-white" />
      <div className="mt-3 flex flex-wrap gap-1.5">
        {files.length === 0 ? <EvidenceBadge>未选择</EvidenceBadge> : files.map(file => <EvidenceBadge key={`${file.name}-${file.size}`} tone="blue">{file.name}</EvidenceBadge>)}
      </div>
    </label>
  )
}

function CheckRow({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <CheckCircle2 className={`h-4 w-4 ${ok ? 'text-emerald-600' : 'text-slate-300'}`} />
      {label}
    </div>
  )
}
