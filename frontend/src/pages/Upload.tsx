import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowLeft, CheckCircle2, FileVideo2, Loader2, UploadCloud } from 'lucide-react'
import { experimentApi } from '../api'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, PageHero, primaryButtonClass, secondaryButtonClass } from '../components/EvidenceUI'

type Slot = 'first' | 'third' | 'top' | 'bottom'
const uploadTimingStoragePrefix = 'realityloop-upload-e2e:'

export default function Upload() {
  const navigate = useNavigate()
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [contextText, setContextText] = useState('')
  const [protocolText, setProtocolText] = useState('')
  const [sessionStartTime, setSessionStartTime] = useState('')
  const [files, setFiles] = useState<Record<Slot, File[]>>({ first: [], third: [], top: [], bottom: [] })
  const [progress, setProgress] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const allFiles = useMemo(() => [...files.first, ...files.third, ...files.top, ...files.bottom], [files])
  const canSubmit = Boolean(title.trim()) && files.first.length > 0 && files.third.length > 0 && !submitting

  function updateFiles(slot: Slot, nextFiles: FileList | null) {
    setFiles(previous => ({ ...previous, [slot]: Array.from(nextFiles || []) }))
  }

  async function submit() {
    if (!canSubmit) return
    const submitStartedAtMs = Date.now()
    setSubmitting(true)
    setError(null)
    setProgress(0)
    try {
      const created = await experimentApi.create({
        title: title.trim(),
        description: description.trim(),
        context_text: contextText.trim() || undefined,
        protocol_text: protocolText.trim() || undefined,
      })
      try {
        window.sessionStorage.setItem(
          `${uploadTimingStoragePrefix}${created.experiment_id}`,
          JSON.stringify({
            startedAtMs: submitStartedAtMs,
            startedAtIso: new Date(submitStartedAtMs).toISOString(),
            fileCount: allFiles.length,
            totalBytes: allFiles.reduce((sum, file) => sum + file.size, 0),
          }),
        )
      } catch {
        // 会话级耗时仅用于页面展示，不作为后台真值。
      }
      await experimentApi.uploadAndAnalyze(
        created.experiment_id,
        {
          firstPersonVideo: files.first[0] || null,
          thirdPersonVideo: files.third[0] || null,
          topVideo: files.top[0] || null,
          bottomVideo: files.bottom[0] || null,
          sessionStartTime: sessionStartTime || null,
        },
        setProgress,
      )
      navigate(`/experiments/${created.experiment_id}/workspace#result-materials`)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '上传分析失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-5">
      <PageHero
        eyebrow={<Link to="/experiments" className="hover:text-[color:var(--ui-text)]">实验</Link>}
        title="开始分析"
        description="上传第一视角和第三视角视频后，系统将自动启动实验窗口识别与材料生成。"
        actions={(
          <Link to="/experiments" className={secondaryButtonClass()}>
            <ArrowLeft className="h-4 w-4" />
            返回实验
          </Link>
        )}
      />

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_22rem]">
        <EvidenceCard className="p-5">
          <div className="grid gap-4 lg:grid-cols-2">
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-[color:var(--ui-text-muted)]">实验名称</span>
              <input
                value={title}
                onChange={event => setTitle(event.target.value)}
                className="w-full rounded-lg border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] px-3 py-2 text-sm outline-none focus:border-[color:var(--ui-accent)]"
                placeholder="例如：溶剂更换流程"
              />
            </label>
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-[color:var(--ui-text-muted)]">实验描述</span>
              <input
                value={description}
                onChange={event => setDescription(event.target.value)}
                className="w-full rounded-lg border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] px-3 py-2 text-sm outline-none focus:border-[color:var(--ui-accent)]"
                placeholder="批次、样本或流程说明"
              />
            </label>
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-[color:var(--ui-text-muted)]">全局开始时间</span>
              <input
                type="datetime-local"
                value={sessionStartTime}
                onChange={event => setSessionStartTime(event.target.value)}
                className="w-full rounded-lg border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] px-3 py-2 text-sm outline-none focus:border-[color:var(--ui-accent)]"
              />
            </label>
            <div className="rounded-lg border border-[color:var(--ui-accent-soft)] bg-[color:var(--ui-accent-soft)] p-3 text-sm text-[color:var(--ui-text-muted)]">
              未填写开始时间时，系统仍可按视频内置时序完成分析。
            </div>
          </div>

          <div className="mt-5 grid gap-4 lg:grid-cols-4">
            <UploadSlot title="第一视角" helper="操作者视角" files={files.first} onChange={list => updateFiles('first', list)} />
            <UploadSlot title="第三视角" helper="台面场景" files={files.third} onChange={list => updateFiles('third', list)} />
            <UploadSlot title="顶部视角" helper="可选辅助视角" files={files.top} onChange={list => updateFiles('top', list)} />
            <UploadSlot title="底部视角" helper="可选补充视角" files={files.bottom} onChange={list => updateFiles('bottom', list)} />
          </div>

          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-[color:var(--ui-text-muted)]">实验上下文（可选）</span>
              <textarea
                value={contextText}
                onChange={event => setContextText(event.target.value)}
                rows={5}
                className="w-full resize-none rounded-lg border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] px-3 py-2 text-sm outline-none focus:border-[color:var(--ui-accent)]"
                placeholder="实验样本、参数、已知异常"
              />
            </label>
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-[color:var(--ui-text-muted)]">SOP / 协议（可选）</span>
              <textarea
                value={protocolText}
                onChange={event => setProtocolText(event.target.value)}
                rows={5}
                className="w-full resize-none rounded-lg border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] px-3 py-2 text-sm outline-none focus:border-[color:var(--ui-accent)]"
                placeholder="步骤要求、检查项、结果输出约束"
              />
            </label>
          </div>

          {error && <div className="mt-4 rounded-lg bg-red-50 p-3 text-sm font-medium text-red-700">{error}</div>}
          {submitting && (
            <div className="mt-4">
              <div className="mb-2 flex justify-between text-sm font-medium text-[color:var(--ui-text-muted)]"><span>开始分析</span><span>{progress}%</span></div>
              <div className="h-2 overflow-hidden rounded-full bg-[color:var(--ui-bg-muted)]"><div className="h-full rounded-full bg-[color:var(--ui-accent)] transition-all" style={{ width: `${progress}%` }} /></div>
            </div>
          )}
        </EvidenceCard>

        <EvidenceCard className="p-5">
          <div className="mb-4 flex items-center gap-2">
            <UploadCloud className="h-4 w-4 text-[color:var(--ui-accent)]" />
            <h3 className="font-medium text-[color:var(--ui-text)]">导入校验</h3>
          </div>
          <div className="space-y-3 text-sm text-[color:var(--ui-text-muted)]">
            <CheckRow ok={Boolean(title.trim())} label="实验名称已填写" />
            <CheckRow ok={files.first.length > 0} label="第一视角视频" />
            <CheckRow ok={files.third.length > 0} label="第三视角视频" />
            <CheckRow ok={allFiles.length > 0} label={`${allFiles.length} 个视频待上传`} />
          </div>
          <button
            type="button"
            disabled={!canSubmit}
            onClick={() => void submit()}
            className={`${primaryButtonClass('primary')} mt-5 w-full disabled:cursor-not-allowed disabled:opacity-50`}
          >
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <UploadCloud className="h-4 w-4" />}
            开始分析
          </button>
          <EmptyEvidence title="请先完善配置" description="需要填写实验名称，且至少包含第一视角和第三视角视频后才可开始分析。" />
        </EvidenceCard>
      </div>
    </div>
  )
}

function UploadSlot({ title, helper, files, onChange, multiple = false }: { title: string; helper: string; files: File[]; onChange: (files: FileList | null) => void; multiple?: boolean }) {
  return (
    <label className="block rounded-lg border border-dashed border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] p-4 transition">
      <span className="mb-3 flex items-center justify-between gap-2">
        <span>
          <span className="block font-medium text-[color:var(--ui-text)]">{title}</span>
          <span className="block text-xs text-[color:var(--ui-text-muted)]">{helper}</span>
        </span>
        <FileVideo2 className="h-5 w-5 text-[color:var(--ui-text-muted)]" />
      </span>
      <input
        type="file"
        accept="video/*"
        multiple={multiple}
        onChange={event => onChange(event.target.files)}
        className="block w-full text-sm font-medium text-[color:var(--ui-text-muted)] file:mr-3 file:rounded-md file:border-0 file:bg-[color:var(--ui-accent)] file:px-3 file:py-2 file:text-sm file:font-medium file:text-white"
      />
      <div className="mt-3 flex flex-wrap gap-1.5">
        {files.length === 0 ? <EvidenceBadge>未选择</EvidenceBadge> : files.map(file => <EvidenceBadge key={`${file.name}-${file.size}`}>{file.name}</EvidenceBadge>)}
      </div>
    </label>
  )
}

function CheckRow({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <CheckCircle2 className={`h-4 w-4 ${ok ? 'text-[color:var(--ui-success)]' : 'text-[color:var(--ui-text-muted)]'}`} />
      <span>{label}</span>
    </div>
  )
}
