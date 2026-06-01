import { useMemo, useState, type ReactNode } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowLeft, CheckCircle2, Circle, ClipboardCheck, FileVideo2, Loader2, UploadCloud } from 'lucide-react'
import { experimentApi } from '../api'
import { EvidenceBadge, EvidenceCard, primaryButtonClass, secondaryButtonClass } from '../components/EvidenceUI'

type Slot = 'first' | 'third'
const uploadTimingStoragePrefix = 'realityloop-upload-e2e:'

export default function Upload() {
  const navigate = useNavigate()
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [contextText, setContextText] = useState('')
  const [protocolText, setProtocolText] = useState('')
  const [sessionStartTime, setSessionStartTime] = useState('')
  const [files, setFiles] = useState<Record<Slot, File[]>>({ first: [], third: [] })
  const [progress, setProgress] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const allFiles = useMemo(() => [...files.first, ...files.third], [files])
  const hasTitle = Boolean(title.trim())
  const hasFirstVideo = files.first.length > 0
  const hasThirdVideo = files.third.length > 0
  const readyCount = [hasTitle, hasFirstVideo, hasThirdVideo].filter(Boolean).length
  const isReady = readyCount === 3
  const canSubmit = isReady && !submitting

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
    <div className="space-y-4">
      <section className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/90 px-4 py-3 shadow-[var(--ui-shadow-subtle)] sm:px-5">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="min-w-0">
            <Link to="/experiments" className="text-sm font-semibold text-[color:var(--ui-accent)] hover:text-[color:var(--ui-text)]">
              实验
            </Link>
            <h2 className="mt-0.5 text-2xl font-semibold tracking-tight text-[color:var(--ui-text)]">
              新建实验
            </h2>
            <p className="mt-1 max-w-3xl text-sm font-medium leading-6 text-[color:var(--ui-text-muted)]">
              填写实验信息并导入第一视角、第三视角视频，创建后系统会自动开始分析。
            </p>
          </div>
          <Link to="/experiments" className={secondaryButtonClass()}>
            <ArrowLeft className="h-4 w-4" />
            返回实验
          </Link>
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_20rem]">
        <EvidenceCard className="p-4 sm:p-5">
          <FormSection index="01" title="实验信息" description="给分析任务一个可检索、可复盘的上下文。">
            <div className="grid gap-3 lg:grid-cols-3">
              <label className="block">
                <span className="mb-2 block text-sm font-medium text-[color:var(--ui-text)]">实验名称</span>
                <input
                  value={title}
                  onChange={event => setTitle(event.target.value)}
                  className="h-10 w-full rounded-lg border border-[color:var(--ui-border)] bg-white px-3 text-sm outline-none transition focus:border-[color:var(--ui-accent)] focus:ring-2 focus:ring-[color:var(--ui-accent-soft)]"
                  placeholder="例如：溶剂更换流程"
                />
              </label>
              <label className="block">
                <span className="mb-2 block text-sm font-medium text-[color:var(--ui-text)]">实验描述</span>
                <input
                  value={description}
                  onChange={event => setDescription(event.target.value)}
                  className="h-10 w-full rounded-lg border border-[color:var(--ui-border)] bg-white px-3 text-sm outline-none transition focus:border-[color:var(--ui-accent)] focus:ring-2 focus:ring-[color:var(--ui-accent-soft)]"
                  placeholder="批次、样本或流程说明"
                />
              </label>
              <label className="block">
                <span className="mb-2 block text-sm font-medium text-[color:var(--ui-text)]">全局开始时间</span>
                <input
                  type="datetime-local"
                  value={sessionStartTime}
                  onChange={event => setSessionStartTime(event.target.value)}
                  className="h-10 w-full rounded-lg border border-[color:var(--ui-border)] bg-white px-3 text-sm outline-none transition focus:border-[color:var(--ui-accent)] focus:ring-2 focus:ring-[color:var(--ui-accent-soft)]"
                />
              </label>
            </div>
          </FormSection>

          <FormSection index="02" title="视频素材" description="两个必需视角会一起进入实验分析链路。">
            <div className="grid gap-3 lg:grid-cols-2">
              <UploadSlot title="第一视角" helper="操作者视角" files={files.first} onChange={list => updateFiles('first', list)} />
              <UploadSlot title="第三视角" helper="台面场景" files={files.third} onChange={list => updateFiles('third', list)} />
            </div>
          </FormSection>

          <FormSection index="03" title="补充说明" description="可选内容会作为分析上下文提交。">
            <div className="grid gap-3 lg:grid-cols-2">
              <label className="block">
                <span className="mb-2 block text-sm font-medium text-[color:var(--ui-text)]">实验上下文（可选）</span>
                <textarea
                  value={contextText}
                  onChange={event => setContextText(event.target.value)}
                  rows={3}
                  className="w-full resize-none rounded-lg border border-[color:var(--ui-border)] bg-white px-3 py-2 text-sm outline-none transition focus:border-[color:var(--ui-accent)] focus:ring-2 focus:ring-[color:var(--ui-accent-soft)]"
                  placeholder="实验样本、参数、已知异常"
                />
              </label>
              <label className="block">
                <span className="mb-2 block text-sm font-medium text-[color:var(--ui-text)]">SOP / 协议（可选）</span>
                <textarea
                  value={protocolText}
                  onChange={event => setProtocolText(event.target.value)}
                  rows={3}
                  className="w-full resize-none rounded-lg border border-[color:var(--ui-border)] bg-white px-3 py-2 text-sm outline-none transition focus:border-[color:var(--ui-accent)] focus:ring-2 focus:ring-[color:var(--ui-accent-soft)]"
                  placeholder="步骤要求、检查项、结果输出约束"
                />
              </label>
            </div>
          </FormSection>

          {error && <div className="mt-4 rounded-lg bg-red-50 p-3 text-sm font-medium text-red-700">{error}</div>}
        </EvidenceCard>

        <aside className="xl:sticky xl:top-24 xl:self-start">
          <EvidenceCard className="p-4">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <ClipboardCheck className="h-4 w-4 text-[color:var(--ui-accent)]" />
                <h3 className="font-semibold text-[color:var(--ui-text)]">就绪状态</h3>
              </div>
              <EvidenceBadge tone={isReady ? 'success' : 'slate'}>{readyCount}/3</EvidenceBadge>
            </div>

            <div className="mt-4 space-y-2.5 text-sm">
              <CheckRow ok={hasTitle} label="实验名称" />
              <CheckRow ok={hasFirstVideo} label="第一视角视频" />
              <CheckRow ok={hasThirdVideo} label="第三视角视频" />
            </div>

            <button
              type="button"
              disabled={!canSubmit}
              onClick={() => void submit()}
              className={`${primaryButtonClass('primary')} mt-4 w-full disabled:cursor-not-allowed disabled:opacity-50`}
            >
              {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <UploadCloud className="h-4 w-4" />}
              创建实验并开始分析
            </button>

            {submitting && (
              <div className="mt-4 rounded-lg border border-[color:var(--ui-accent-soft)] bg-[color:var(--ui-accent-soft)] p-3">
                <div className="mb-2 flex justify-between text-xs font-medium text-[color:var(--ui-text-muted)]">
                  <span>上传并启动分析</span>
                  <span>{progress}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-white">
                  <div className="h-full rounded-full bg-[color:var(--ui-accent)] transition-all" style={{ width: `${progress}%` }} />
                </div>
              </div>
            )}

            <div className="mt-4 rounded-lg border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] p-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-[color:var(--ui-text-muted)]">待上传视频</div>
              <div className="mt-2 text-xl font-semibold text-[color:var(--ui-text)]">{allFiles.length}</div>
              <div className="mt-1 text-xs leading-5 text-[color:var(--ui-text-muted)]">
                {allFiles.length > 0 ? `${formatBytes(allFiles.reduce((sum, file) => sum + file.size, 0))}` : '尚未选择文件'}
              </div>
            </div>

            <div className={`mt-3 rounded-lg border p-3 ${isReady ? 'border-cyan-200 bg-cyan-50' : 'border-[color:var(--ui-border)] bg-white'}`}>
              <div className={`text-sm font-semibold ${isReady ? 'text-cyan-800' : 'text-[color:var(--ui-text)]'}`}>
                {isReady ? '可以开始' : '请完善必填项'}
              </div>
              <p className="mt-1 text-xs leading-5 text-[color:var(--ui-text-muted)]">
                {isReady
                  ? '提交后会创建实验、上传双视角视频，并自动进入分析流程。'
                  : '需要填写实验名称，并选择第一视角和第三视角视频。'}
              </p>
            </div>
          </EvidenceCard>
        </aside>
      </div>
    </div>
  )
}

function FormSection({ index, title, description, children }: { index: string; title: string; description: string; children: ReactNode }) {
  return (
    <section className="border-b border-[color:var(--ui-border)] py-3 first:pt-0 last:border-b-0 last:pb-0">
      <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="rounded-md bg-[color:var(--ui-accent-soft)] px-2 py-1 text-xs font-semibold text-[color:var(--ui-accent)]">{index}</span>
            <h3 className="text-base font-semibold text-[color:var(--ui-text)]">{title}</h3>
          </div>
          <p className="mt-1 hidden text-sm leading-5 text-[color:var(--ui-text-muted)] 2xl:block">{description}</p>
        </div>
      </div>
      {children}
    </section>
  )
}

function UploadSlot({ title, helper, files, onChange }: { title: string; helper: string; files: File[]; onChange: (files: FileList | null) => void }) {
  const file = files[0]
  const selected = Boolean(file)

  return (
    <label className={`group block cursor-pointer rounded-lg border border-dashed p-3 transition ${selected ? 'border-[color:var(--ui-accent)] bg-[color:var(--ui-accent-soft)]' : 'border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] hover:border-[color:var(--ui-accent-weak)] hover:bg-white'}`}>
      <span className="flex items-start justify-between gap-3">
        <span className="min-w-0">
          <span className="block text-base font-semibold text-[color:var(--ui-text)]">{title}</span>
          <span className="block text-sm text-[color:var(--ui-text-muted)]">{helper}</span>
        </span>
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white text-[color:var(--ui-accent)] shadow-[var(--ui-shadow-subtle)]">
          <FileVideo2 className="h-4 w-4" />
        </span>
      </span>
      <input
        type="file"
        accept="video/*"
        onChange={event => onChange(event.target.files)}
        className="sr-only"
      />
      <span className="mt-3 flex items-center justify-between gap-3">
        <span className="min-w-0 text-sm font-medium text-[color:var(--ui-text)]">
          {file ? <span className="block truncate">{file.name}</span> : '未选择文件'}
        </span>
        <span className="shrink-0 rounded-md bg-[color:var(--ui-accent)] px-3 py-1.5 text-sm font-medium text-white">
          {file ? '更换文件' : '选择文件'}
        </span>
      </span>
      {file && (
        <span className="mt-3 flex flex-wrap gap-2">
          <EvidenceBadge tone="success">已选择</EvidenceBadge>
          <EvidenceBadge>{formatBytes(file.size)}</EvidenceBadge>
        </span>
      )}
    </label>
  )
}

function CheckRow({ ok, label }: { ok: boolean; label: string }) {
  const Icon = ok ? CheckCircle2 : Circle
  return (
    <div className="flex items-center gap-2 text-[color:var(--ui-text-muted)]">
      <Icon className={`h-4 w-4 ${ok ? 'text-[color:var(--ui-success)]' : 'text-[color:var(--ui-text-muted)]'}`} />
      <span>{label}</span>
    </div>
  )
}

function formatBytes(bytes: number) {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / 1024 ** index
  return `${value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[index]}`
}
