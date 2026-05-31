import { useEffect, useMemo, useState } from 'react'
import { Link, useLocation, useParams } from 'react-router-dom'
import { ArrowLeft, CheckCircle2, ChevronRight, FileText, Film, RefreshCw, Search } from 'lucide-react'
import { experimentApi, prefetchExperimentRoute } from '../api'
import AnalysisTimingSummary from '../components/AnalysisTimingSummary'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, primaryButtonClass, secondaryButtonClass, toneForStatus } from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import { experimentFileUrl } from '../mediaUrl'
import type {
  AnalysisOverview,
  MaterialCandidateFile,
  MaterialCandidateGroup,
  MaterialDiagnosticsResponse,
  MaterialSearchItem,
} from '../types'

const DEMO_MODES = new Set(['1', 'true', 'yes'])

function formatHHMMSS(value?: number | null) {
  const seconds = Number.isFinite(Number(value)) ? Math.max(0, Math.floor(Number(value))) : 0
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function formatHHMMSSFromUs(value?: number | null) {
  if (!Number.isFinite(Number(value))) return '-'
  const v = Number(value)
  if (v > 1_000_000) return formatHHMMSS(v / 1_000_000)
  return formatHHMMSS(v)
}

function finiteNumber(value?: unknown) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function formatRangeText(start?: unknown, end?: unknown) {
  const startSec = finiteNumber(start)
  const endSec = finiteNumber(end)
  if (startSec == null) return '-'
  return `${formatHHMMSS(startSec)} - ${endSec == null ? '-' : formatHHMMSS(endSec)}`
}

function safeText(value?: unknown, fallback = '') {
  if (value == null) return fallback
  const text = String(value).trim()
  return text || fallback
}

function actionLabelFromGroup(group: MaterialCandidateGroup) {
  return cleanDisplayText(
    group.canonical_action_type,
    group.action_name,
    group.semantic_action,
    group.interaction_family,
    '待确认',
  )
}

function canonicalConfidence(group: MaterialCandidateGroup) {
  return finiteNumber(
    group.quality_score ??
      group.payload?.quality_score ??
      group.quality_reasons?.length ??
      undefined,
  ) || 0
}

function windowLabel(group: MaterialCandidateGroup) {
  return safeText(
    group.parent_segment_id ||
    group.micro_segment_id ||
    group.video_id ||
    group.candidate_group_id,
    '未归属',
  )
}

function candidateAnchorTs(group: MaterialCandidateGroup) {
  const firstFile = [...group.keyframes, ...group.clips, ...group.files].find(Boolean) as MaterialCandidateFile | undefined
  const value = firstFile ? finiteNumber(firstFile.timestamp_sec ?? firstFile.local_timestamp_sec ?? firstFile.time_start) : null
  return value == null ? '-' : formatHHMMSS(value)
}

function candidateId(group: MaterialCandidateGroup) {
  return group.candidate_group_id || group.parent_segment_id || `candidate-${group.display_title || Math.random().toString(36).slice(2)}`
}

function candidateVideoOrFrame(file: MaterialCandidateFile, experimentId?: string) {
  return experimentFileUrl(
    safeText(file.clip_url) ||
      safeText(file.clip_file_path) ||
      safeText(file.preview_url) ||
      safeText(file.frame_path) ||
      safeText(file.url),
    experimentId,
  )
}

function isImageUrl(url?: string | null) {
  if (!url) return false
  return /\.(png|jpg|jpeg|webp|gif|bmp|avif)$/i.test(url) || url.includes('image=') || url.includes('frame')
}

function evidenceReasonText(group: MaterialCandidateGroup) {
  const candidates: unknown[] = [
    group.review_reason_codes,
    group.quality_reasons,
    group.reason,
    group.approval_reason,
    group.rejection_reason,
    group.decision_reason,
    group.missing_reason,
    group.review_reason,
  ]
  const flatten = candidates
    .flatMap(item => {
      if (!item) return []
      if (Array.isArray(item)) return item
      return [item]
    })
    .map(item => safeText(item))
    .filter(Boolean)
  return flatten.length ? flatten.join(' / ') : '-'
}

export default function MaterialSearch() {
  const { id } = useParams<{ id: string }>()
  const location = useLocation()
  const [overview, setOverview] = useState<AnalysisOverview | null>(null)
  const [windows, setWindows] = useState<Array<Record<string, unknown>>>([])
  const [materials, setMaterials] = useState<MaterialSearchItem[]>([])
  const [candidates, setCandidates] = useState<MaterialCandidateGroup[]>([])
  const [diagnostics, setDiagnostics] = useState<MaterialDiagnosticsResponse | null>(null)
  const [busyAction, setBusyAction] = useState<Record<string, string>>({})
  const [detailsOpen, setDetailsOpen] = useState<Record<string, boolean>>({})
  const [renaming, setRenaming] = useState<Record<string, boolean>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showCandidateDetails, setShowCandidateDetails] = useState<Record<string, boolean>>({})

  const searchParams = new URLSearchParams(location.search)
  const demoMode = DEMO_MODES.has(String(searchParams.get('demo') || '').toLowerCase()) || Boolean(localStorage.getItem('realityloop-demo-mode'))
  const runAdvanced = String(searchParams.get('advanced') || '').toLowerCase() === '1' && Boolean(diagnostics)

  const totalOfficial = materials.length
  const totalCandidateGroups = candidates.length

  async function load(force = false) {
    if (!id) return
    setLoading(true)
    setError(null)
    try {
      const [nextOverview, nextWindows, nextPublished, nextCandidates, nextDiagnostics] = await Promise.all([
        experimentApi.getAnalysisOverview(id, { force }),
        experimentApi.getSubExperiments(id, { force }).catch(() => null),
        experimentApi.getPublishedMaterials(id, { limit: 500 }, { force }),
        experimentApi.getMaterialCandidates(id, { force }),
        force ? experimentApi.getMaterialDiagnostics(id) : Promise.resolve(null),
      ])
      const published = nextPublished
      const candidateItems = nextCandidates
      setOverview(nextOverview)
      setWindows(Array.isArray((nextWindows as { segments?: unknown[] } | null)?.segments) ? (nextWindows as { segments: unknown[] }).segments : [])
      setMaterials(Array.isArray((published as { items?: unknown[] }).items) ? (published.items as MaterialSearchItem[]).filter(item => item.review_status !== 'rejected') : [])
      setCandidates(Array.isArray((candidateItems as { items?: unknown[] }).items) ? (candidateItems.items as MaterialCandidateGroup[]) : [])
      setDiagnostics(runAdvanced ? nextDiagnostics : null)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '材料页读取失败')
      setWindows([])
      setMaterials([])
      setCandidates([])
      setDiagnostics(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load(true)
  }, [id, location.search])

  const experimentId = id || ''
  const filteredCandidates = useMemo(() => {
    const keyword = query.trim().toLowerCase()
    if (!keyword) return candidates
    return candidates.filter(item => {
      const source = [item.display_title, item.candidate_group_id, item.parent_segment_id, item.interaction_family, item.canonical_action_type, item.action_name]
        .filter(Boolean)
        .map(item => String(item).toLowerCase())
        .join(' ')
      return source.includes(keyword)
    })
  }, [candidates, query])

  const statusTone = toneForStatus(overview?.run.status)
  const windowCount = windows.length
  const officialCount = totalOfficial
  const candidateCount = totalCandidateGroups

  async function performAction(groupId: string, action: 'confirm' | 'upgrade' | 'reject' | 'rename', payload?: { display_title?: string }) {
    if (!id) return
    setBusyAction(previous => ({ ...previous, [groupId]: action }))
    try {
      if (action === 'confirm') {
        await experimentApi.confirmMaterialCandidate(id, groupId)
      }
      if (action === 'upgrade') {
        await experimentApi.approveMaterialCandidate(id, groupId)
      }
      if (action === 'reject') {
        await experimentApi.decideMaterialCandidate(id, groupId, {
          decision: 'false_positive',
          reason_code: 'rejected',
          reason: '系统判定暂不采用',
        })
      }
      if (action === 'rename' && payload?.display_title) {
        await experimentApi.renameMaterialCandidate(id, groupId, { display_title: payload.display_title })
      }
      await load(true)
    } finally {
      setBusyAction(previous => {
        const next = { ...previous }
        delete next[groupId]
        return next
      })
      setRenaming(previous => {
        const next = { ...previous }
        delete next[groupId]
        return next
      })
    }
  }

  function onRename(group: MaterialCandidateGroup) {
    if (!id) return
    const idValue = candidateId(group)
    const current = safeText(group.display_title, actionLabelFromGroup(group))
    const next = window.prompt('请输入重命名标题', current)
    if (!next || !next.trim()) return
    setRenaming(previous => ({ ...previous, [idValue]: true }))
    void performAction(idValue, 'rename', { display_title: next.trim() })
  }

  return (
    <div className="space-y-6">
      <PageHero
        eyebrow={<Link to="/experiments" className="hover:text-[color:var(--ui-text)]">实验</Link>}
        title={overview ? cleanDisplayText(overview.experiment.experiment_name, `实验 ${id}`) : '实验材料'}
        description={overview ? `关键素材库：先看窗口，再判断待确认素材。当前状态 ${toneForStatus(overview.run.status)}` : '先看窗口，再判断待确认素材。结果导向，默认不展示技术细节。'}
        actions={
          <>
            <Link to={`/experiments/${experimentId}/workspace`} onMouseEnter={() => prefetchExperimentRoute(experimentId, 'workspace')} className={secondaryButtonClass()}>
              <ArrowLeft className="h-4 w-4" />
              回到窗口
            </Link>
            <button type="button" onClick={() => void load(true)} className={secondaryButtonClass()}>
              <RefreshCw className="h-4 w-4" />
              刷新
            </button>
            <button
              type="button"
              onClick={() => {
                setShowAdvanced(value => !value)
              }}
              className={secondaryButtonClass(showAdvanced ? 'warning' : 'primary')}
            >
              {showAdvanced ? '关闭高级信息' : '高级信息'}
            </button>
          </>
        }
      />

      {loading && <EmptyEvidence title="加载中" description="正在读取窗口与候选，稍等片刻后会自动展示核心结果。" />}
      {error && <EmptyEvidence title="加载失败" description={error} />}
      {!loading && !error && (
        <>
          <section className="grid gap-3 sm:grid-cols-3">
            <MetricTile label="窗口" value={windowCount} tone="primary" helper="pipeline 结果" />
            <MetricTile label="关键素材" value={officialCount} tone="success" helper="可直接进入证据片段" />
            <MetricTile label="待确认素材" value={candidateCount} tone="warning" helper="可逐条确认或排除" />
          </section>

          {overview && (
            <AnalysisTimingSummary
              run={overview.run}
              statusLabel={cleanDisplayText(overview.run.message, overview.run.status)}
              clientEndToEndSec={overview.run.timing?.elapsed_sec}
            />
          )}

          <section>
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-[color:var(--ui-text)]">窗口（pipeline 结果）</h2>
            </div>
            <div className="grid gap-4 xl:grid-cols-2">
              {windows.length === 0 && <EmptyEvidence title="暂无窗口" description="尚未生成窗口结果。请先在分析完成后重试刷新。" />}
              {windows.map(window => {
                const sourceWindow = window as Record<string, unknown>
                const windowId = safeText(sourceWindow.segment_id as string, sourceWindow.sub_id as string, `窗口-${sourceWindow.index || '-'}`)
                const windowStatus = String(sourceWindow.formal_action_status || 'analysis')
                const windowRange = formatRangeText(sourceWindow.start_sec as number, sourceWindow.end_sec as number)
                const previewUrl = experimentFileUrl(safeText(
                  sourceWindow.preview_video_url as string,
                  sourceWindow.sample_grid_url as string,
                  sourceWindow.window_preview_url as string,
                  sourceWindow.first_person_preview_url as string,
                  sourceWindow.first_preview_url as string,
                ), overview?.experiment.experiment_id)
                const keyCount = Number(sourceWindow.micro_segment_count || 0)
                return (
                  <EvidenceCard key={windowId} className="p-4">
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="text-sm font-medium text-[color:var(--ui-text)]">{windowId}</h3>
                      <EvidenceBadge tone={toneForStatus(windowStatus)}>{statusTone}</EvidenceBadge>
                    </div>
                    <p className="mt-1 text-xs text-[color:var(--ui-text-muted)]">时间锚点：{windowRange}</p>
                    <p className="mt-1 text-xs text-[color:var(--ui-text-muted)]">关键片段数：{keyCount}</p>

                    <div className="mt-3 rounded-lg bg-[color:var(--ui-bg-muted)] p-2">
                      <div className="relative aspect-video w-full overflow-hidden rounded-md bg-[color:var(--ui-bg)]">
                        {previewUrl ? (
                          <video
                            src={previewUrl}
                            controls
                            playsInline
                            preload="metadata"
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <div className="absolute inset-0 animate-pulse bg-gradient-to-br from-[color:var(--ui-bg-muted)] to-white" />
                        )}
                      </div>
                    </div>
                  </EvidenceCard>
                )
              })}
            </div>
          </section>

          <section id="pending-materials">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-[color:var(--ui-text)]">待确认素材</h2>
              <label className="text-sm text-[color:var(--ui-text-muted)]">
                <span className="sr-only">候选搜索</span>
                <span className="inline-flex items-center gap-2 rounded-md border border-[color:var(--ui-border)] px-3 py-1.5 text-xs"> 
                  <Search className="h-3.5 w-3.5" />
                  <input
                    value={query}
                    onChange={event => setQuery(event.target.value)}
                    placeholder="按窗口编号 / 动作类型筛选"
                    className="h-7 border-0 bg-transparent p-0 text-xs text-[color:var(--ui-text)] placeholder:text-[color:var(--ui-text-muted)] outline-none"
                  />
                </span>
              </label>
            </div>

            {filteredCandidates.length === 0 ? (
              <EmptyEvidence title="暂无候选" description="当前窗口未产生待确认素材；可直接转向关键素材库复核。" />
            ) : (
              <div className="space-y-3">
                {filteredCandidates.map(group => {
                  const gid = candidateId(group)
                  const isBusy = Boolean(busyAction[gid])
                  const detailKey = `${gid}:detail`
                  const firstFile = [...group.keyframes, ...group.files, ...group.clips].find(Boolean) as MaterialCandidateFile | undefined
                  const mediaUrl = candidateVideoOrFrame(firstFile || {}, overview?.experiment.experiment_id)
                  const isMovie = Boolean(firstFile && (firstFile.clip_url || firstFile.clip_file_path))
                  const previewMode = isMovie ? 'clip' : 'frame'
                  const confidence = Number.isFinite(canonicalConfidence(group)) ? canonicalConfidence(group).toFixed(3) : '-'
                  const detailsOpen = demoMode ? false : Boolean(showCandidateDetails[gid])
                  const qualityReasons = evidenceReasonText(group)
                  return (
                    <EvidenceCard key={gid} className="p-4">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <EvidenceBadge tone={group.review_status ? toneForStatus(group.review_status) : 'warning'}>{actionLabelFromGroup(group)}</EvidenceBadge>
                            <EvidenceBadge tone="slate">动作类型</EvidenceBadge>
                          </div>
                          <h3 className="mt-2 text-sm font-semibold text-[color:var(--ui-text)]">{safeText(group.display_title, windowLabel(group))}</h3>
                          <p className="mt-1 text-xs text-[color:var(--ui-text-muted)]">窗口编号：{windowLabel(group)}</p>
                          <p className="mt-1 text-xs text-[color:var(--ui-text-muted)]">时间锚点：{candidateAnchorTs(group)} · 关键帧：{previewMode}</p>
                          <p className="mt-1 text-xs text-[color:var(--ui-text-muted)]">置信度：{confidence}</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => {
                            if (demoMode) return
                            setShowCandidateDetails(current => ({ ...current, [gid]: !detailsOpen }))
                          }}
                          className="inline-flex min-h-8 items-center gap-1 rounded-md border border-[color:var(--ui-border)] px-2.5 py-1 text-xs text-[color:var(--ui-text-muted)]"
                        >
                          <ChevronRight className={`h-3.5 w-3.5 transition-transform ${detailsOpen ? 'rotate-90' : ''}`} />
                          展开详情
                        </button>
                      </div>

                      <div className="mt-3 rounded-lg bg-[color:var(--ui-bg-muted)]">
                        <div className="relative aspect-video w-full overflow-hidden rounded-md">
                          {mediaUrl ? (
                            isMovie ? (
                              <video src={mediaUrl} controls preload="metadata" playsInline className="h-full w-full bg-[color:var(--ui-bg)] object-contain" />
                            ) : (
                              <img src={mediaUrl} alt="待确认素材关键帧" className="h-full w-full object-cover" />
                            )
                          ) : (
                            <div className="absolute inset-0 animate-pulse bg-gradient-to-br from-[color:var(--ui-bg-muted)] to-white" />
                          )}
                        </div>
                      </div>

                      {detailsOpen && !demoMode && (
                        <div className="mt-3 grid gap-2 rounded-lg border border-[color:var(--ui-border)] bg-white p-3 text-xs text-[color:var(--ui-text-muted)]">
                          <p><span className="font-medium text-[color:var(--ui-text)]">阈值： </span>{safeText(group.quality_label, '-')} / {safeText(group.quality_tier, '-')} / {safeText(group.quality_grade, '-')}</p>
                          <p><span className="font-medium text-[color:var(--ui-text)]">质量原因： </span>{qualityReasons}</p>
                          <p><span className="font-medium text-[color:var(--ui-text)]">证据片段数： </span>{group.files?.length || 0}</p>
                          <p><span className="font-medium text-[color:var(--ui-text)]">窗口编号： </span>{windowLabel(group)}</p>
                        </div>
                      )}

                      <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => void performAction(gid, 'confirm')}
                          className={`${primaryButtonClass('success')} disabled:cursor-not-allowed disabled:opacity-50`}
                        >
                          <CheckCircle2 className="h-4 w-4" />
                          {isBusy ? '处理中' : '确认'}
                        </button>
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => void performAction(gid, 'upgrade')}
                          className={`${secondaryButtonClass()} disabled:cursor-not-allowed disabled:opacity-50`}
                        >
                          <CheckCircle2 className="h-4 w-4" />
                          {isBusy ? '处理中' : '加入关键素材'}
                        </button>
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => void performAction(gid, 'reject')}
                          className={`${secondaryButtonClass('warning')} disabled:cursor-not-allowed disabled:opacity-50`}
                        >
                          <FileText className="h-4 w-4" />
                          {isBusy ? '处理中' : '暂不采用'}
                        </button>
                        <button
                          type="button"
                          disabled={renaming[gid]}
                          onClick={() => onRename(group)}
                          className={`${secondaryButtonClass()} disabled:cursor-not-allowed disabled:opacity-50`}
                        >
                          <Film className="h-4 w-4" />
                          {renaming[gid] ? '处理中' : '重命名'}
                        </button>
                      </div>
                    </EvidenceCard>
                  )
                })}
              </div>
            )}
          </section>

          <section>
            <div className="mb-3">
              <h2 className="text-lg font-semibold text-[color:var(--ui-text)]">关键素材库</h2>
            </div>
            <div className="grid gap-4 xl:grid-cols-3">
              {materials.length === 0 ? <EmptyEvidence title="暂无关键素材" description="候选不足时默认不生成关键素材，请先确认高质量候选。" /> : materials.slice(0, 9).map(material => {
                const materialRange = formatRangeText(material.time_start, material.time_end)
                const media = experimentFileUrl(material.preview_url || material.clip_url || material.frame_path, id)
                return (
                  <EvidenceCard key={material.item_id || material.material_id} className="p-3">
                    <div className="relative aspect-video w-full overflow-hidden rounded-md bg-[color:var(--ui-bg-muted)]">
                      {media ? <img src={media} alt={safeText(material.display_name, '关键素材')} className="h-full w-full object-cover" /> : <div className="flex h-full items-center justify-center text-sm text-[color:var(--ui-text-muted)]">无缩略图</div>}
                    </div>
                    <div className="mt-2 text-xs text-[color:var(--ui-text-muted)]">时间锚点：{materialRange}</div>
                    <div className="mt-1 font-medium text-[color:var(--ui-text)]">{safeText(material.display_name, '关键素材')}</div>
                    <EvidenceBadge tone="success" className="mt-1">关键素材</EvidenceBadge>
                  </EvidenceCard>
                )
              })}
            </div>
          </section>

          {showAdvanced && diagnostics && (
            <details className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/80 p-4 text-xs text-[color:var(--ui-text-muted)] shadow-[var(--ui-shadow-subtle)]">
              <summary className="cursor-pointer font-medium text-[color:var(--ui-text)]">高级信息</summary>
              <div className="mt-3 space-y-2">
                <p>待确认：{diagnostics.candidate_total || '-'}，待确认素材：{diagnostics.candidate_pending_total || '-'}</p>
                <p>关键素材：{diagnostics.formal_material_total || '-'}，窗口：{diagnostics.published_total || '-'}</p>
                <p>缺失文件：{diagnostics.missing_file_count || 0}</p>
              </div>
            </details>
          )}
        </>
      )}
    </div>
  )
}
