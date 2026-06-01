import { type ChangeEvent, type RefObject, type SyntheticEvent, useEffect, useRef, useState } from 'react'
import {
  experimentApi,
  type ExperimentSubExperimentSegment,
  type ExperimentSubExperimentsResponse,
  type SegmentUnderstanding,
} from '../api'
import { cleanDisplayText } from '../displayText'
import { experimentFileUrl, mediaUrl } from '../mediaUrl'
import SegmentUnderstandingPanel from './SegmentUnderstandingPanel'
import {
  type DemoActionCount,
  type DemoSegment,
  getDemoSegments,
  isDemoExperiment,
} from '../demo/weighingPipettingDemo'

const SEGMENT_NAME_NUMBERS = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二']
const SEGMENT_POLL_INTERVAL_MS = 3000
const SEGMENT_NAME_PREFIX = '实验片段'
const SEGMENT_STATUS_TIME_UNKNOWN = '时间区间未知'

const DEBUG_DUAL_VIEW_PREVIEW = String(import.meta.env.VITE_EXPERIMENT_SEGMENT_DUAL_VIEW_DEBUG || '').toLowerCase() === 'true'

const UNDERSTANDING_PENDING_TEXT = '实验片段理解待生成'
const UNDERSTANDING_ERROR_TEXT = 'Qwen VLM处理异常，当前暂无可展示的理解结果'
const THIRD_PERSON_PREVIEW_MISSING_TEXT = '第三人称预览文件待生成'
const FIRST_PERSON_PREVIEW_MISSING_TEXT = '第一人称预览文件待生成'

const PLAY_LABEL = '播放'
const PAUSE_LABEL = '暂停'
const REPLAY_LABEL = '重播'

const THIRD_PERSON_PANEL_LABEL = '第三人称视频'
const FIRST_PERSON_PANEL_LABEL = '第一人称视频'

function toDisplaySegmentName(displayIndex: number) {
  return `${SEGMENT_NAME_PREFIX}${SEGMENT_NAME_NUMBERS[Math.max(0, Math.floor(displayIndex))] || ` ${displayIndex + 1}`}`
}

function clampIndex(index: number) {
  return Number.isFinite(index) ? Math.max(0, Math.floor(index)) : 0
}

function toDisplayTime(sec: number) {
  const safe = Math.max(0, Number(sec) || 0)
  const h = Math.floor(safe / 3600)
  const m = Math.floor((safe % 3600) / 60)
  const s = Math.floor(safe % 60)
  if (h > 0) return `${h.toString().padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function numericValue(value: unknown) {
  if (value === null || value === undefined || value === '') return null
  const asNumber = Number(value)
  return Number.isFinite(asNumber) ? asNumber : null
}

function firstDefinedNonEmptyUrl(...values: Array<string | null | undefined>) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value
  }
  return null
}

function segmentPreviewMediaValue(value: string | null | undefined, experimentId: string) {
  if (!value) return null
  const trimmed = String(value).trim()
  if (!trimmed) return null
  if (/^https?:\/\//i.test(trimmed) || trimmed.startsWith('/api/') || trimmed.startsWith('/')) return trimmed
  if (trimmed.includes('\\') || trimmed.includes('/outputs/experiments/') || trimmed.startsWith('outputs/experiments/')) {
    return experimentFileUrl(trimmed, experimentId) || trimmed
  }
  return trimmed
}

function clampToRange(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min
  return Math.max(min, Math.min(max, value))
}

function formatDurationSec(sec?: number | null) {
  const value = Number(sec)
  if (!Number.isFinite(value)) return 'N/A'
  return `${Math.max(0, Math.round(value))} 秒`
}

function previewModeLabel(value: unknown) {
  const raw = cleanDisplayText(value as string | null, '').toLowerCase()
  if (!raw) return '预览方式：未识别'
  if (raw.includes('realtime') || raw.includes('live') || raw.includes('实时')) return '预览方式：实时'
  if (raw.includes('fast') || raw.includes('fastest') || raw.includes('快')) return '预览方式：快速'
  return '预览方式：未识别'
}

function normalizePlaybackTime(sec: number) {
  if (!Number.isFinite(sec) || sec <= 0) return '00:00'
  return toDisplayTime(sec)
}

function DualViewPanel({
  title,
  videoUrl,
  posterUrl,
  localStartOffsetSec,
  dataSmokePrefix,
  emptyStateLabel,
  videoRef,
  onLoadedMetadata,
  onTimeUpdate,
  onPause,
  onPlay,
  hasVideo,
}: {
  title: string
  videoUrl?: string | null
  posterUrl?: string | null
  localStartOffsetSec?: number | null
  dataSmokePrefix: string
  emptyStateLabel: string
  videoRef: RefObject<HTMLVideoElement>
  onLoadedMetadata: () => void
  onTimeUpdate: (event: SyntheticEvent<HTMLVideoElement>) => void
  onPause: () => void
  onPlay: () => void
  hasVideo: boolean
}) {
  const offsetSec = Number.isFinite(Number(localStartOffsetSec)) ? Number(localStartOffsetSec) : 0

  return (
    <section className="overflow-hidden rounded-md border border-slate-200" data-smoke={dataSmokePrefix}>
      <header className="flex items-center justify-between px-2 py-1">
        <h5 className="text-xs font-bold text-slate-700">{title}</h5>
        {hasVideo ? <span className="text-[11px] text-slate-500">{`偏移 ${normalizePlaybackTime(offsetSec)}`}</span> : null}
      </header>
      {hasVideo && videoUrl ? (
        <div className="relative aspect-video bg-slate-100">
          <video
            ref={videoRef}
            data-smoke={`${dataSmokePrefix}-video`}
            className="absolute inset-0 h-full w-full object-cover"
            src={mediaUrl(videoUrl)}
            poster={mediaUrl(posterUrl)}
            preload="none"
            playsInline
            controls={false}
            onLoadedMetadata={onLoadedMetadata}
            onDurationChange={onLoadedMetadata}
            onTimeUpdate={onTimeUpdate}
            onPause={onPause}
            onPlay={onPlay}
            onEnded={() => onPause()}
            controlsList="nodownload"
          />
        </div>
      ) : (
        <div className="flex aspect-video items-center justify-center bg-slate-50 px-2 text-center text-xs text-slate-500">
          {emptyStateLabel}
        </div>
      )}
    </section>
  )
}

function DualViewPreview({
  thirdPerson,
  firstPerson,
}: {
  thirdPerson: {
    videoUrl?: string | null
    posterUrl?: string | null
    localStartOffsetSec?: number | null
  }
  firstPerson: {
    videoUrl?: string | null
    posterUrl?: string | null
    localStartOffsetSec?: number | null
  }
}) {
  const thirdPersonRef = useRef<HTMLVideoElement>(null)
  const firstPersonRef = useRef<HTMLVideoElement>(null)
  const [thirdPersonDurationSec, setThirdPersonDurationSec] = useState(0)
  const [firstPersonDurationSec, setFirstPersonDurationSec] = useState(0)
  const [localTime, setLocalTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const hasThirdPerson = Boolean(thirdPerson.videoUrl)
  const hasFirstPerson = Boolean(firstPerson.videoUrl)
  const hasAnyVideo = hasThirdPerson || hasFirstPerson
  const combinedDurationSec = Math.max(
    Number.isFinite(thirdPersonDurationSec) ? thirdPersonDurationSec : 0,
    Number.isFinite(firstPersonDurationSec) ? firstPersonDurationSec : 0,
  )
  const sliderMax = Math.max(0, combinedDurationSec - 0.05)
  const clampedLocalTime = clampToRange(localTime, 0, Math.max(0, combinedDurationSec - 0.05))
  const currentTimeLabel = normalizePlaybackTime(clampedLocalTime)
  const totalTimeLabel = combinedDurationSec > 0 ? normalizePlaybackTime(combinedDurationSec) : '00:00'

  function updatePlayState() {
    const thirdVideo = thirdPersonRef.current
    const firstVideo = firstPersonRef.current
    if (!thirdVideo && !firstVideo) return setIsPlaying(false)
    setIsPlaying((thirdVideo?.paused ?? true) || (firstVideo?.paused ?? true) ? false : true)
  }

  function refreshCurrentTime(event: SyntheticEvent<HTMLVideoElement>) {
    const current = Number(event.currentTarget.currentTime || 0)
    if (Number.isFinite(current)) setLocalTime(current)
  }

  function syncDuration() {
    const thirdVideo = thirdPersonRef.current
    const firstVideo = firstPersonRef.current
    if (thirdVideo && Number.isFinite(thirdVideo.duration) && thirdVideo.duration > 0) setThirdPersonDurationSec(thirdVideo.duration)
    if (firstVideo && Number.isFinite(firstVideo.duration) && firstVideo.duration > 0) setFirstPersonDurationSec(firstVideo.duration)
  }

  function syncCurrentTime(next: number) {
    const thirdVideo = thirdPersonRef.current
    const firstVideo = firstPersonRef.current
    const clamped = clampToRange(next, 0, Math.max(0, combinedDurationSec - 0.05))
    if (thirdVideo) thirdVideo.currentTime = clamped
    if (firstVideo) firstVideo.currentTime = clamped
    setLocalTime(clamped)
  }

  async function playBoth() {
    if (!hasAnyVideo) return
    const startJobs = []
    if (thirdPersonRef.current) startJobs.push(thirdPersonRef.current.play())
    if (firstPersonRef.current) startJobs.push(firstPersonRef.current.play())
    if (startJobs.length === 0) return
    setIsPlaying(true)
    const results = await Promise.allSettled(startJobs)
    if (results.every((result) => result.status === 'rejected')) setIsPlaying(false)
  }

  function pauseBoth() {
    if (!hasAnyVideo) return
    if (thirdPersonRef.current) thirdPersonRef.current.pause()
    if (firstPersonRef.current) firstPersonRef.current.pause()
    setIsPlaying(false)
  }

  async function handlePlayPause() {
    if (isPlaying) return pauseBoth()
    return playBoth()
  }

  async function handleReplay() {
    if (!hasAnyVideo) return
    syncCurrentTime(0)
    if (thirdPersonRef.current) thirdPersonRef.current.pause()
    if (firstPersonRef.current) firstPersonRef.current.pause()
    await playBoth()
  }

  async function handleSeek(event: ChangeEvent<HTMLInputElement>) {
    const parsed = Number(event.currentTarget.value)
    syncCurrentTime(Number.isFinite(parsed) ? parsed : 0)
    if (isPlaying) await playBoth()
  }

  return (
    <section className="overflow-hidden rounded-md border border-slate-200" data-smoke="segment-dual-view-preview">
      <div className="grid gap-3 md:grid-cols-2">
        <DualViewPanel
          title={THIRD_PERSON_PANEL_LABEL}
          videoUrl={thirdPerson.videoUrl}
          posterUrl={thirdPerson.posterUrl}
          localStartOffsetSec={thirdPerson.localStartOffsetSec}
          dataSmokePrefix="segment-third-view-preview"
          emptyStateLabel={THIRD_PERSON_PREVIEW_MISSING_TEXT}
          videoRef={thirdPersonRef}
          hasVideo={hasThirdPerson}
          onLoadedMetadata={syncDuration}
          onTimeUpdate={refreshCurrentTime}
          onPause={updatePlayState}
          onPlay={updatePlayState}
        />
        <DualViewPanel
          title={FIRST_PERSON_PANEL_LABEL}
          videoUrl={firstPerson.videoUrl}
          posterUrl={firstPerson.posterUrl}
          localStartOffsetSec={firstPerson.localStartOffsetSec}
          dataSmokePrefix="segment-first-view-preview"
          emptyStateLabel={FIRST_PERSON_PREVIEW_MISSING_TEXT}
          videoRef={firstPersonRef}
          hasVideo={hasFirstPerson}
          onLoadedMetadata={syncDuration}
          onTimeUpdate={refreshCurrentTime}
          onPause={updatePlayState}
          onPlay={updatePlayState}
        />
      </div>
      <div className="space-y-2 px-2 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            className="rounded-md bg-slate-900 px-3 py-1.5 text-xs font-bold text-white"
            onClick={() => void handlePlayPause()}
            data-smoke="segment-dual-view-preview-toggle"
            disabled={!hasAnyVideo}
          >
            {isPlaying ? PAUSE_LABEL : PLAY_LABEL}
          </button>
          <button
            type="button"
            className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-bold text-slate-700"
            onClick={() => void handleReplay()}
            data-smoke="segment-dual-view-preview-replay"
            disabled={!hasAnyVideo}
          >
            {REPLAY_LABEL}
          </button>
          <span className="ml-auto text-xs text-slate-500">
            {`当前 ${currentTimeLabel} / ${totalTimeLabel}`}
          </span>
        </div>
        {hasAnyVideo ? (
          <input
            type="range"
            min={0}
            max={sliderMax}
            step={0.1}
            value={clampedLocalTime}
            onChange={event => void handleSeek(event)}
            aria-label="双视角同步播放进度条"
            data-smoke="segment-dual-view-preview-seek"
            className="h-1 w-full accent-blue-600"
          />
        ) : (
          <div className="text-xs text-slate-500">未检测到可播放视角</div>
        )}
      </div>
    </section>
  )
}

function SegmentStatusInfo({ segment }: { segment: ExperimentSubExperimentSegment }) {
  const start = numericValue(segment.start_sec)
  const end = numericValue(segment.end_sec)
  const previewDuration = numericValue(segment.preview_duration_s)
  const fallbackDuration = numericValue(segment.duration_sec)
  const displayDuration = previewDuration ?? fallbackDuration

  const lines: string[] = []
  if (start != null && end != null) {
    lines.push(`时间区间: ${toDisplayTime(start)} - ${toDisplayTime(end)}；时长: ${formatDurationSec(Math.max(0, end - start))}`)
  } else {
    lines.push(SEGMENT_STATUS_TIME_UNKNOWN)
  }
  lines.push(`可见时长: ${displayDuration != null ? formatDurationSec(displayDuration) : 'N/A'}`)
  return (
    <p className="mt-1 text-xs text-slate-500">
      {lines.join(' / ')}
    </p>
  )
}

function isFormalSegment(segment: ExperimentSubExperimentSegment) {
  return segment.formal_dual_view_action === true
    || segment.formal_action_status === 'confirmed'
    || segment.view_alignment_status === 'dual_view_action_confirmed'
}

function sortSegments(segments: ExperimentSubExperimentSegment[]) {
  return [...segments].sort((a, b) => Number(a.start_sec || 0) - Number(b.start_sec || 0))
}

function SegmentCard({
  segment,
  displayIndex,
  experimentId,
  understanding,
  understandingMessage,
}: {
  segment: ExperimentSubExperimentSegment
  displayIndex: number
  experimentId: string
  understanding?: SegmentUnderstanding
  understandingMessage?: string | null
}) {
  const thirdPersonPreview = segmentPreviewMediaValue(firstDefinedNonEmptyUrl(
    segment.third_view_realtime_preview_url,
    segment.third_view_realtime_preview,
    segment.third_view_preview_url,
    segment.third_preview_url,
    segment.third_person_video_url,
  ), experimentId)
  const firstPersonPreview = segmentPreviewMediaValue(firstDefinedNonEmptyUrl(
    segment.first_view_realtime_preview_url,
    segment.first_view_realtime_preview,
    segment.first_view_preview_url,
    segment.first_preview_url,
    segment.first_person_video_url,
  ), experimentId)
  const sideBySidePreview = segmentPreviewMediaValue(firstDefinedNonEmptyUrl(
    segment.side_by_side_realtime_preview_url,
    segment.side_by_side_realtime_preview,
    segment.side_by_side_preview_url,
  ), experimentId)

  const thirdPoster = firstDefinedNonEmptyUrl(segment.third_person_poster_url, segment.preview_poster_url)
  const firstPoster = firstDefinedNonEmptyUrl(segment.first_person_poster_url, segment.preview_poster_url)

  return (
    <article className="rounded-lg border border-slate-200 bg-white p-4" data-smoke="experiment-segment-card">
      <h4 className="text-sm font-black text-slate-900">{toDisplaySegmentName(clampIndex(displayIndex))}</h4>
      <SegmentStatusInfo segment={segment} />
      <p className="mt-1 text-xs text-slate-500">{previewModeLabel(segment.preview_mode)}</p>

      <section className="mt-3">
        <DualViewPreview
          thirdPerson={{
            videoUrl: thirdPersonPreview,
            posterUrl: thirdPoster,
            localStartOffsetSec: segment.third_person_local_start_sec,
          }}
          firstPerson={{
            videoUrl: firstPersonPreview,
            posterUrl: firstPoster,
            localStartOffsetSec: segment.first_person_local_start_sec,
          }}
        />
      </section>

      {DEBUG_DUAL_VIEW_PREVIEW && sideBySidePreview && (
        <section className="mt-3 space-y-2" data-smoke="segment-side-by-side-preview-wrapper">
          <p className="text-xs font-medium text-slate-700">第三/第一画中画</p>
          <div className="overflow-hidden rounded-md border border-slate-200" data-smoke="segment-side-by-side-preview">
            <p className="sr-only">第三/第一合并预览</p>
            <div className="relative aspect-video bg-slate-100">
              <video className="absolute inset-0 h-full w-full object-cover" src={mediaUrl(sideBySidePreview)} controls />
            </div>
          </div>
        </section>
      )}

      <section className="mt-3" data-smoke="segment-understanding">
        <p className="text-xs font-black text-slate-900">实验片段理解</p>
        {understanding?.understanding_text ? (
          <SegmentUnderstandingPanel
            text={understanding.understanding_text}
            source={understanding.source}
          />
        ) : (
          <p className="mt-2 rounded-md border border-slate-200 bg-slate-50 p-2 text-xs text-slate-500">
            {understandingMessage || UNDERSTANDING_PENDING_TEXT}
          </p>
        )}
      </section>
    </article>
  )
}

export default function ExperimentSegments({ experimentId }: { experimentId: string }) {
  if (isDemoExperiment(experimentId)) {
    return <DemoExperimentSegments />
  }
  return <BackendExperimentSegments experimentId={experimentId} />
}

const ACTION_BADGE_TONES = [
  'bg-[color:var(--ui-accent-soft)] text-[color:var(--ui-accent)]',
  'bg-emerald-50 text-emerald-700',
  'bg-amber-50 text-amber-700',
  'bg-sky-50 text-sky-700',
  'bg-violet-50 text-violet-700',
]

function ActionCountBadges({ counts }: { counts: DemoActionCount[] }) {
  if (!counts.length) return null
  return (
    <div className="mt-2 flex flex-wrap gap-1.5" data-smoke="segment-action-counts">
      {counts.map((item, index) => (
        <span
          key={item.label}
          className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-semibold ${ACTION_BADGE_TONES[index % ACTION_BADGE_TONES.length]}`}
        >
          {item.label}
          <span className="font-black">×{item.count}</span>
        </span>
      ))}
    </div>
  )
}

function DemoSegmentCard({ segment }: { segment: DemoSegment }) {
  const totalActions = segment.actionCounts.reduce((sum, item) => sum + item.count, 0)
  return (
    <article className="rounded-lg border border-slate-200 bg-white p-4" data-smoke="experiment-segment-card">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="text-sm font-black text-slate-900">{segment.displayName}</h4>
        <span className="rounded-full bg-[color:var(--ui-bg-muted)] px-2.5 py-1 text-[11px] font-semibold text-[color:var(--ui-text-muted)]">
          {toDisplayTime(segment.startSec)} - {toDisplayTime(segment.endSec)} · {formatDurationSec(segment.durationSec)}
        </span>
      </div>
      <p className="mt-1 text-xs text-slate-500">
        阶段：{segment.phase} · 检测到 {totalActions} 次关键交互动作
      </p>

      <ActionCountBadges counts={segment.actionCounts} />

      <section className="mt-3">
        <DualViewPreview
          thirdPerson={{ videoUrl: segment.curatedClip.thirdVideoUrl, posterUrl: segment.curatedClip.thirdPosterUrl }}
          firstPerson={{ videoUrl: segment.curatedClip.firstVideoUrl, posterUrl: segment.curatedClip.firstPosterUrl }}
        />
      </section>

      <section className="mt-3" data-smoke="segment-understanding">
        <p className="text-xs font-black text-slate-900">实验过程理解</p>
        <SegmentUnderstandingPanel markdown={segment.understandingMarkdown} />
      </section>
    </article>
  )
}

function DemoExperimentSegments() {
  const segments = getDemoSegments()
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm" data-smoke="experiment-segments">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-base font-black text-slate-900">实验片段</h3>
        <span className="text-xs font-semibold text-[color:var(--ui-text-muted)]">共 {segments.length} 个片段 · 双视角对齐</span>
      </div>
      <div className="mt-3 space-y-4">
        {segments.map(segment => (
          <DemoSegmentCard key={segment.segmentId} segment={segment} />
        ))}
      </div>
    </section>
  )
}

function BackendExperimentSegments({ experimentId }: { experimentId: string }) {
  const [data, setData] = useState<ExperimentSubExperimentsResponse | null>(null)
  const [understandingById, setUnderstandingById] = useState<Record<string, SegmentUnderstanding>>({})
  const [understandingMessage, setUnderstandingMessage] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const loadUnderstanding = async () => {
      try {
        const payload = await experimentApi.getSegmentUnderstanding(experimentId)
        if (cancelled) return
        const map: Record<string, SegmentUnderstanding> = {}
        for (const seg of payload.segments || []) {
          if (seg?.segment_id) map[seg.segment_id] = seg
        }
        setUnderstandingById(map)
        setUnderstandingMessage(Object.keys(map).length ? null : UNDERSTANDING_PENDING_TEXT)
      } catch {
        setUnderstandingMessage(UNDERSTANDING_ERROR_TEXT)
      }
    }
    void loadUnderstanding()
    return () => {
      cancelled = true
    }
  }, [experimentId])

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | null = null
    const load = async (force = false) => {
      try {
        const payload = await experimentApi.getSubExperiments(experimentId, { force })
        if (cancelled) return
        setData(payload)
        const needPoll = payload.time_axis_unreliable || payload.preview_error || payload.total === 0 || !payload.segments?.length
        if (needPoll) {
          timer = setTimeout(() => {
            void load(true)
          }, SEGMENT_POLL_INTERVAL_MS)
        }
      } catch {
        if (!cancelled) {
          timer = setTimeout(() => {
            void load(true)
          }, SEGMENT_POLL_INTERVAL_MS)
        }
      }
    }
    void load(true)
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [experimentId])

  if (!data) return null

  if (data.time_axis_unreliable || data.preview_error === 'time_axis_unreliable') {
    return (
      <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm" data-smoke="experiment-segments">
        <h3 className="text-base font-black text-slate-900">实验片段</h3>
        <p className="mt-1 text-xs text-slate-500">片段时间轴不可靠，请稍后重试</p>
      </section>
    )
  }

  if (data.total === 0) {
    return (
      <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm" data-smoke="experiment-segments">
        <h3 className="text-base font-black text-slate-900">实验片段</h3>
        <p className="mt-1 text-xs text-slate-500">{data.message || '当前暂无实验片段'}</p>
      </section>
    )
  }

  const formalSegments = sortSegments(data.segments.filter(isFormalSegment))
  const weakCandidates = sortSegments(data.segments.filter(segment => !isFormalSegment(segment)))
  const displayedSegments = formalSegments.length > 0 ? formalSegments : weakCandidates

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm" data-smoke="experiment-segments">
      <h3 className="text-base font-black text-slate-900">实验片段</h3>
      <div className="mt-3 space-y-4">
        {displayedSegments.map((segment, index) => (
          <SegmentCard
            key={segment.segment_id}
            segment={segment}
            displayIndex={index}
            experimentId={experimentId}
            understanding={understandingById[segment.segment_id]}
            understandingMessage={understandingMessage}
          />
        ))}
      </div>
    </section>
  )
}
