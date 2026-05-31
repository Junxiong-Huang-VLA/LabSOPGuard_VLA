import { useEffect, useRef, useState } from 'react'
import { experimentApi, type ExperimentSubExperimentSegment, type ExperimentSubExperimentsResponse, type SegmentUnderstanding } from '../api'
import { cleanDisplayText } from '../displayText'
import { experimentFileUrl, mediaUrl } from '../mediaUrl'
import SegmentUnderstandingPanel from './SegmentUnderstandingPanel'

const SEGMENT_NAME_NUMBERS = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二']
const SEGMENT_POLL_INTERVAL_MS = 3000

type PreviewMode = 'realtime' | 'fast' | 'unknown'
type SegmentSyncVideoKind = 'third' | 'first'

function toDisplaySegmentName(displayIndex: number) {
  return `实验片段${SEGMENT_NAME_NUMBERS[Math.max(0, Math.floor(displayIndex))] || String(displayIndex + 1)}`
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

function formatDuration(sec: number | null | undefined) {
  if (!Number.isFinite(Number(sec))) return '未记录'
  return `${Math.round(Number(sec))} 秒`
}

function clampPlaybackNumber(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min
  return Math.max(min, Math.min(max, value))
}

function clampToDuration(time: number, duration: number) {
  if (!Number.isFinite(duration) || duration <= 0) return 0
  return Math.max(0, Math.min(time, Math.max(0, duration - 0.05)))
}

function isFormalSegment(seg: ExperimentSubExperimentSegment) {
  return seg.formal_dual_view_action === true
    || seg.formal_action_status === 'confirmed'
    || seg.view_alignment_status === 'dual_view_action_confirmed'
}

function normalizePreviewMode(value: unknown): PreviewMode {
  const raw = cleanDisplayText(value as string | null, '').toLowerCase()
  if (!raw) return 'unknown'
  if (raw.includes('realtime') || raw.includes('实时') || raw.includes('live') || raw.includes('liveview')) return 'realtime'
  if (raw.includes('fast') || raw.includes('快速') || raw.includes('压缩') || raw.includes('compressed')) return 'fast'
  return 'unknown'
}

function DualViewPreviewSection({
  thirdViewUrl,
  firstViewUrl,
  thirdStartOffsetSec,
  firstStartOffsetSec,
  thirdPosterUrl,
  firstPosterUrl,
}: {
  thirdViewUrl?: string | null
  firstViewUrl?: string | null
  thirdStartOffsetSec?: number | null
  firstStartOffsetSec?: number | null
  thirdPosterUrl?: string | null
  firstPosterUrl?: string | null
}) {
  const thirdVideoRef = useRef<HTMLVideoElement>(null)
  const firstVideoRef = useRef<HTMLVideoElement>(null)
  const [durations, setDurations] = useState<{ third: number; first: number }>({ third: 0, first: 0 })
  const [syncTimeSec, setSyncTimeSec] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const hasThird = Boolean(thirdViewUrl)
  const hasFirst = Boolean(firstViewUrl)
  const thirdOffset = Number.isFinite(Number(thirdStartOffsetSec)) ? Number(thirdStartOffsetSec) : 0
  const firstOffset = Number.isFinite(Number(firstStartOffsetSec)) ? Number(firstStartOffsetSec) : 0

  const timelineStart = Math.min(thirdOffset, firstOffset, 0)
  const timelineEnd = Math.max(
    hasThird ? thirdOffset + durations.third : Number.NEGATIVE_INFINITY,
    hasFirst ? firstOffset + durations.first : Number.NEGATIVE_INFINITY,
    timelineStart + 0.1,
  )
  const timelineMax = clampPlaybackNumber(timelineEnd, timelineStart + 0.1, Number.POSITIVE_INFINITY)
  const controlTime = clampPlaybackNumber(syncTimeSec, timelineStart, timelineMax)

  useEffect(() => {
    setSyncTimeSec(previous => clampPlaybackNumber(previous, timelineStart, timelineMax))
  }, [timelineStart, timelineMax])

  const hasDuration = {
    third: durations.third > 0 && Number.isFinite(durations.third),
    first: durations.first > 0 && Number.isFinite(durations.first),
  }

  function getActiveVideos() {
    const list: Array<{
      kind: SegmentSyncVideoKind
      ref: HTMLVideoElement
      offset: number
      duration: number
    }> = []
    if (hasThird && thirdVideoRef.current) list.push({
      kind: 'third',
      ref: thirdVideoRef.current,
      offset: thirdOffset,
      duration: durations.third,
    })
    if (hasFirst && firstVideoRef.current) list.push({
      kind: 'first',
      ref: firstVideoRef.current,
      offset: firstOffset,
      duration: durations.first,
    })
    return list
  }

  function syncToExperimentTime(targetSec: number) {
    const clamped = clampPlaybackNumber(targetSec, timelineStart, timelineMax)
    setSyncTimeSec(clamped)
    const videos = getActiveVideos()
    videos.forEach(item => {
      const localTime = clamped - item.offset
      const safeLocal = clampToDuration(localTime, item.duration)
      if (Math.abs(item.ref.currentTime - safeLocal) > 0.05) item.ref.currentTime = safeLocal
    })
  }

  function handleLoadedMetadata(kind: SegmentSyncVideoKind) {
    const video = kind === 'third' ? thirdVideoRef.current : firstVideoRef.current
    if (!video || !Number.isFinite(video.duration)) return
    setDurations(previous => {
      if (kind === 'third') {
        if (previous.third === video.duration) return previous
        return { ...previous, third: video.duration }
      }
      if (previous.first === video.duration) return previous
      return { ...previous, first: video.duration }
    })
  }

  function syncTimeFromLocal(kind: SegmentSyncVideoKind, localTime: number, duration: number) {
    if (!duration) return
    const offset = kind === 'third' ? thirdOffset : firstOffset
    const nextTime = clampPlaybackNumber(localTime + offset, timelineStart, timelineMax)
    setSyncTimeSec(nextTime)
    if (!isPlaying) return

    const videos = getActiveVideos()
    const others = videos.filter(item => item.kind !== kind)
    others.forEach(item => {
      const targetLocal = clampToDuration(nextTime - item.offset, item.duration)
      if (Math.abs(item.ref.currentTime - targetLocal) > 0.25) item.ref.currentTime = targetLocal
    })
  }

  function pauseBoth() {
    getActiveVideos().forEach(item => item.ref.pause())
    setIsPlaying(false)
  }

  async function playBoth() {
    const videos = getActiveVideos()
    if (!videos.length) return
    syncToExperimentTime(controlTime)
    await Promise.all(videos.map(item => item.ref.play().catch(() => undefined)))
    setIsPlaying(true)
  }

  function handleReplay() {
    pauseBoth()
    syncToExperimentTime(timelineStart)
    void playBoth()
  }

  function handleSeek(rawValue: string) {
    const next = Number(rawValue)
    pauseBoth()
    syncToExperimentTime(Number.isFinite(next) ? next : timelineStart)
  }

  return (
    <section className="rounded-md border border-slate-200 p-2">
      <div className="mb-2 flex flex-wrap items-center gap-2" data-smoke="segment-dual-view-comparison-controls">
        <button
          type="button"
          className="rounded-md bg-slate-900 px-3 py-1.5 text-xs font-bold text-white"
          data-smoke="segment-dual-view-comparison-play"
          onClick={isPlaying ? pauseBoth : () => void playBoth()}
        >
          {isPlaying ? '暂停' : '播放'}
        </button>
        <button
          type="button"
          className="rounded-md border border-slate-200 px-3 py-1.5 text-xs font-bold text-slate-700"
          data-smoke="segment-dual-view-comparison-replay"
          onClick={handleReplay}
        >
          重放
        </button>
        <input
          type="range"
          min={timelineStart}
          max={timelineMax}
          step={0.1}
          value={Math.min(controlTime, timelineMax)}
          onChange={event => handleSeek(event.currentTarget.value)}
          aria-label="segment dual playback timeline"
          data-smoke="segment-dual-view-comparison-seek"
          className="h-1 flex-1 accent-blue-600"
        />
      </div>
      <div className="grid gap-3 md:grid-cols-2" data-smoke="segment-dual-preview-grid">
        <section className="overflow-hidden rounded-md border border-slate-200" data-smoke="segment-third-view-preview">
          <h5 className="px-2 py-1 text-xs font-bold text-slate-700">第三人称视角</h5>
          {hasThird ? (
            <div className="relative aspect-video">
              <video
                ref={thirdVideoRef}
                data-smoke="segment-third-person-preview-video"
                className="absolute inset-0 h-full w-full bg-slate-900 object-cover"
                src={mediaUrl(thirdViewUrl)}
                poster={mediaUrl(thirdPosterUrl)}
                preload="metadata"
                playsInline
                controls={false}
                onLoadedMetadata={() => handleLoadedMetadata('third')}
                onDurationChange={() => handleLoadedMetadata('third')}
                onTimeUpdate={event => {
                  const current = event.currentTarget.currentTime
                  if (!hasDuration.third && event.currentTarget.duration > 0) {
                    setDurations(previous => ({ ...previous, third: event.currentTarget.duration }))
                  }
                  syncTimeFromLocal('third', current, event.currentTarget.duration)
                }}
                controlsList="nodownload"
              />
            </div>
          ) : (
            <div className="flex aspect-video items-center justify-center bg-slate-50 text-xs text-slate-500">第三人称预览待生成</div>
          )}
        </section>
        <section className="overflow-hidden rounded-md border border-slate-200" data-smoke="segment-first-view-preview">
          <h5 className="px-2 py-1 text-xs font-bold text-slate-700">第一人称视角</h5>
          {hasFirst ? (
            <div className="relative aspect-video">
              <video
                ref={firstVideoRef}
                data-smoke="segment-first-person-preview-video"
                className="absolute inset-0 h-full w-full bg-slate-900 object-cover"
                src={mediaUrl(firstViewUrl)}
                poster={mediaUrl(firstPosterUrl)}
                preload="metadata"
                playsInline
                controls={false}
                onLoadedMetadata={() => handleLoadedMetadata('first')}
                onDurationChange={() => handleLoadedMetadata('first')}
                onTimeUpdate={event => {
                  const current = event.currentTarget.currentTime
                  if (!hasDuration.first && event.currentTarget.duration > 0) {
                    setDurations(previous => ({ ...previous, first: event.currentTarget.duration }))
                  }
                  syncTimeFromLocal('first', current, event.currentTarget.duration)
                }}
                controlsList="nodownload"
              />
            </div>
          ) : (
            <div className="flex aspect-video items-center justify-center bg-slate-50 text-xs text-slate-500">第一人称预览待生成</div>
          )}
        </section>
      </div>
    </section>
  )
}

function SegmentStatusInfo({ segment }: { segment: ExperimentSubExperimentSegment }) {
  const start = numericValue(segment.start_sec)
  const end = numericValue(segment.end_sec)
  if (start == null || end == null) return null
  const duration = Math.max(0, end - start)
  const hasDuration = Number.isFinite(duration)
  if (!hasDuration) return <p className="mt-1 text-xs text-slate-500">时间范围：{toDisplayTime(start)} - {toDisplayTime(end)}</p>
  return <p className="mt-1 text-xs text-slate-500">时间范围：{toDisplayTime(start)} - {toDisplayTime(end)}，时长 {formatDuration(duration)}</p>
}

function toDisplayPreviewMode(segment: ExperimentSubExperimentSegment) {
  const mode = normalizePreviewMode(segment.preview_mode)
  if (mode === 'realtime') return '实时预览'
  if (mode === 'fast') return '快速预览'
  return '预览模式未记录'
}

function SegmentCard({
  segment,
  displayIndex,
  experimentId,
  understanding,
}: {
  segment: ExperimentSubExperimentSegment
  displayIndex: number
  experimentId: string
  understanding?: SegmentUnderstanding
}) {
  const thirdPersonPreview = segmentPreviewMediaValue(firstDefinedNonEmptyUrl(
    segment.third_view_realtime_preview_url,
    segment.third_view_realtime_preview,
    segment.third_view_preview_url,
    segment.third_preview_url,
  ), experimentId)
  const firstPersonPreview = segmentPreviewMediaValue(firstDefinedNonEmptyUrl(
    segment.first_view_realtime_preview_url,
    segment.first_view_realtime_preview,
    segment.first_view_preview_url,
    segment.first_preview_url,
  ), experimentId)
  const thirdPoster = firstDefinedNonEmptyUrl(segment.third_person_poster_url, segment.preview_poster_url)
  const firstPoster = firstDefinedNonEmptyUrl(segment.first_person_poster_url, segment.preview_poster_url)

  return (
    <article className="rounded-lg border border-slate-200 bg-white p-4" data-smoke="experiment-segment-card">
      <h4 className="text-sm font-black text-slate-900">{toDisplaySegmentName(clampIndex(displayIndex))}</h4>
      <SegmentStatusInfo segment={segment} />
      <p className="mt-1 text-xs text-slate-500">{toDisplayPreviewMode(segment)}</p>

      <section className="mt-3" data-smoke="segment-dual-view-comparison">
        <DualViewPreviewSection
          thirdViewUrl={thirdPersonPreview}
          firstViewUrl={firstPersonPreview}
          thirdStartOffsetSec={segment.third_person_local_start_sec}
          firstStartOffsetSec={segment.first_person_local_start_sec}
          thirdPosterUrl={thirdPoster}
          firstPosterUrl={firstPoster}
        />
      </section>

      {understanding?.understanding_text ? (
        <SegmentUnderstandingPanel
          text={understanding.understanding_text}
          source={understanding.source}
        />
      ) : null}
    </article>
  )
}

function sortSegments(segments: ExperimentSubExperimentSegment[]) {
  return [...segments].sort((a, b) => Number(a.start_sec || 0) - Number(b.start_sec || 0))
}

export default function ExperimentSegments({ experimentId }: { experimentId: string }) {
  const [data, setData] = useState<ExperimentSubExperimentsResponse | null>(null)
  const [understandingById, setUnderstandingById] = useState<Record<string, SegmentUnderstanding>>({})

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
      } catch {
        // Understanding is supplementary; ignore failures so segments still render.
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
        <p className="mt-1 text-xs text-slate-500">片段正在生成，请稍后查看。</p>
      </section>
    )
  }

  if (data.total === 0) {
    return (
      <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm" data-smoke="experiment-segments">
        <h3 className="text-base font-black text-slate-900">实验片段</h3>
        <p className="mt-1 text-xs text-slate-500">{data.message || '当前暂无实验片段。'}</p>
      </section>
    )
  }

  const official = sortSegments(data.segments.filter(isFormalSegment))
  const weakCandidates = sortSegments(data.segments.filter(segment => !isFormalSegment(segment)))
  const displayedSegments = official.length > 0 ? official : weakCandidates

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm" data-smoke="experiment-segments">
      <h3 className="text-base font-black text-slate-900">实验片段</h3>
      <div className="mt-3 space-y-4">
        {displayedSegments.map((segment, index) => (
          <SegmentCard key={segment.segment_id} segment={segment} displayIndex={index} experimentId={experimentId} understanding={understandingById[segment.segment_id]} />
        ))}
      </div>
    </section>
  )
}
