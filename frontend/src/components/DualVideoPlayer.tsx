import { useEffect, useMemo, useRef, useState } from 'react'
import type { RefObject } from 'react'

export interface DualVideoPlayerHandle {
  seekTo?: number
  token?: number
}

type VideoKind = 'third' | 'first'
type VideoRef = {
  kind: VideoKind
  video: HTMLVideoElement
  offsetSec: number
}

function seekWhenReady(video: HTMLVideoElement, timeSec: number) {
  const applySeek = () => {
    const duration = Number.isFinite(video.duration) ? video.duration : undefined
    const nextTime = Math.max(0, duration ? Math.min(timeSec, Math.max(0, duration - 0.05)) : timeSec)
    try {
      video.currentTime = nextTime
    } catch {
      // Metadata can arrive after a step-track click; the loadedmetadata listener retries.
    }
  }
  if (video.readyState === 0) {
    video.addEventListener('loadedmetadata', applySeek, { once: true })
    video.load()
    return () => video.removeEventListener('loadedmetadata', applySeek)
  }
  applySeek()
  return undefined
}

function formatTime(value: number) {
  const safe = Number.isFinite(value) && value > 0 ? value : 0
  const minutes = Math.floor(safe / 60)
  const seconds = safe - minutes * 60
  return `${minutes}:${seconds.toFixed(1).padStart(4, '0')}`
}

function safeOffset(value: number | undefined, fallback: number) {
  return Number.isFinite(value) ? Number(value) : fallback
}

export default function DualVideoPlayer({
  firstPersonUrl,
  thirdPersonUrl,
  seekRequest,
  timeOffsetSec = 0,
  firstPersonTimeOffsetSec,
  thirdPersonTimeOffsetSec,
  calibrationOffsetSec = 0,
  onCalibrationSave,
  onCalibrationReset,
}: {
  firstPersonUrl?: string
  thirdPersonUrl?: string
  seekRequest?: DualVideoPlayerHandle | null
  timeOffsetSec?: number
  firstPersonTimeOffsetSec?: number
  thirdPersonTimeOffsetSec?: number
  calibrationOffsetSec?: number
  onCalibrationSave?: (offsetAdjustSec: number) => Promise<void> | void
  onCalibrationReset?: () => Promise<void> | void
}) {
  const firstRef = useRef<HTMLVideoElement>(null)
  const thirdRef = useRef<HTMLVideoElement>(null)
  const savedCalibrationOffset = Number.isFinite(calibrationOffsetSec) ? Number(calibrationOffsetSec) : 0
  const [calibrationDraft, setCalibrationDraft] = useState(savedCalibrationOffset)
  const calibrationPreviewDelta = calibrationDraft - savedCalibrationOffset
  const baseSourceOffset = Number.isFinite(timeOffsetSec) ? timeOffsetSec : 0
  const baseOffset = baseSourceOffset + calibrationPreviewDelta
  const offsets = {
    third: safeOffset(thirdPersonTimeOffsetSec, baseSourceOffset) + calibrationPreviewDelta,
    first: safeOffset(firstPersonTimeOffsetSec, baseSourceOffset) + calibrationPreviewDelta,
  }
  const [syncPlaying, setSyncPlaying] = useState(false)
  const [activeMode, setActiveMode] = useState<'sync' | VideoKind>('sync')
  const [currentExperimentTime, setCurrentExperimentTime] = useState(baseOffset)
  const [durations, setDurations] = useState<Record<VideoKind, number>>({ third: 0, first: 0 })
  const [singlePlaying, setSinglePlaying] = useState<Record<VideoKind, boolean>>({ third: false, first: false })

  const availableVideos = (): VideoRef[] => [
    thirdRef.current ? { kind: 'third' as const, video: thirdRef.current, offsetSec: offsets.third } : null,
    firstRef.current ? { kind: 'first' as const, video: firstRef.current, offsetSec: offsets.first } : null,
  ].filter((item): item is VideoRef => Boolean(item))

  useEffect(() => {
    setCalibrationDraft(savedCalibrationOffset)
  }, [savedCalibrationOffset])

  const timelineStart = useMemo(() => {
    const starts = [
      thirdPersonUrl ? offsets.third : undefined,
      firstPersonUrl ? offsets.first : undefined,
    ].filter((value): value is number => Number.isFinite(value))
    return starts.length ? Math.min(...starts) : baseOffset
  }, [baseOffset, firstPersonUrl, offsets.first, offsets.third, thirdPersonUrl])

  const timelineEnd = useMemo(() => {
    const ends = [
      thirdPersonUrl && durations.third > 0 ? offsets.third + durations.third : undefined,
      firstPersonUrl && durations.first > 0 ? offsets.first + durations.first : undefined,
    ].filter((value): value is number => Number.isFinite(value))
    return Math.max(timelineStart + 0.1, ...(ends.length ? ends : [currentExperimentTime + 0.1]))
  }, [currentExperimentTime, durations.first, durations.third, firstPersonUrl, offsets.first, offsets.third, thirdPersonUrl, timelineStart])

  const clampExperimentTime = (value: number) => Math.max(timelineStart, Math.min(value, Math.max(timelineStart, timelineEnd - 0.05)))
  const localTimeFor = (item: VideoRef, experimentTime: number) => Math.max(0, experimentTime - item.offsetSec)
  const experimentTimeFor = (kind: VideoKind, localTime: number) => localTime + offsets[kind]

  function syncSeekExperiment(experimentTime: number) {
    const target = clampExperimentTime(experimentTime)
    setCurrentExperimentTime(target)
    const cleanups = availableVideos()
      .map(item => seekWhenReady(item.video, localTimeFor(item, target)))
      .filter((cleanup): cleanup is () => void => Boolean(cleanup))
    return () => cleanups.forEach(cleanup => cleanup())
  }

  useEffect(() => {
    if (seekRequest?.seekTo == null) return
    return syncSeekExperiment(seekRequest.seekTo)
  }, [seekRequest?.seekTo, seekRequest?.token, timelineStart, timelineEnd, offsets.first, offsets.third])

  useEffect(() => {
    if (currentExperimentTime < timelineStart || currentExperimentTime > timelineEnd) {
      setCurrentExperimentTime(clampExperimentTime(currentExperimentTime))
    }
  }, [timelineStart, timelineEnd])

  useEffect(() => {
    if (!syncPlaying) return
    const interval = window.setInterval(() => {
      const videos = availableVideos()
      const leader = videos.find(item => !item.video.paused) || videos[0]
      if (!leader) return
      const nextExperimentTime = clampExperimentTime(leader.video.currentTime + leader.offsetSec)
      setCurrentExperimentTime(nextExperimentTime)
      videos.forEach(item => {
        const targetLocal = localTimeFor(item, nextExperimentTime)
        if (Math.abs(item.video.currentTime - targetLocal) > 0.25) item.video.currentTime = targetLocal
        if (item.video.paused && !leader.video.paused) void item.video.play().catch(() => undefined)
      })
    }, 250)
    return () => window.clearInterval(interval)
  }, [syncPlaying, firstPersonUrl, thirdPersonUrl, offsets.first, offsets.third, timelineStart, timelineEnd])

  async function playBoth() {
    const target = clampExperimentTime(currentExperimentTime || timelineStart)
    const videos = availableVideos()
    videos.forEach(item => {
      const targetLocal = localTimeFor(item, target)
      if (Math.abs(item.video.currentTime - targetLocal) > 0.25) item.video.currentTime = targetLocal
    })
    await Promise.all(videos.map(item => item.video.play().catch(() => undefined)))
    setActiveMode('sync')
    setSyncPlaying(true)
  }

  function pauseBoth() {
    availableVideos().forEach(item => item.video.pause())
    setSyncPlaying(false)
  }

  function toggleSyncPlayback() {
    if (syncPlaying) pauseBoth()
    else void playBoth()
  }

  function handleTimelineScrub(experimentTime: number) {
    pauseBoth()
    syncSeekExperiment(experimentTime)
    setActiveMode('sync')
  }

  function updateDuration(kind: VideoKind) {
    const video = kind === 'third' ? thirdRef.current : firstRef.current
    if (!video || !Number.isFinite(video.duration)) return
    setDurations(previous => ({ ...previous, [kind]: video.duration }))
  }

  function updateTime(kind: VideoKind) {
    const video = kind === 'third' ? thirdRef.current : firstRef.current
    if (!video) return
    if (!syncPlaying || activeMode === kind) setCurrentExperimentTime(clampExperimentTime(experimentTimeFor(kind, video.currentTime)))
  }

  async function playSingle(kind: VideoKind) {
    const target = kind === 'third' ? thirdRef.current : firstRef.current
    if (!target) return
    availableVideos().forEach(item => {
      if (item.kind !== kind) item.video.pause()
    })
    setSyncPlaying(false)
    setActiveMode(kind)
    await target.play().catch(() => undefined)
    setSinglePlaying(previous => ({ ...previous, [kind]: true }))
    setCurrentExperimentTime(clampExperimentTime(experimentTimeFor(kind, target.currentTime)))
  }

  function pauseSingle(kind: VideoKind) {
    const target = kind === 'third' ? thirdRef.current : firstRef.current
    target?.pause()
    setSinglePlaying(previous => ({ ...previous, [kind]: false }))
    setSyncPlaying(false)
  }

  function scrubSingle(kind: VideoKind, localTime: number) {
    const target = kind === 'third' ? thirdRef.current : firstRef.current
    if (!target) return
    target.pause()
    target.currentTime = Math.max(0, localTime)
    setSinglePlaying(previous => ({ ...previous, [kind]: false }))
    setActiveMode(kind)
    setSyncPlaying(false)
    setCurrentExperimentTime(clampExperimentTime(experimentTimeFor(kind, target.currentTime)))
  }

  function syncFromPane(kind: VideoKind) {
    const video = kind === 'third' ? thirdRef.current : firstRef.current
    if (!video) return
    pauseBoth()
    syncSeekExperiment(experimentTimeFor(kind, video.currentTime))
    setActiveMode('sync')
  }

  function requestPaneFullscreen(kind: VideoKind) {
    const video = kind === 'third' ? thirdRef.current : firstRef.current
    void video?.requestFullscreen?.().catch(() => undefined)
  }

  const statusText = syncPlaying ? '同步播放中' : activeMode === 'sync' ? '同步已暂停' : `${activeMode === 'third' ? '第三人称' : '第一人称'}单独播放`
  function nudgeCalibration(deltaSec: number) {
    pauseBoth()
    setCalibrationDraft(previous => Number((previous + deltaSec).toFixed(3)))
  }

  async function saveCalibration() {
    await onCalibrationSave?.(Number(calibrationDraft.toFixed(3)))
  }

  async function resetCalibration() {
    pauseBoth()
    setCalibrationDraft(0)
    await onCalibrationReset?.()
  }

  const elapsedExperimentTime = Math.max(0, currentExperimentTime - timelineStart)
  const elapsedExperimentDuration = Math.max(0, timelineEnd - timelineStart)
  const calibrationDirty = Math.abs(calibrationDraft - savedCalibrationOffset) > 0.001
  const calibrationLabel = `${calibrationDraft >= 0 ? '+' : ''}${calibrationDraft.toFixed(1)}s`

  return (
    <div className="space-y-4">
      <div className="rounded-lg bg-slate-950 p-4 text-white">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={toggleSyncPlayback}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm font-black text-white transition hover:bg-blue-500"
            >
              {syncPlaying ? '暂停同步' : '同步播放'}
            </button>
            <div className="text-sm font-bold text-slate-300">
              实验时间 <span className="ml-2 font-mono text-lg text-white">{formatTime(elapsedExperimentTime)}</span>
              <span className="mx-2 text-slate-500">/</span>
              <span className="font-mono">{formatTime(elapsedExperimentDuration)}</span>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm font-bold text-slate-300">
            <span className={`h-2.5 w-2.5 rounded-full ${syncPlaying ? 'bg-emerald-400' : 'bg-slate-500'}`} />
            {statusText}
          </div>
        </div>
        <input
          type="range"
          min={timelineStart}
          max={timelineEnd}
          step={0.1}
          value={Math.min(currentExperimentTime, timelineEnd)}
          onChange={event => handleTimelineScrub(Number(event.currentTarget.value))}
          className="mt-4 w-full accent-blue-500"
          aria-label="dual video experiment timeline"
        />
        <div className="mt-2 flex justify-between text-xs font-bold text-slate-500">
          <span>源视频起点 {formatTime(timelineStart)}</span>
          <span>顶部按实验开始归零显示</span>
        </div>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-md bg-white/5 px-3 py-2">
          <div className="text-xs font-bold text-slate-300">
            offset <span className="ml-1 font-mono text-sm text-white">{calibrationLabel}</span>
            {calibrationDirty && <span className="ml-2 text-amber-300">unsaved</span>}
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => nudgeCalibration(-0.5)} className="rounded-md border border-white/10 bg-white/10 px-3 py-1.5 text-xs font-black text-white transition hover:bg-white/15">
              -0.5s
            </button>
            <button type="button" onClick={() => nudgeCalibration(0.5)} className="rounded-md border border-white/10 bg-white/10 px-3 py-1.5 text-xs font-black text-white transition hover:bg-white/15">
              +0.5s
            </button>
            <button type="button" onClick={() => void saveCalibration()} disabled={!onCalibrationSave || !calibrationDirty} className="rounded-md bg-emerald-500 px-3 py-1.5 text-xs font-black text-slate-950 transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50">
              保存 offset
            </button>
            <button type="button" onClick={() => void resetCalibration()} disabled={!onCalibrationReset && Math.abs(calibrationDraft) < 0.001} className="rounded-md border border-white/10 bg-white/10 px-3 py-1.5 text-xs font-black text-white transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-50">
              恢复默认
            </button>
          </div>
        </div>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <VideoPane
          kind="third"
          title="第三人称（俯视桌面）"
          url={thirdPersonUrl}
          refObject={thirdRef}
          offsetSec={offsets.third}
          timelineStart={timelineStart}
          duration={durations.third}
          playing={singlePlaying.third || (syncPlaying && Boolean(thirdRef.current && !thirdRef.current.paused))}
          onLoadedMetadata={() => updateDuration('third')}
          onTimeUpdate={() => updateTime('third')}
          onPlaySingle={() => void playSingle('third')}
          onPauseSingle={() => pauseSingle('third')}
          onScrub={value => scrubSingle('third', value)}
          onSyncFromPane={() => syncFromPane('third')}
          onFullscreen={() => requestPaneFullscreen('third')}
        />
        <VideoPane
          kind="first"
          title="第一人称（操作者视角）"
          url={firstPersonUrl}
          refObject={firstRef}
          offsetSec={offsets.first}
          timelineStart={timelineStart}
          duration={durations.first}
          playing={singlePlaying.first || (syncPlaying && Boolean(firstRef.current && !firstRef.current.paused))}
          onLoadedMetadata={() => updateDuration('first')}
          onTimeUpdate={() => updateTime('first')}
          onPlaySingle={() => void playSingle('first')}
          onPauseSingle={() => pauseSingle('first')}
          onScrub={value => scrubSingle('first', value)}
          onSyncFromPane={() => syncFromPane('first')}
          onFullscreen={() => requestPaneFullscreen('first')}
        />
      </div>
    </div>
  )
}

function VideoPane({
  kind,
  title,
  url,
  refObject,
  offsetSec,
  timelineStart,
  duration,
  playing,
  onLoadedMetadata,
  onTimeUpdate,
  onPlaySingle,
  onPauseSingle,
  onScrub,
  onSyncFromPane,
  onFullscreen,
}: {
  kind: VideoKind
  title: string
  url?: string
  refObject: RefObject<HTMLVideoElement>
  offsetSec: number
  timelineStart: number
  duration: number
  playing: boolean
  onLoadedMetadata: () => void
  onTimeUpdate: () => void
  onPlaySingle: () => void
  onPauseSingle: () => void
  onScrub: (localTime: number) => void
  onSyncFromPane: () => void
  onFullscreen: () => void
}) {
  const localTime = refObject.current?.currentTime || 0
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2 text-sm font-black text-slate-700">
        <span>{title}</span>
        <span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-bold text-slate-500">{kind === 'third' ? 'top view' : 'operator view'}</span>
      </div>
      {url ? (
        <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
          <video
            ref={refObject}
            src={url}
            className="aspect-video w-full bg-slate-950 object-contain"
            playsInline
            preload="metadata"
            onLoadedMetadata={onLoadedMetadata}
            onDurationChange={onLoadedMetadata}
            onTimeUpdate={onTimeUpdate}
            onPause={onPauseSingle}
          />
          <div className="space-y-2 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="text-xs font-bold text-slate-500">
                本地 {formatTime(localTime)} · 实验 +{formatTime(Math.max(0, localTime + offsetSec - timelineStart))}
              </div>
              <div className="flex flex-wrap gap-2">
                <button type="button" onClick={playing ? onPauseSingle : onPlaySingle} className="rounded-md bg-slate-900 px-3 py-1.5 text-xs font-black text-white transition hover:bg-slate-700">
                  {playing ? '暂停单路' : '单独播放'}
                </button>
                <button type="button" onClick={onSyncFromPane} className="rounded-md border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-black text-blue-700 transition hover:bg-blue-100">
                  同步到此处
                </button>
                <button type="button" onClick={onFullscreen} className="rounded-md border border-slate-200 bg-white px-3 py-1.5 text-xs font-black text-slate-700 transition hover:bg-slate-50">
                  全屏
                </button>
              </div>
            </div>
            <input
              type="range"
              min={0}
              max={Math.max(duration, localTime, 0.1)}
              step={0.1}
              value={Math.min(localTime, Math.max(duration, localTime, 0.1))}
              onChange={event => onScrub(Number(event.currentTarget.value))}
              className="w-full accent-slate-700"
              aria-label={`${kind} video local timeline`}
            />
          </div>
        </div>
      ) : (
        <div className="flex aspect-video items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 text-sm font-semibold text-slate-400">暂无视频</div>
      )}
    </div>
  )
}
