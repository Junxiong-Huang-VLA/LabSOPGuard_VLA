import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'

export interface DualVideoPlayerHandle {
  seekTo?: number
  token?: number
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

export default function DualVideoPlayer({
  firstPersonUrl,
  thirdPersonUrl,
  seekRequest,
  timeOffsetSec = 0,
}: {
  firstPersonUrl?: string
  thirdPersonUrl?: string
  seekRequest?: DualVideoPlayerHandle | null
  timeOffsetSec?: number
}) {
  const firstRef = useRef<HTMLVideoElement>(null)
  const thirdRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    if (seekRequest?.seekTo == null) return
    const relativeTime = Math.max(0, seekRequest.seekTo - timeOffsetSec)
    const cleanups = [firstRef.current, thirdRef.current]
      .filter((video): video is HTMLVideoElement => Boolean(video))
      .map(video => seekWhenReady(video, relativeTime))
      .filter((cleanup): cleanup is () => void => Boolean(cleanup))
    return () => cleanups.forEach(cleanup => cleanup())
  }, [seekRequest?.seekTo, seekRequest?.token, timeOffsetSec])

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <VideoPane title="第三人称（侧视台面）" url={thirdPersonUrl} refObject={thirdRef} />
      <VideoPane title="第一人称（操作者视角）" url={firstPersonUrl} refObject={firstRef} />
    </div>
  )
}

function VideoPane({ title, url, refObject }: { title: string; url?: string; refObject: RefObject<HTMLVideoElement> }) {
  return (
    <div className="space-y-2">
      <div className="text-sm font-black text-slate-700">{title}</div>
      {url ? (
        <video
          ref={refObject}
          src={url}
          className="aspect-video w-full rounded-lg bg-slate-950 object-contain"
          controls
          playsInline
          preload="metadata"
        />
      ) : (
        <div className="flex aspect-video items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 text-sm font-semibold text-slate-400">暂无视频</div>
      )}
    </div>
  )
}
