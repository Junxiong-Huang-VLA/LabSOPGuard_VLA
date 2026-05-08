import { useEffect, useRef } from 'react'

export interface DualVideoPlayerHandle {
  seekTo?: number
  token?: number
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
    for (const ref of [firstRef, thirdRef]) {
      if (ref.current) ref.current.currentTime = relativeTime
    }
  }, [seekRequest?.seekTo, seekRequest?.token, timeOffsetSec])

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <VideoPane title="第三人称（俯视桌面）" url={thirdPersonUrl} refObject={thirdRef} />
      <VideoPane title="第一人称（操作者视角）" url={firstPersonUrl} refObject={firstRef} />
    </div>
  )
}

function VideoPane({ title, url, refObject }: { title: string; url?: string; refObject: React.RefObject<HTMLVideoElement> }) {
  return (
    <div className="space-y-2">
      <div className="text-sm font-black text-slate-700">{title}</div>
      {url ? (
        <video ref={refObject} src={url} className="aspect-video w-full rounded-lg bg-slate-950 object-contain" controls preload="metadata" />
      ) : (
        <div className="flex aspect-video items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 text-sm font-semibold text-slate-400">暂无视频</div>
      )}
    </div>
  )
}
