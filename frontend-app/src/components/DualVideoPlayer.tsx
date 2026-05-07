import { useEffect, useRef } from 'react'

export interface DualVideoPlayerHandle {
  seekTo?: number
  token?: number
}

export default function DualVideoPlayer({
  firstPersonUrl,
  thirdPersonUrl,
  seekRequest,
}: {
  firstPersonUrl?: string
  thirdPersonUrl?: string
  seekRequest?: DualVideoPlayerHandle | null
}) {
  const firstRef = useRef<HTMLVideoElement>(null)
  const thirdRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    if (seekRequest?.seekTo == null) return
    for (const ref of [firstRef, thirdRef]) {
      if (ref.current) ref.current.currentTime = Math.max(0, seekRequest.seekTo)
    }
  }, [seekRequest?.seekTo, seekRequest?.token])

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <VideoPane title="第一视角" url={firstPersonUrl} refObject={firstRef} />
      <VideoPane title="第三视角" url={thirdPersonUrl} refObject={thirdRef} />
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
