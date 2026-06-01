import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertCircle, Download, Square, X } from 'lucide-react'

type RecorderStatus = 'idle' | 'requesting' | 'recording' | 'saving'

type DownloadState = {
  url: string
  fileName: string
}

type ExtendedDisplayMediaOptions = DisplayMediaStreamOptions & {
  preferCurrentTab?: boolean
  selfBrowserSurface?: 'include' | 'exclude'
  surfaceSwitching?: 'include' | 'exclude'
  systemAudio?: 'include' | 'exclude'
  monitorTypeSurfaces?: 'include' | 'exclude'
}

const RECORDING_FRAME_RATE = 60
const RECORDING_VIDEO_BITS_PER_SECOND = 8_000_000

function formatDuration(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

function recordingFileName() {
  const stamp = new Date().toISOString().replace(/\.\d+Z$/, '').replace(/[:T]/g, '-')
  return `realityloop-frontend-${stamp}.webm`
}

function supportedMimeType() {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') return ''
  return [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ].find(type => MediaRecorder.isTypeSupported(type)) || ''
}

function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach(track => track.stop())
}

function saveBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  document.body.appendChild(link)
  link.click()
  link.remove()
  return url
}

export default function FrontendRecorder() {
  const [status, setStatus] = useState<RecorderStatus>('idle')
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState('')
  const [download, setDownload] = useState<DownloadState | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const startedAtRef = useRef(0)
  const statusRef = useRef<RecorderStatus>('idle')
  const downloadUrlRef = useRef('')

  useEffect(() => {
    statusRef.current = status
  }, [status])

  const clearDownload = useCallback(() => {
    if (downloadUrlRef.current) URL.revokeObjectURL(downloadUrlRef.current)
    downloadUrlRef.current = ''
    setDownload(null)
  }, [])

  const finishRecording = useCallback((recorder: MediaRecorder) => {
    stopStream(streamRef.current)
    streamRef.current = null
    recorderRef.current = null

    const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'video/webm' })
    chunksRef.current = []
    if (!blob.size) {
      setStatus('idle')
      setError('录制内容为空，请重新开始录制。')
      return
    }

    clearDownload()
    const fileName = recordingFileName()
    const url = saveBlob(blob, fileName)
    downloadUrlRef.current = url
    setDownload({ url, fileName })
    setStatus('idle')
  }, [clearDownload])

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current
    if (!recorder || recorder.state === 'inactive') return
    setStatus('saving')
    recorder.stop()
  }, [])

  const startRecording = useCallback(async () => {
    if (statusRef.current !== 'idle') return
    setError('')

    if (!navigator.mediaDevices?.getDisplayMedia) {
      setError('当前浏览器不支持前端页录制。')
      return
    }
    if (typeof MediaRecorder === 'undefined') {
      setError('当前浏览器不支持视频编码录制。')
      return
    }

    setStatus('requesting')
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          displaySurface: 'browser',
          frameRate: { ideal: RECORDING_FRAME_RATE, max: RECORDING_FRAME_RATE },
        },
        audio: false,
        preferCurrentTab: true,
        selfBrowserSurface: 'include',
        surfaceSwitching: 'exclude',
        systemAudio: 'exclude',
        monitorTypeSurfaces: 'exclude',
      } as ExtendedDisplayMediaOptions)

      const videoTrack = stream.getVideoTracks()[0]
      const settings = videoTrack?.getSettings() as MediaTrackSettings & { displaySurface?: string }
      if (settings.displaySurface && settings.displaySurface !== 'browser') {
        stopStream(stream)
        setStatus('idle')
        setError('请选择当前标签页录制，不能选择窗口或整个屏幕。')
        return
      }

      const mimeType = supportedMimeType()
      const recorder = new MediaRecorder(stream, {
        ...(mimeType ? { mimeType } : {}),
        videoBitsPerSecond: RECORDING_VIDEO_BITS_PER_SECOND,
      })
      chunksRef.current = []
      recorderRef.current = recorder
      streamRef.current = stream
      startedAtRef.current = Date.now()
      setElapsed(0)

      recorder.ondataavailable = event => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstop = () => finishRecording(recorder)
      videoTrack?.addEventListener('ended', () => {
        if (recorderRef.current?.state === 'recording') stopRecording()
      })

      clearDownload()
      recorder.start(1000)
      setStatus('recording')
    } catch (caught) {
      stopStream(streamRef.current)
      streamRef.current = null
      recorderRef.current = null
      chunksRef.current = []
      setStatus('idle')
      const message = caught instanceof Error ? caught.message : ''
      if (message && message.toLowerCase().includes('permission')) {
        setError('录制权限未开启。')
      } else if (message) {
        setError(message)
      } else {
        setError('录制已取消。')
      }
    }
  }, [clearDownload, finishRecording, stopRecording])

  useEffect(() => {
    if (status !== 'recording') return undefined
    const timer = window.setInterval(() => {
      setElapsed(Math.max(0, Math.floor((Date.now() - startedAtRef.current) / 1000)))
    }, 500)
    return () => window.clearInterval(timer)
  }, [status])

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if (!event.ctrlKey || !event.altKey || event.key.toLowerCase() !== 'r' || event.repeat) return
      event.preventDefault()
      if (recorderRef.current?.state === 'recording') {
        stopRecording()
      } else {
        void startRecording()
      }
    }
    window.addEventListener('keydown', handleShortcut)
    return () => window.removeEventListener('keydown', handleShortcut)
  }, [startRecording, stopRecording])

  useEffect(() => () => {
    stopStream(streamRef.current)
    if (downloadUrlRef.current) URL.revokeObjectURL(downloadUrlRef.current)
  }, [])

  const recording = status === 'recording'
  const showStatus = status !== 'idle'

  if (!showStatus && !error && !download) return null

  return (
    <div className="fixed bottom-5 right-5 z-[80] flex max-w-[calc(100vw-2rem)] flex-col items-end gap-2 print:hidden">
      {error ? (
        <div role="alert" className="flex max-w-sm items-start gap-2 rounded-[var(--ui-radius-md)] border border-red-200 bg-white px-3 py-2 text-sm font-semibold text-red-700 shadow-[var(--ui-shadow-subtle)]">
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
          <span className="min-w-0">{error}</span>
          <button type="button" onClick={() => setError('')} className="ml-1 rounded p-0.5 text-red-500 hover:bg-red-50" aria-label="关闭录制提示">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ) : null}

      {download ? (
        <div className="flex items-center gap-2 rounded-[var(--ui-radius-md)] border border-emerald-200 bg-white px-3 py-2 text-sm font-semibold text-emerald-700 shadow-[var(--ui-shadow-subtle)]">
          <span>录制已保存</span>
          <a href={download.url} download={download.fileName} className="inline-flex h-7 items-center gap-1 rounded-md bg-emerald-600 px-2 text-xs text-white hover:bg-emerald-700">
            <Download className="h-3.5 w-3.5" />
            下载
          </a>
          <button type="button" onClick={clearDownload} className="rounded p-0.5 text-emerald-600 hover:bg-emerald-50" aria-label="关闭录制结果">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ) : null}

      {showStatus ? (
        <div
          role="status"
          aria-live="polite"
          className={`inline-flex h-10 items-center gap-2 rounded-full border px-4 text-sm font-black shadow-[0_16px_40px_rgba(15,23,42,0.16)] ${
          recording
            ? 'border-red-200 bg-red-600 text-white'
            : 'border-[color:var(--ui-border)] bg-white/95 text-[color:var(--ui-text)] backdrop-blur'
        }`}
        >
          {recording ? (
            <>
              <span className="h-2.5 w-2.5 rounded-full bg-white" />
              <span>录制中 {formatDuration(elapsed)}</span>
              <Square className="h-4 w-4 fill-current" />
            </>
          ) : (
            <span>{status === 'requesting' ? '准备录制' : '保存中'}</span>
          )}
        </div>
      ) : null}
    </div>
  )
}
