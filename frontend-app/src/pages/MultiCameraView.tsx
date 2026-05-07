import { useState, useEffect, useCallback, useRef } from 'react'
import { createPortal } from 'react-dom'
import axios from 'axios'

// ── Types ────────────────────────────────────────────────────────

interface CameraInfo { camera_id: string; label: string; sender_id: string; online: boolean; source: string }
interface Profile { stream_name: string; width: number; height: number; fps: number; pixel_format: string }
interface QualitySummary { camera_id?: string; samples?: number; loss_rate?: number; avg_fps_out?: number; avg_latency_ms?: number; avg_pkt_rate?: number }
interface CameraStats { camera_id: string; label: string; sender_id: string; online: boolean; source: string; status: string | null; profiles: Profile[]; quality: QualitySummary; jpeg_quality: number }
interface CaptureStatus { running: boolean; interval_sec: number; total_captured: number; per_camera: Record<string, number>; session_dir: string | null; started_at: string | null }
interface RecordingStatus { recording: boolean; cameras: Record<string, { path: string; frames: number }> }
interface Alert { ts: string; camera_id: string; type: string; message: string }
interface CaptureSession { session_id: string; path: string; total_frames: number; started_at?: string; stopped_at?: string; per_camera?: Record<string, number> }
function usePageVisible() {
  const [visible, setVisible] = useState(() => typeof document === 'undefined' || !document.hidden)
  useEffect(() => {
    const update = () => setVisible(!document.hidden)
    document.addEventListener('visibilitychange', update)
    return () => document.removeEventListener('visibilitychange', update)
  }, [])
  return visible
}

// ── Auto-reconnecting MJPEG ──────────────────────────────────────

function MjpegImage({ src, alt, className }: { src: string; alt: string; className: string }) {
  const [key, setKey] = useState(0)
  const [err, setErr] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout>>()
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current) }, [])
  const onError = useCallback(() => {
    setErr(true)
    timer.current = setTimeout(() => { setErr(false); setKey(k => k + 1) }, 3000)
  }, [])
  return (
    <>
      <img key={key} src={err ? undefined : src} alt={alt} className={`${className} ${err ? 'hidden' : 'block'}`} onError={onError} />
      {err && <div className="absolute inset-0 flex items-center justify-center text-gray-600 text-xs animate-pulse">reconnecting...</div>}
    </>
  )
}

// ── Camera card ──────────────────────────────────────────────────

function CameraCard({ camera, stats, captureCount, isRecording, onFullscreen, onSnapshot, onFeishuPush, feishuSending }: {
  camera: CameraInfo; stats?: CameraStats; captureCount?: number; isRecording: boolean
  onFullscreen: () => void
  onSnapshot: () => void
  onFeishuPush: () => void
  feishuSending: boolean
}) {
  const streamUrl = `/api/v1/cameras/${camera.camera_id}/stream`
  const p = stats?.profiles?.find(p => p.stream_name === 'rgb')
  const q = stats?.quality
  return (
    <div className="relative bg-black rounded overflow-hidden group">
      {/* top */}
      <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between px-2 py-1 bg-gradient-to-b from-black/80 to-transparent pointer-events-none">
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full inline-block ${camera.online ? 'bg-green-400 animate-pulse' : 'bg-red-500'}`} />
          <span className="text-white text-xs font-bold">{camera.label}</span>
          <span className="text-gray-500 text-[10px]">{camera.source === 'usb' ? 'USB' : 'WiFi'}</span>
          {isRecording && <span className="text-red-400 text-[10px] font-mono animate-pulse">REC</span>}
        </div>
        <div className="flex items-center gap-2">
          {p && <span className="text-gray-400 text-[10px] font-mono">{p.width}x{p.height}</span>}
          {p && <span className="text-green-400 text-[10px] font-mono">{p.fps.toFixed(1)} fps</span>}
          {q && q.loss_rate !== undefined && q.loss_rate > 0 && (
            <span className={`text-[10px] font-mono ${q.loss_rate > 0.2 ? 'text-red-400' : 'text-yellow-400'}`}>
              loss {(q.loss_rate * 100).toFixed(0)}%
            </span>
          )}
          {stats && <span className="text-gray-600 text-[10px] font-mono">q{stats.jpeg_quality}</span>}
        </div>
      </div>
      {/* video */}
      <div className="w-full bg-black relative cursor-pointer" style={{ aspectRatio: '16 / 9' }} onClick={onFullscreen}>
        <MjpegImage src={streamUrl} alt={camera.label} className="w-full h-full object-contain" />
      </div>
      {/* bottom */}
      <div className="absolute bottom-0 left-0 right-0 z-10 flex items-center justify-between px-2 py-1 bg-gradient-to-t from-black/80 to-transparent">
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-mono ${camera.online ? 'text-green-400' : 'text-red-400'}`}>{camera.online ? 'LIVE' : 'OFFLINE'}</span>
          {captureCount !== undefined && captureCount > 0 && <span className="text-blue-400 text-[10px] font-mono">cap:{captureCount}</span>}
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={e => { e.stopPropagation(); onFeishuPush() }}
            disabled={feishuSending}
            className="opacity-0 group-hover:opacity-100 transition-opacity text-amber-200 hover:text-white text-[10px] bg-amber-700/70 px-2 py-0.5 rounded pointer-events-auto disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {feishuSending ? 'Sending...' : 'Push Feishu'}
          </button>
          <button
            onClick={e => { e.stopPropagation(); onSnapshot() }}
            className="opacity-0 group-hover:opacity-100 transition-opacity text-gray-300 hover:text-white text-[10px] bg-black/60 px-2 py-0.5 rounded pointer-events-auto"
          >
            Screenshot
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Fullscreen ───────────────────────────────────────────────────

function FullscreenView({ camera, stats, onClose }: { camera: CameraInfo; stats?: CameraStats; onClose: () => void }) {
  const streamUrl = `/api/v1/cameras/${camera.camera_id}/stream`
  const p = stats?.profiles?.find(p => p.stream_name === 'rgb')
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose])
  return createPortal(
    <div className="fixed inset-0 z-[9999] bg-black flex flex-col">
      <div className="flex items-center justify-between px-4 py-2 bg-black/90 shrink-0">
        <div className="flex items-center gap-3">
          <span className={`w-3 h-3 rounded-full inline-block ${camera.online ? 'bg-green-400 animate-pulse' : 'bg-red-500'}`} />
          <span className="text-white text-lg font-bold">{camera.label}</span>
          <span className="text-gray-500 text-sm">{camera.camera_id}</span>
          {p && <span className="text-gray-300 text-sm font-mono">{p.width}x{p.height} @ {p.fps.toFixed(1)}fps</span>}
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-white p-2">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
        </button>
      </div>
      <div className="flex-1 min-h-0 flex items-center justify-center bg-black">
        <MjpegImage src={streamUrl} alt={camera.label} className="max-w-full max-h-full object-contain" />
      </div>
    </div>, document.body)
}

// ── Control panels ───────────────────────────────────────────────

function CapturePanel({ cameras, capStatus, onRefresh }: { cameras: CameraInfo[]; capStatus: CaptureStatus | null; onRefresh: () => void }) {
  const [interval, setIv] = useState(5)
  const [selCams, setSelCams] = useState<string[]>([])
  const [allSel, setAllSel] = useState(true)
  const running = capStatus?.running ?? false
  const toggle = (id: string) => { setSelCams(p => p.includes(id) ? p.filter(c => c !== id) : [...p, id]); setAllSel(false) }
  const start = async () => {
    const body: any = { interval_sec: interval, sync_timestamps: true }
    if (!allSel && selCams.length > 0) body.camera_ids = selCams
    await axios.post('/api/v1/cameras/capture/start', body)
    onRefresh()
  }
  const stop = async () => { await axios.post('/api/v1/cameras/capture/stop'); onRefresh() }
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-[10px] text-gray-500 mb-0.5 uppercase tracking-wider">Interval</label>
          <div className="flex items-center gap-1">
            <input type="number" min={0.5} step={0.5} value={interval} onChange={e => setIv(Number(e.target.value))} disabled={running} className="w-16 px-2 py-1 border border-gray-300 rounded text-sm disabled:bg-gray-100" />
            <span className="text-xs text-gray-400">sec</span>
          </div>
        </div>
        <div>
          <label className="block text-[10px] text-gray-500 mb-0.5 uppercase tracking-wider">Cameras</label>
          <div className="flex items-center gap-1 flex-wrap">
            <button onClick={() => { setAllSel(true); setSelCams([]) }} disabled={running} className={`px-2 py-0.5 rounded text-xs border ${allSel ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-300'} disabled:opacity-50`}>All</button>
            {cameras.map(c => <button key={c.camera_id} onClick={() => toggle(c.camera_id)} disabled={running} className={`px-2 py-0.5 rounded text-xs border ${!allSel && selCams.includes(c.camera_id) ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-300'} disabled:opacity-50`}>{c.label}</button>)}
          </div>
        </div>
        {running
          ? <button onClick={stop} className="px-4 py-1.5 bg-red-600 text-white rounded text-sm hover:bg-red-700">Stop</button>
          : <button onClick={start} className="px-4 py-1.5 bg-green-600 text-white rounded text-sm hover:bg-green-700">Start Capture</button>
        }
        {capStatus && <div className="flex items-center gap-2 text-xs text-gray-500">
          {running && <span className="text-green-600 font-medium animate-pulse">Capturing...</span>}
          <span>Frames: <strong className="text-gray-800">{capStatus.total_captured}</strong></span>
        </div>}
      </div>
      {capStatus && capStatus.total_captured > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1">{Object.entries(capStatus.per_camera).map(([k, v]) => <span key={k} className="text-[10px] bg-gray-100 px-1.5 py-0.5 rounded">{k}:{v}</span>)}</div>
      )}
    </div>
  )
}

function RecordingPanel({ cameras, recStatus, onRefresh }: { cameras: CameraInfo[]; recStatus: RecordingStatus | null; onRefresh: () => void }) {
  const [selCams, setSelCams] = useState<string[]>(cameras.map(c => c.camera_id))
  const recording = recStatus?.recording ?? false
  const start = async () => { await axios.post('/api/v1/cameras/recording/start', { camera_ids: selCams, fps: 15 }); onRefresh() }
  const stop = async () => { await axios.post('/api/v1/cameras/recording/stop'); onRefresh() }
  const toggle = (id: string) => setSelCams(p => p.includes(id) ? p.filter(c => c !== id) : [...p, id])
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-[10px] text-gray-500 mb-0.5 uppercase tracking-wider">Record cameras</label>
          <div className="flex items-center gap-1 flex-wrap">
            {cameras.map(c => <button key={c.camera_id} onClick={() => toggle(c.camera_id)} disabled={recording} className={`px-2 py-0.5 rounded text-xs border ${selCams.includes(c.camera_id) ? 'bg-red-600 text-white border-red-600' : 'bg-white text-gray-600 border-gray-300'} disabled:opacity-50`}>{c.label}</button>)}
          </div>
        </div>
        {recording
          ? <button onClick={stop} className="px-4 py-1.5 bg-gray-700 text-white rounded text-sm hover:bg-gray-600">Stop Recording</button>
          : <button onClick={start} className="px-4 py-1.5 bg-red-600 text-white rounded text-sm hover:bg-red-700">Record MP4</button>
        }
        {recStatus && recording && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-red-500 animate-pulse font-medium">Recording...</span>
            {Object.entries(recStatus.cameras).map(([k, v]) => <span key={k} className="text-gray-500">{k}: {v.frames}f</span>)}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Alert bar ────────────────────────────────────────────────────

function AlertBar({ alerts }: { alerts: Alert[] }) {
  if (alerts.length === 0) return null
  const recent = alerts.slice(-3)
  return (
    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-2 flex items-center gap-2 overflow-x-auto">
      <span className="text-yellow-600 text-xs font-bold shrink-0">ALERTS ({alerts.length})</span>
      {recent.map((a, i) => (
        <span key={i} className="text-xs text-yellow-800 bg-yellow-100 px-2 py-0.5 rounded whitespace-nowrap">
          {a.camera_id}: {a.message}
        </span>
      ))}
    </div>
  )
}

// ── History panel ────────────────────────────────────────────────

function HistoryPanel() {
  const [sessions, setSessions] = useState<CaptureSession[]>([])
  const [open, setOpen] = useState(false)
  const fetch = useCallback(async () => {
    try {
      const { data } = await axios.get<{ sessions: CaptureSession[] }>('/api/v1/cameras/capture/history')
      setSessions(data.sessions)
    } catch { /* ignore */ }
  }, [])
  useEffect(() => { if (open) fetch() }, [open, fetch])
  const del = async (id: string) => {
    if (!confirm(`Delete session ${id}?`)) return
    await axios.delete(`/api/v1/cameras/capture/history/${id}`)
    fetch()
  }
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3">
      <button onClick={() => setOpen(!open)} className="text-sm font-bold text-gray-700 flex items-center gap-1">
        Capture History {open ? '▼' : '▶'} <span className="text-gray-400 font-normal text-xs">{sessions.length > 0 ? `(${sessions.length} sessions)` : ''}</span>
      </button>
      {open && (
        <div className="mt-2 max-h-48 overflow-y-auto space-y-1">
          {sessions.length === 0 && <p className="text-xs text-gray-400">No capture sessions yet</p>}
          {sessions.map(s => (
            <div key={s.session_id} className="flex items-center justify-between bg-gray-50 px-2 py-1 rounded text-xs">
              <div className="flex items-center gap-2">
                <span className="font-mono text-gray-700">{s.session_id}</span>
                <span className="text-gray-500">{s.total_frames} frames</span>
                {s.per_camera && <span className="text-gray-400">{Object.keys(s.per_camera).length} cams</span>}
              </div>
              <button onClick={() => del(s.session_id)} className="text-red-400 hover:text-red-600 px-1">Delete</button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Main page ────────────────────────────────────────────────────

export default function MultiCameraView() {
  const pageVisible = usePageVisible()
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [stats, setStats] = useState<Record<string, CameraStats>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [fullscreen, setFullscreen] = useState<string | null>(null)
  const [capStatus, setCapStatus] = useState<CaptureStatus | null>(null)
  const [recStatus, setRecStatus] = useState<RecordingStatus | null>(null)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [tab, setTab] = useState<'capture' | 'record' | 'history'>('capture')
  const [feishuSendingCameraId, setFeishuSendingCameraId] = useState<string | null>(null)
  const [feishuPushNotice, setFeishuPushNotice] = useState<{ level: 'success' | 'error'; text: string } | null>(null)

  const fetchCameras = useCallback(async () => { try { const { data } = await axios.get<{ cameras: CameraInfo[] }>('/api/v1/cameras'); setCameras(data.cameras); setError(null) } catch (e: any) { setError(e?.message || 'Failed') } finally { setLoading(false) } }, [])
  const fetchStats = useCallback(async () => { try { const { data } = await axios.get<{ cameras: CameraStats[] }>('/api/v1/cameras/stats'); const m: Record<string, CameraStats> = {}; for (const s of data.cameras) m[s.camera_id] = s; setStats(m) } catch {} }, [])
  const fetchCapture = useCallback(async () => { try { const { data } = await axios.get<CaptureStatus>('/api/v1/cameras/capture/status'); setCapStatus(data) } catch {} }, [])
  const fetchRecording = useCallback(async () => { try { const { data } = await axios.get<RecordingStatus>('/api/v1/cameras/recording/status'); setRecStatus(data) } catch {} }, [])
  const fetchAlerts = useCallback(async () => { try { const { data } = await axios.get<{ alerts: Alert[] }>('/api/v1/cameras/capture/alerts'); setAlerts(data.alerts) } catch {} }, [])

  useEffect(() => { if (!pageVisible) return; fetchCameras(); const id = setInterval(fetchCameras, 5000); return () => clearInterval(id) }, [fetchCameras, pageVisible])
  useEffect(() => { if (!pageVisible) return; fetchStats(); const id = setInterval(fetchStats, 2000); return () => clearInterval(id) }, [fetchStats, pageVisible])
  useEffect(() => { if (!pageVisible) return; fetchCapture(); const id = setInterval(fetchCapture, 2000); return () => clearInterval(id) }, [fetchCapture, pageVisible])
  useEffect(() => { if (!pageVisible) return; fetchRecording(); const id = setInterval(fetchRecording, 2000); return () => clearInterval(id) }, [fetchRecording, pageVisible])
  useEffect(() => { if (!pageVisible) return; fetchAlerts(); const id = setInterval(fetchAlerts, 3000); return () => clearInterval(id) }, [fetchAlerts, pageVisible])
  useEffect(() => {
    if (!feishuPushNotice) return
    const timer = setTimeout(() => setFeishuPushNotice(null), 4000)
    return () => clearTimeout(timer)
  }, [feishuPushNotice])

  const onlineCount = cameras.filter(c => c.online).length
  const fullscreenCam = cameras.find(c => c.camera_id === fullscreen)
  const topRow = cameras.slice(0, 3)
  const bottomRow = cameras.slice(3)
  const snapshot = async (id: string) => { try { await axios.get(`/api/v1/cameras/${id}/snapshot`, { params: { save: true } }) } catch {} }
  const pushSnapshotToFeishu = async (camera: CameraInfo) => {
    if (feishuSendingCameraId === camera.camera_id) return
    setFeishuSendingCameraId(camera.camera_id)
    const triggerTime = new Date().toLocaleString('zh-CN', { hour12: false })
    const message = `手动触发飞书推送\n摄像头: ${camera.label} (${camera.camera_id})\n时间: ${triggerTime}`
    try {
      await axios.post('/api/v1/cameras/feishu/snapshot', {
        camera_id: camera.camera_id,
        message,
      })
      setFeishuPushNotice({
        level: 'success',
        text: `已推送到飞书：${camera.label} (${camera.camera_id})`,
      })
    } catch (error: unknown) {
      const detail = axios.isAxiosError(error)
        ? ((error.response?.data as { detail?: string } | undefined)?.detail || error.message)
        : 'unknown error'
      setFeishuPushNotice({
        level: 'error',
        text: `飞书推送失败：${detail}`,
      })
    } finally {
      setFeishuSendingCameraId(null)
    }
  }

  if (loading) return <div className="flex items-center justify-center h-64"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" /></div>

  return (
    <div className="-mx-4 sm:-mx-6 lg:-mx-8">
      {/* header */}
      <div className="flex items-center justify-between mb-2 px-4 sm:px-6 lg:px-8">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Multi-Camera Monitor</h2>
          <p className="text-sm text-gray-500">{cameras.length} configured, {onlineCount} online</p>
        </div>
        <button onClick={() => { setLoading(true); fetchCameras() }} className="px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm">Refresh</button>
      </div>

      {error && <div className="mx-4 sm:mx-6 lg:mx-8 mb-2 p-2 bg-red-50 border border-red-200 rounded text-red-700 text-sm">{error}</div>}
      {feishuPushNotice && (
        <div
          className={`mx-4 sm:mx-6 lg:mx-8 mb-2 p-2 border rounded text-sm ${
            feishuPushNotice.level === 'success'
              ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
              : 'bg-red-50 border-red-200 text-red-700'
          }`}
        >
          {feishuPushNotice.text}
        </div>
      )}

      {/* alerts */}
      <div className="mx-4 sm:mx-6 lg:mx-8 mb-2"><AlertBar alerts={alerts} /></div>

      {/* tabs */}
      <div className="mx-4 sm:mx-6 lg:mx-8 mb-2 flex gap-1">
        {(['capture', 'record', 'history'] as const).map(t => (
          <button key={t} onClick={() => setTab(t)} className={`px-3 py-1 rounded text-xs font-medium ${tab === t ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>
            {t === 'capture' ? 'Frame Capture' : t === 'record' ? 'Video Record' : 'History'}
          </button>
        ))}
      </div>

      {/* panels */}
      <div className="mx-4 sm:mx-6 lg:mx-8 mb-2">
        {tab === 'capture' && <CapturePanel cameras={cameras} capStatus={capStatus} onRefresh={fetchCapture} />}
        {tab === 'record' && <RecordingPanel cameras={cameras} recStatus={recStatus} onRefresh={fetchRecording} />}
        {tab === 'history' && <HistoryPanel />}
      </div>

      {/* video grid */}
      {cameras.length === 0 ? (
        <div className="flex items-center justify-center h-64 text-gray-500 text-sm">No cameras detected</div>
      ) : (
        <div className="bg-gray-900 p-1 flex flex-col gap-1">
          {topRow.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-1">
              {topRow.map(c => <CameraCard key={c.camera_id} camera={c} stats={stats[c.camera_id]} captureCount={capStatus?.per_camera[c.camera_id]} isRecording={!!recStatus?.cameras[c.camera_id]} onFullscreen={() => setFullscreen(c.camera_id)} onSnapshot={() => snapshot(c.camera_id)} onFeishuPush={() => pushSnapshotToFeishu(c)} feishuSending={feishuSendingCameraId === c.camera_id} />)}
            </div>
          )}
          {bottomRow.length > 0 && (
            <div className="flex justify-center gap-1 flex-col sm:flex-row">
              {bottomRow.map(c => <div key={c.camera_id} className="sm:w-1/2"><CameraCard camera={c} stats={stats[c.camera_id]} captureCount={capStatus?.per_camera[c.camera_id]} isRecording={!!recStatus?.cameras[c.camera_id]} onFullscreen={() => setFullscreen(c.camera_id)} onSnapshot={() => snapshot(c.camera_id)} onFeishuPush={() => pushSnapshotToFeishu(c)} feishuSending={feishuSendingCameraId === c.camera_id} /></div>)}
            </div>
          )}
        </div>
      )}

      {fullscreenCam && <FullscreenView camera={fullscreenCam} stats={stats[fullscreenCam.camera_id]} onClose={() => setFullscreen(null)} />}
    </div>
  )
}
