import { useCallback, useEffect, useRef, useState } from 'react'
import axios from 'axios'

interface Violation {
  ts: number
  types: string[]
  lab_coat_conf: number
  gloves_conf: number
}

interface PtzStatus {
  started: boolean
  start_error: string | null
  state: 'idle' | 'tracking' | 'lost' | 'manual'
  fps: number
  mqtt_connected: boolean
  mqtt_broker?: string
  mqtt_port?: number
  mqtt_topic?: string
  mqtt_remote_client_count?: number
  mqtt_remote_clients?: string[]
  tracking_require_mqtt_connected?: boolean
  tracking_require_remote_mqtt_client?: boolean
  ptz_pitch: number
  ptz_yaw: number
  ptz_speed: number
  persons_count: number
  compliance: {
    mode?: string
    has_lab_coat: boolean
    has_gloves: boolean
    lab_coat_conf: number
    gloves_conf: number
    compliant: boolean
    glove_conf_threshold?: number
    detect_lab_coat?: boolean
  }
  alert: { active: boolean; reasons: string[] }
  violations: Violation[]
  error: string | null
  timestamp: number
}

const REASON_TEXT: Record<string, string> = {
  no_lab_coat: '未穿白大褂',
  no_gloves: '未戴手套',
}

function usePageVisible() {
  const [visible, setVisible] = useState(() => typeof document === 'undefined' || !document.hidden)
  useEffect(() => {
    const update = () => setVisible(!document.hidden)
    document.addEventListener('visibilitychange', update)
    return () => document.removeEventListener('visibilitychange', update)
  }, [])
  return visible
}
const DEFAULT_STATUS: PtzStatus = {
  started: false,
  start_error: null,
  state: 'idle',
  fps: 0,
  mqtt_connected: false,
  mqtt_broker: '127.0.0.1',
  mqtt_port: 1883,
  mqtt_topic: 'gimbal',
  mqtt_remote_client_count: 0,
  mqtt_remote_clients: [],
  tracking_require_mqtt_connected: true,
  tracking_require_remote_mqtt_client: true,
  ptz_pitch: 90,
  ptz_yaw: 90,
  ptz_speed: 1.0,
  persons_count: 0,
  compliance: {
    mode: 'gloves_only',
    has_lab_coat: false,
    has_gloves: false,
    lab_coat_conf: 0,
    gloves_conf: 0,
    compliant: false,
    glove_conf_threshold: 0.55,
    detect_lab_coat: false,
  },
  alert: { active: false, reasons: [] },
  violations: [],
  error: null,
  timestamp: 0,
}

function StatusPill({ label, ok, text }: { label: string; ok: boolean; text?: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 px-3 bg-gray-50 rounded">
      <span className="text-xs text-gray-600">{label}</span>
      <span
        className={`text-xs font-semibold ${ok ? 'text-green-600' : 'text-red-600'}`}
      >
        {text ?? (ok ? 'OK' : 'OFF')}
      </span>
    </div>
  )
}

export default function PtzTracker() {
  const pageVisible = usePageVisible()
  const [status, setStatus] = useState<PtzStatus>(DEFAULT_STATUS)
  const [busy, setBusy] = useState(false)
  const [imgError, setImgError] = useState(false)
  const [streamKey, setStreamKey] = useState(0)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>()
  const imgRef = useRef<HTMLImageElement>(null)

  const streamUrl = `/api/v1/ptz-tracker/stream?t=${streamKey}`

  const fetchStatus = useCallback(async () => {
    try {
      const { data } = await axios.get<PtzStatus>('/api/v1/ptz-tracker/status')
      setStatus(data)
    } catch {
      // ignore polling errors
    }
  }, [])

  useEffect(() => {
    if (!pageVisible) return
    fetchStatus()
    const id = setInterval(fetchStatus, 1000)
    return () => clearInterval(id)
  }, [fetchStatus, pageVisible])

  useEffect(() => {
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    }
  }, [])

  useEffect(() => {
    if (!pageVisible || !status.started) return
    setImgError(false)
    setStreamKey(Date.now())
  }, [pageVisible, status.started])

  const call = useCallback(
    async (path: string, body?: Record<string, unknown>) => {
      setBusy(true)
      try {
        const { data } = await axios.post<PtzStatus>(`/api/v1/ptz-tracker${path}`, body ?? {})
        setStatus(data)
      } catch (e: any) {
        const msg = e?.response?.data?.detail || e?.message || String(e)
        alert(`操作失败: ${msg}`)
      } finally {
        setBusy(false)
      }
    },
    []
  )

  const onStart = () => {
    setImgError(false)
    setStreamKey(Date.now())
    call('/start')
  }
  const onStartTrack = () => {
    if (status.tracking_require_mqtt_connected && !status.mqtt_connected) {
      alert('MQTT broker 未连接，不能开始自动跟踪。')
      return
    }
    if (
      status.tracking_require_remote_mqtt_client &&
      (status.mqtt_remote_client_count ?? 0) <= 0
    ) {
      alert('未检测到云台控制端连接 MQTT broker，不能开始自动跟踪。')
      return
    }
    setImgError(false)
    setStreamKey(Date.now())
    call('/track/start')
  }
  const onStopTrack = () => call('/track/stop')
  const onCenter = () => call('/ptz/center')
  const onManual = () => call('/ptz/manual')
  const onMove = (direction: 'up' | 'down' | 'left' | 'right') =>
    call('/ptz/move', { direction })
  const onSpeed = (speed: number) => call('/ptz/speed', { speed })

  const onReloadStream = () => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    setImgError(false)
    setStreamKey(Date.now())
  }

  const onStreamError = () => {
    setImgError(true)
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    reconnectTimer.current = setTimeout(() => {
      setImgError(false)
      setStreamKey(Date.now())
    }, 1500)
  }

  const onClearAlerts = () => call('/alerts/clear')

  const stateColor = {
    idle: 'text-gray-500',
    tracking: 'text-green-600',
    lost: 'text-red-600',
    manual: 'text-yellow-600',
  }[status.state]
  const remoteClientCount = status.mqtt_remote_client_count ?? 0
  const remoteClientReady =
    !status.tracking_require_remote_mqtt_client || remoteClientCount > 0
  const canStartTracking =
    status.started &&
    (!status.tracking_require_mqtt_connected || status.mqtt_connected) &&
    remoteClientReady

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">云台人体目标跟随</h2>
          <p className="text-sm text-gray-500 mt-1">
            有线云台 · 自动人体跟踪 · 白大褂/手套合规检测
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={onStart}
            disabled={busy || status.started}
            className="px-3 py-2 bg-indigo-600 text-white text-sm rounded hover:bg-indigo-700 disabled:opacity-50"
          >
            {status.started ? '已启动' : '启动服务'}
          </button>
          <button
            onClick={onReloadStream}
            className="px-3 py-2 bg-gray-100 text-gray-700 text-sm rounded hover:bg-gray-200"
          >
            刷新视频
          </button>
        </div>
      </div>

      {status.start_error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
          启动错误: {status.start_error}
        </div>
      )}

      {status.started && (!status.mqtt_connected || !remoteClientReady) && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded text-yellow-800 text-sm">
          PTZ 自动跟踪未就绪：MQTT {status.mqtt_connected ? '已连接' : '未连接'}，
          云台客户端 {remoteClientCount} 个。Broker: {status.mqtt_broker ?? '127.0.0.1'}:
          {status.mqtt_port ?? 1883}，Topic: {status.mqtt_topic ?? 'gimbal'}
        </div>
      )}

      {status.alert.active && (
        <div className="mb-4 p-4 bg-red-600 text-white rounded-lg shadow-lg animate-pulse flex items-center justify-between">
          <div>
            <div className="text-lg font-bold">⚠ PPE 违规告警</div>
            <div className="text-sm mt-1">
              {status.alert.reasons.map((r) => REASON_TEXT[r] ?? r).join('、')}
            </div>
          </div>
          <button
            onClick={onClearAlerts}
            className="px-3 py-2 bg-white text-red-600 text-sm rounded font-semibold hover:bg-red-50"
          >
            清除
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* 左：视频 */}
        <div className="lg:col-span-2">
          <div className="relative bg-gray-900 rounded-lg overflow-hidden border border-gray-700 aspect-video">
            {!status.started ? (
              <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-2">
                <span className="text-sm">PTZ service stopped</span>
              </div>
            ) : !imgError ? (
              <img
                ref={imgRef}
                src={streamUrl}
                alt="ptz stream"
                className="w-full h-full object-contain"
                onError={onStreamError}
                onLoad={() => setImgError(false)}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-2">
                <span className="text-sm">视频流连接中...</span>
                <button
                  onClick={onReloadStream}
                  className="px-3 py-1 bg-gray-700 text-white text-xs rounded hover:bg-gray-600"
                >
                  重试
                </button>
              </div>
            )}
          </div>
          <div className="mt-2 text-xs text-gray-500 flex gap-4">
            <span>State: <span className={`font-semibold ${stateColor}`}>{status.state.toUpperCase()}</span></span>
            <span>FPS: {status.fps.toFixed(1)}</span>
            <span>Persons: {status.persons_count}</span>
            <span>PTZ: P{status.ptz_pitch} / Y{status.ptz_yaw}</span>
          </div>
        </div>

        {/* 右：控制面板 */}
        <div className="space-y-4">
          {/* 状态 */}
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">系统状态</h3>
            <div className="space-y-2">
              <StatusPill label="服务" ok={status.started} text={status.started ? 'RUNNING' : 'STOPPED'} />
              <StatusPill label="MQTT 云台" ok={status.mqtt_connected} text={status.mqtt_connected ? 'CONNECTED' : 'DISCONNECTED'} />
              <StatusPill
                label="云台客户端"
                ok={remoteClientReady}
                text={`${remoteClientCount} ONLINE`}
              />
              <StatusPill
                label="白大褂"
                ok
                text="PAUSED"
              />
              <StatusPill
                label="手套"
                ok={status.compliance.has_gloves}
                text={`${status.compliance.has_gloves ? 'YES' : 'NO'} (${status.compliance.gloves_conf.toFixed(2)} / ≥${(status.compliance.glove_conf_threshold ?? 0.55).toFixed(2)})`}
              />
              <StatusPill
                label="手套检测"
                ok={status.compliance.compliant}
                text={status.compliance.compliant ? 'GLOVES OK' : 'NO GLOVES'}
              />
            </div>
          </div>

          {/* 跟踪控制 */}
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">跟踪控制</h3>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={onStartTrack}
                disabled={busy || !canStartTracking}
                className="px-3 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-700 disabled:opacity-50"
                title={!canStartTracking ? 'MQTT broker 或云台控制端未就绪' : undefined}
              >
                开始跟踪
              </button>
              <button
                onClick={onStopTrack}
                disabled={busy}
                className="px-3 py-2 bg-gray-500 text-white text-sm rounded hover:bg-gray-600 disabled:opacity-50"
              >
                停止跟踪
              </button>
              <button
                onClick={onCenter}
                disabled={busy}
                className="px-3 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
              >
                回正
              </button>
              <button
                onClick={onManual}
                disabled={busy}
                className="px-3 py-2 bg-yellow-500 text-white text-sm rounded hover:bg-yellow-600 disabled:opacity-50"
              >
                手动模式
              </button>
            </div>
          </div>

          {/* 手动方向 */}
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">云台方向</h3>
            <div className="grid grid-cols-3 gap-2 max-w-[200px] mx-auto">
              <div />
              <button
                onClick={() => onMove('up')}
                disabled={busy}
                className="p-2 bg-gray-100 hover:bg-gray-200 rounded disabled:opacity-50"
              >
                ↑
              </button>
              <div />
              <button
                onClick={() => onMove('left')}
                disabled={busy}
                className="p-2 bg-gray-100 hover:bg-gray-200 rounded disabled:opacity-50"
              >
                ←
              </button>
              <button
                onClick={onCenter}
                disabled={busy}
                className="p-2 bg-blue-100 hover:bg-blue-200 rounded disabled:opacity-50 text-xs"
              >
                中
              </button>
              <button
                onClick={() => onMove('right')}
                disabled={busy}
                className="p-2 bg-gray-100 hover:bg-gray-200 rounded disabled:opacity-50"
              >
                →
              </button>
              <div />
              <button
                onClick={() => onMove('down')}
                disabled={busy}
                className="p-2 bg-gray-100 hover:bg-gray-200 rounded disabled:opacity-50"
              >
                ↓
              </button>
              <div />
            </div>
          </div>

          {/* 速度 */}
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-gray-900">速度</h3>
              <span className="text-sm text-gray-600">{status.ptz_speed.toFixed(1)}x</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="3.0"
              step="0.1"
              value={status.ptz_speed}
              onChange={(e) => onSpeed(parseFloat(e.target.value))}
              disabled={busy}
              className="w-full"
            />
          </div>

          {/* 违规历史 */}
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-900">
                违规历史 ({status.violations.length})
              </h3>
              {status.violations.length > 0 && (
                <button
                  onClick={onClearAlerts}
                  className="text-xs text-gray-500 hover:text-gray-700 underline"
                >
                  清空
                </button>
              )}
            </div>
            {status.violations.length === 0 ? (
              <div className="text-xs text-gray-400 text-center py-4">暂无记录</div>
            ) : (
              <div className="max-h-48 overflow-y-auto space-y-2">
                {[...status.violations].reverse().map((v, i) => (
                  <div
                    key={`${v.ts}-${i}`}
                    className="text-xs p-2 bg-red-50 border border-red-100 rounded"
                  >
                    <div className="font-medium text-red-700">
                      {v.types.map((t) => REASON_TEXT[t] ?? t).join('、')}
                    </div>
                    <div className="text-gray-500 mt-0.5">
                      {new Date(v.ts * 1000).toLocaleTimeString()}
                      {' · '}
                      手套 {v.gloves_conf.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
