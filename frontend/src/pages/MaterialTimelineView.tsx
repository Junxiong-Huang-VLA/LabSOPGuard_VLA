import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Clock3, Filter, Layers3, Video } from 'lucide-react'
import { experimentApi } from '../api'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, primaryButtonClass, secondaryButtonClass } from '../components/EvidenceUI'
import ExperimentPageShell from '../components/ExperimentSideNav'
import { cleanDisplayText } from '../displayText'
import { experimentFileUrl } from '../mediaUrl'
import type { MaterialSearchItem } from '../types'

function itemTime(item: MaterialSearchItem) {
  return Number(item.timestamp_sec ?? item.time_start ?? item.local_timestamp_sec ?? 0)
}

function itemClip(item: MaterialSearchItem, experimentId?: string) {
  const paths = item.published_paths || {}
  return experimentFileUrl(item.clip_url || item.clip_file_path || paths.clip || undefined, experimentId)
}

export default function MaterialTimelineView() {
  const { id } = useParams<{ id: string }>()
  const [items, setItems] = useState<MaterialSearchItem[]>([])
  const [camera, setCamera] = useState('')
  const [kind, setKind] = useState('')
  const [pendingOnly, setPendingOnly] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!id) return
    setLoading(true)
    experimentApi.getMaterialTimeline(id, { limit: 1000 }, { force: true })
      .then(data => setItems((data.items || []) as MaterialSearchItem[]))
      .catch(() => setItems([]))
      .finally(() => setLoading(false))
  }, [id])

  const cameras = useMemo(() => Array.from(new Set(items.map(item => item.camera_id || item.stream_id).filter(Boolean))) as string[], [items])
  const kinds = useMemo(() => Array.from(new Set(items.map(item => item.event_type || item.event_types?.[0]).filter(Boolean))) as string[], [items])
  const filtered = useMemo(() => items
    .filter(item => !camera || item.camera_id === camera || item.stream_id === camera)
    .filter(item => !kind || item.event_type === kind || item.event_types?.includes(kind))
    .filter(item => !pendingOnly || String(item.review_status || item.payload?.review_status || '').toLowerCase().includes('pending'))
    .sort((a, b) => itemTime(a) - itemTime(b)), [camera, items, kind, pendingOnly])

  const content = (
    <div className="space-y-5">
      <PageHero
        eyebrow={id ? <Link to={`/experiments/${id}/materials`} className="hover:text-slate-900">证据包</Link> : 'Timeline'}
        title="多机位证据时间轴"
        description="按时间顺序查看不同机位、动作、对象和审核状态，辅助双视角证据对齐。"
        actions={id ? (
          <>
            <Link to={`/experiments/${id}/workspace`} className={secondaryButtonClass()}><ArrowLeft className="h-4 w-4" />工作台</Link>
            <Link to={`/experiments/${id}/materials`} className={primaryButtonClass('blue')}><Layers3 className="h-4 w-4" />证据库</Link>
          </>
        ) : null}
      />

      <section className="grid gap-4 md:grid-cols-3">
        <MetricTile label="时间点" value={items.length} helper={`${filtered.length} visible`} tone="blue" Icon={Clock3} />
        <MetricTile label="机位" value={cameras.length} helper="camera / stream" tone="cyan" Icon={Video} />
        <MetricTile label="片段" value={items.filter(item => Boolean(itemClip(item, id))).length} helper="clip assets" tone="emerald" Icon={Layers3} />
      </section>

      <EvidenceCard className="p-4">
        <div className="flex flex-wrap gap-2">
          <select value={camera} onChange={event => setCamera(event.target.value)} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-bold text-slate-700"><option value="">全部机位</option>{cameras.map(item => <option key={item} value={item}>{item}</option>)}</select>
          <select value={kind} onChange={event => setKind(event.target.value)} className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-bold text-slate-700"><option value="">全部类型</option>{kinds.map(item => <option key={item} value={item}>{item}</option>)}</select>
          <button type="button" onClick={() => setPendingOnly(value => !value)} className={`${pendingOnly ? primaryButtonClass('amber') : secondaryButtonClass('amber')}`}><Filter className="h-4 w-4" />只看待审核</button>
        </div>
      </EvidenceCard>

      {loading ? <EmptyEvidence title="正在加载时间轴..." /> : filtered.length === 0 ? <EmptyEvidence title="暂无时间轴事件" /> : (
        <EvidenceCard className="p-4">
          <div className="space-y-3">
            {filtered.map(item => (
              <div key={item.item_id || item.event_id} className="grid gap-3 rounded-lg border border-slate-200 p-3 md:grid-cols-[7rem_minmax(0,1fr)_10rem]">
                <div className="font-mono text-sm font-black text-blue-700">{itemTime(item).toFixed(2)}s</div>
                <div className="min-w-0">
                  <div className="font-black text-slate-950">{cleanDisplayText(item.display_name || item.event_type || item.item_id, '事件')}</div>
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    {[item.camera_id, item.stream_id, item.event_type, ...(item.object_labels || []), ...(item.actions || [])].filter(Boolean).slice(0, 8).map(label => <EvidenceBadge key={String(label)}>{cleanDisplayText(label)}</EvidenceBadge>)}
                  </div>
                </div>
                <div className="flex items-start justify-end gap-2">
                  {itemClip(item, id) && <a href={itemClip(item, id)} target="_blank" rel="noreferrer" className={secondaryButtonClass('blue')}>片段</a>}
                  {item.review_status && <EvidenceBadge tone="amber">{item.review_status}</EvidenceBadge>}
                </div>
              </div>
            ))}
          </div>
        </EvidenceCard>
      )}
    </div>
  )

  return id ? <ExperimentPageShell experimentId={id}>{content}</ExperimentPageShell> : content
}
