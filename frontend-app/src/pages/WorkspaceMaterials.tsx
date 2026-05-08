import { FormEvent, useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { AlertTriangle, Boxes, DatabaseZap, Film, Image, Search } from 'lucide-react'
import { workspaceMaterialApi } from '../api'
import { EmptyEvidence, EvidenceBadge, EvidenceCard, MetricTile, PageHero, primaryButtonClass, toneForStatus } from '../components/EvidenceUI'
import { cleanDisplayText } from '../displayText'
import { experimentFileUrl } from '../mediaUrl'
import type { MaterialSearchItem, WorkspacePublishedHealthResponse } from '../types'

function itemPreview(item: MaterialSearchItem) {
  const paths = item.published_paths || {}
  return experimentFileUrl(item.preview_url || item.frame_path || paths.preview || paths.keyframe || undefined, item.experiment_id)
}

function itemClip(item: MaterialSearchItem) {
  const paths = item.published_paths || {}
  return experimentFileUrl(item.clip_url || item.clip_file_path || paths.clip || undefined, item.experiment_id)
}

function formatRange(item: MaterialSearchItem) {
  const start = Number(item.time_start ?? item.timestamp_sec ?? item.local_timestamp_sec ?? 0)
  const end = Number(item.time_end ?? start)
  return `${Number.isFinite(start) ? start.toFixed(2) : '-'}-${Number.isFinite(end) ? end.toFixed(2) : '-'}s`
}

function materialKey(item: MaterialSearchItem, index: number) {
  return `workspace-material-${item.experiment_id || 'workspace'}-${item.material_id || item.item_id || item.event_id || index}`
}

export default function WorkspaceMaterials() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [items, setItems] = useState<MaterialSearchItem[]>([])
  const [health, setHealth] = useState<WorkspacePublishedHealthResponse | null>(null)
  const [searchText, setSearchText] = useState(searchParams.get('text') || '')
  const [submittedText, setSubmittedText] = useState(searchParams.get('text') || '')
  const [experimentFilter, setExperimentFilter] = useState(searchParams.get('experiment_id') || '')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  async function load(text: string) {
    setLoading(true)
    setError(null)
    try {
      const [published, nextHealth] = await Promise.all([
        workspaceMaterialApi.getPublishedMaterials({
          limit: 500,
          text: text || undefined,
          sort_by: text ? 'relevance' : 'time_start',
        }),
        workspaceMaterialApi.getPublishedHealth(true).catch(() => null),
      ])
      setItems(published.items || [])
      setHealth((published.index_lifecycle || nextHealth) ?? null)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '工作区正式素材加载失败')
      setItems([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load(submittedText)
  }, [submittedText])

  const filtered = useMemo(() => {
    const exp = experimentFilter.trim()
    if (!exp) return items
    return items.filter(item => String(item.experiment_id || '').includes(exp))
  }, [experimentFilter, items])

  const previewCount = filtered.filter(item => Boolean(itemPreview(item))).length
  const clipCount = filtered.filter(item => Boolean(itemClip(item))).length
  const formalImageCount = filtered.filter(item => Boolean(itemPreview(item))).length
  const formalVideoCount = filtered.filter(item => !itemPreview(item) && Boolean(itemClip(item))).length
  const warnings = health?.warnings || health?.warnings_before_rebuild || []

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const next = new URLSearchParams()
    if (searchText.trim()) next.set('text', searchText.trim())
    if (experimentFilter.trim()) next.set('experiment_id', experimentFilter.trim())
    setSearchParams(next, { replace: true })
    setSubmittedText(searchText.trim())
  }

  return (
    <div className="space-y-5">
      <PageHero
        eyebrow={<Link to="/experiments" className="hover:text-slate-900">工作区</Link>}
        title="全局正式素材检索"
        description="直接读取工作区发布索引，只展示审批后进入正式素材库的关键帧和关键片段。"
        actions={<Link to="/experiments" className={primaryButtonClass('slate')}>实验列表</Link>}
      />

      <section
        className="grid gap-4 md:grid-cols-2 xl:grid-cols-4"
        data-smoke="workspace-material-metrics"
        data-total={items.length}
        data-filtered={filtered.length}
        data-health-status={health?.status || ''}
      >
        <MetricTile label="索引素材" value={items.length} helper={`${filtered.length} current filter`} tone="blue" Icon={Boxes} />
        <MetricTile label="正式关键帧" value={formalImageCount} helper={`${previewCount} preview URLs`} tone="emerald" Icon={Image} />
        <MetricTile label="正式片段" value={formalVideoCount} helper={`${clipCount} clip URLs`} tone="cyan" Icon={Film} />
        <MetricTile label="索引状态" value={health?.status || '-'} helper={`${health?.sqlite_count ?? '-'} sqlite / ${health?.expected_indexable_count ?? '-'} expected`} tone={toneForStatus(health?.status)} Icon={DatabaseZap} />
      </section>

      <EvidenceCard className="p-4">
        <form onSubmit={submit} className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_18rem_auto]">
          <label className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <input
              value={searchText}
              onChange={event => setSearchText(event.target.value)}
              placeholder="烧杯 / 容器操作 / 戴手套操作 / pouring liquid"
              className="w-full rounded-lg border border-slate-200 py-2 pl-9 pr-3 text-sm font-semibold outline-none focus:border-blue-400"
              data-smoke="workspace-material-search"
            />
          </label>
          <input
            value={experimentFilter}
            onChange={event => setExperimentFilter(event.target.value)}
            placeholder="实验 ID 筛选"
            className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold outline-none focus:border-blue-400"
            data-smoke="workspace-experiment-filter"
          />
          <button type="submit" className={primaryButtonClass('blue')} data-smoke="workspace-material-search-submit">
            <Search className="h-4 w-4" />
            搜索
          </button>
        </form>
      </EvidenceCard>

      {warnings.length > 0 && (
        <EvidenceCard className="border-amber-200 bg-amber-50 p-4 text-sm font-semibold text-amber-800" data-smoke="workspace-index-warnings">
          <div className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <div className="font-black">发布索引生命周期提示</div>
              <div className="mt-1">{warnings.map(item => String(item.code || item.message || 'warning')).join(' / ')}</div>
            </div>
          </div>
        </EvidenceCard>
      )}

      {error && <EvidenceCard className="border-red-200 bg-red-50 p-4 text-red-700">{error}</EvidenceCard>}
      {loading ? <EmptyEvidence title="正在加载工作区正式素材..." /> : filtered.length === 0 ? <EmptyEvidence title="暂无匹配的正式素材" /> : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3" data-smoke="workspace-material-grid" data-filtered-count={filtered.length}>
          {filtered.map((item, index) => (
            <EvidenceCard key={materialKey(item, index)} className="overflow-hidden" data-smoke="workspace-material-card">
              <WorkspaceMaterialPreview item={item} />
              <div className="p-4">
                <div className="mb-2 flex flex-wrap gap-2">
                  <EvidenceBadge tone="blue">{formatRange(item)}</EvidenceBadge>
                  <EvidenceBadge tone={itemClip(item) ? 'emerald' : 'slate'}>{itemClip(item) ? 'clip' : 'frame'}</EvidenceBadge>
                  {item.review_status && <EvidenceBadge tone={toneForStatus(item.review_status)}>{item.review_status}</EvidenceBadge>}
                </div>
                <h3 className="line-clamp-2 font-black text-slate-950">{cleanDisplayText(item.display_name || item.event_type || item.item_id, '正式素材')}</h3>
                <div className="mt-1 break-all text-xs font-semibold text-slate-500">{item.experiment_id}</div>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {[...(item.object_labels || []), ...(item.actions || []), item.event_type].filter(Boolean).slice(0, 8).map((label, labelIndex) => (
                    <span key={`${materialKey(item, index)}-label-${labelIndex}`} className="rounded bg-slate-100 px-2 py-1 text-xs font-bold text-slate-600">{cleanDisplayText(label)}</span>
                  ))}
                </div>
                <div className="mt-3 flex flex-wrap gap-3 text-sm font-bold">
                  {itemClip(item) && <a href={itemClip(item)} target="_blank" rel="noreferrer" className="text-blue-700 hover:text-blue-900">打开片段</a>}
                  <Link to={`/experiments/${item.experiment_id}/materials`} className="text-slate-700 hover:text-slate-950">实验素材页</Link>
                </div>
              </div>
            </EvidenceCard>
          ))}
        </div>
      )}
    </div>
  )
}

function WorkspaceMaterialPreview({ item }: { item: MaterialSearchItem }) {
  const preview = itemPreview(item)
  const clip = itemClip(item)
  if (preview) {
    return <img src={preview} alt={cleanDisplayText(item.display_name || item.item_id, 'material')} className="aspect-video w-full bg-slate-100 object-cover" data-smoke="workspace-formal-image" />
  }
  if (clip) {
    return <video src={clip} className="aspect-video w-full bg-slate-950 object-contain" controls preload="metadata" data-smoke="workspace-formal-video" />
  }
  return <div className="flex aspect-video items-center justify-center bg-slate-100 text-sm font-semibold text-slate-400">no preview</div>
}
