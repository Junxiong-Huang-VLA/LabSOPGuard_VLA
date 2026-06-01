import { Link, Outlet, useLocation } from 'react-router-dom'
import { FlaskConical, Layers, Newspaper, UploadCloud, type LucideIcon } from 'lucide-react'
import { prefetchExperimentsList } from '../api'
import FrontendRecorder from './FrontendRecorder'

type NavItem = {
  to: string
  label: string
  hint: string
  Icon: LucideIcon
  isActive: (pathname: string) => boolean
  onHover?: () => void
}

const NAV_ITEMS: NavItem[] = [
  {
    to: '/experiments',
    label: '实验分析',
    hint: '片段 · 双视角 · 理解',
    Icon: FlaskConical,
    isActive: (p) => p === '/experiments' || p === '/windows' || p.startsWith('/experiments/'),
    onHover: prefetchExperimentsList,
  },
  {
    to: '/upload',
    label: '新建实验',
    hint: '上传双视角视频',
    Icon: UploadCloud,
    isActive: (p) => p.startsWith('/upload'),
  },
  {
    to: '/memory',
    label: '实验室日报',
    hint: '记忆与态势汇总',
    Icon: Newspaper,
    isActive: (p) => p.startsWith('/memory') || p.startsWith('/video-memory'),
  },
]

function NavLink({ item, active }: { item: NavItem; active: boolean }) {
  const { Icon } = item
  return (
    <Link
      to={item.to}
      onMouseEnter={item.onHover}
      onFocus={item.onHover}
      aria-current={active ? 'page' : undefined}
      className={`group flex items-center gap-3 rounded-[var(--ui-radius-md)] px-3 py-2.5 transition-colors ${
        active
          ? 'bg-[color:var(--ui-accent)] text-white shadow-[var(--ui-shadow-subtle)]'
          : 'text-[color:var(--ui-text-muted)] hover:bg-[color:var(--ui-bg-muted)] hover:text-[color:var(--ui-text)]'
      }`}
    >
      <Icon className={`h-5 w-5 shrink-0 ${active ? 'text-white' : 'text-[color:var(--ui-text-muted)] group-hover:text-[color:var(--ui-accent)]'}`} />
      <span className="min-w-0">
        <span className="block truncate text-sm font-bold">{item.label}</span>
        <span className={`block truncate text-[11px] ${active ? 'text-white/80' : 'text-[color:var(--ui-text-muted)]'}`}>{item.hint}</span>
      </span>
    </Link>
  )
}

export default function Layout() {
  const location = useLocation()
  const pathname = location.pathname

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto max-w-[1760px] px-5 sm:px-8">
          <div className="flex min-h-16 items-center gap-3 py-3">
            <Link to="/experiments" className="flex min-w-0 items-center gap-2 sm:gap-3">
              <img src="/realityloop-logo.png" alt="RealityLoop logo" className="h-8 w-8 shrink-0 object-contain sm:h-10 sm:w-10" />
              <h1 className="truncate text-base font-bold tracking-tight text-slate-950 sm:text-xl">
                RealityLoop 实验室态势感知理解平台
              </h1>
            </Link>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-[1760px] flex-col gap-6 px-5 py-6 sm:px-8 lg:flex-row">
        <aside className="lg:w-60 lg:shrink-0">
          <nav
            className="flex gap-2 overflow-x-auto rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/85 p-2 shadow-[var(--ui-shadow-subtle)] lg:sticky lg:top-24 lg:flex-col lg:overflow-visible"
            aria-label="主导航"
          >
            <p className="hidden px-2 pb-1 pt-2 text-[11px] font-bold uppercase tracking-wider text-[color:var(--ui-text-muted)] lg:block">
              导航
            </p>
            {NAV_ITEMS.map((item) => (
              <div key={item.to} className="shrink-0 lg:shrink">
                <NavLink item={item} active={item.isActive(pathname)} />
              </div>
            ))}
            <div className="mt-1 hidden rounded-[var(--ui-radius-md)] bg-[color:var(--ui-accent-soft)] p-3 lg:block">
              <div className="flex items-center gap-2 text-[color:var(--ui-accent)]">
                <Layers className="h-4 w-4" />
                <span className="text-xs font-bold">双视角理解</span>
              </div>
              <p className="mt-1 text-[11px] leading-relaxed text-[color:var(--ui-text-muted)]">
                第一/第三人称交叉验证，自动产出关键素材与实验过程理解。
              </p>
            </div>
          </nav>
        </aside>

        <main className="min-w-0 flex-1">
          <Outlet />
        </main>
      </div>
      <FrontendRecorder />
    </div>
  )
}
