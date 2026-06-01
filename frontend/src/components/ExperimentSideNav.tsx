import type { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Boxes, FileText, Layers3, type LucideIcon } from 'lucide-react'

type ExperimentNavItem = {
  to: string
  label: string
  Icon: LucideIcon
  isActive: (pathname: string) => boolean
}

function navItems(experimentId: string): ExperimentNavItem[] {
  const root = `/experiments/${experimentId}`
  return [
    {
      to: `${root}/workspace`,
      label: '实验片段',
      Icon: Layers3,
      isActive: pathname => (
        pathname === `${root}/workspace`
        || pathname === `${root}/json`
        || pathname === `${root}/video-analysis`
        || pathname === `${root}/timeline`
        || pathname.startsWith(`${root}/steps/`)
      ),
    },
    {
      to: `${root}/materials`,
      label: '关键素材',
      Icon: Boxes,
      isActive: pathname => (
        pathname === `${root}/materials`
        || pathname === `${root}/materials/review`
        || pathname === `${root}/materials/timeline`
      ),
    },
    {
      to: `${root}/report`,
      label: '实验日报',
      Icon: FileText,
      isActive: pathname => pathname === `${root}/report`,
    },
  ]
}

function ExperimentNavLink({ item, active }: { item: ExperimentNavItem; active: boolean }) {
  const { Icon } = item
  return (
    <Link
      to={item.to}
      aria-current={active ? 'page' : undefined}
      className={`group inline-flex h-8 items-center gap-1.5 whitespace-nowrap rounded-full px-3 text-sm font-bold transition-colors ${
        active
          ? 'bg-[color:var(--ui-accent)] text-white shadow-sm'
          : 'text-[color:var(--ui-text-muted)] hover:bg-slate-100 hover:text-[color:var(--ui-text)]'
      }`}
    >
      <Icon className={`h-4 w-4 shrink-0 ${active ? 'text-white' : 'text-[color:var(--ui-text-muted)] group-hover:text-[color:var(--ui-accent)]'}`} />
      <span>{item.label}</span>
    </Link>
  )
}

export function ExperimentSideNav({ experimentId }: { experimentId: string }) {
  const location = useLocation()
  const items = navItems(experimentId)

  return (
    <div className="pointer-events-none fixed left-1/2 top-[4.75rem] z-50 flex -translate-x-1/2 justify-center lg:left-[calc(50%+8.25rem)]">
      <nav
        className="pointer-events-auto inline-flex max-w-[calc(100vw-2rem)] items-center gap-1 overflow-x-auto rounded-full border border-[color:var(--ui-border)] bg-white/90 px-2 py-1.5 shadow-[0_16px_40px_rgba(15,23,42,0.14)] backdrop-blur"
        aria-label="实验内导航"
      >
        {items.map(item => (
          <ExperimentNavLink key={item.to} item={item} active={item.isActive(location.pathname)} />
        ))}
      </nav>
    </div>
  )
}

export default function ExperimentPageShell({ experimentId, children }: { experimentId: string; children: ReactNode }) {
  return (
    <div className="relative">
      <ExperimentSideNav experimentId={experimentId} />
      <div className="min-w-0">{children}</div>
    </div>
  )
}
