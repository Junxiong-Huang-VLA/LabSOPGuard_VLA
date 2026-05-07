import { Outlet, Link, useLocation } from 'react-router-dom'
import { prefetchExperimentsList } from '../api'

export default function Layout() {
  const location = useLocation()
  const isExperimentArea = location.pathname.startsWith('/experiments') || location.pathname.startsWith('/upload')
  const linkClass = (active: boolean) => `relative px-2 py-2 text-sm font-semibold transition-colors sm:px-3 sm:py-5 ${
    active
      ? 'text-blue-600 after:absolute after:inset-x-0 after:bottom-0 after:h-0.5 after:rounded-full after:bg-blue-600'
      : 'text-slate-500 hover:text-slate-800'
  }`

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/95 shadow-sm backdrop-blur">
        <div className="mx-auto max-w-[1760px] px-5 sm:px-8">
          <div className="flex min-h-16 flex-col items-stretch justify-center gap-3 py-3 sm:flex-row sm:items-center sm:justify-between sm:gap-8 sm:py-0">
            <Link to="/experiments" onMouseEnter={prefetchExperimentsList} onFocus={prefetchExperimentsList} className="flex min-w-0 items-center gap-2 sm:gap-3">
              <img src="/realityloop-logo.png" alt="RealityLoop logo" className="h-8 w-8 shrink-0 object-contain sm:h-10 sm:w-10" />
              <h1 className="truncate text-base font-bold tracking-tight text-slate-950 sm:text-xl">
                RealityLoop 实验室SOP态势感知理解平台
              </h1>
            </Link>

            <nav className="flex w-full shrink-0 flex-wrap justify-start gap-x-4 gap-y-1 sm:w-auto sm:justify-end sm:gap-x-5" aria-label="Primary">
              <Link to="/experiments" onMouseEnter={prefetchExperimentsList} onFocus={prefetchExperimentsList} className={linkClass(isExperimentArea)}>
                实验列表 Experiments
              </Link>
              <Link to="/ptz-tracker" className={linkClass(location.pathname.startsWith('/ptz-tracker'))}>
                云台跟随 PTZ
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1760px] px-5 py-8 sm:px-8">
        <Outlet />
      </main>
    </div>
  )
}
