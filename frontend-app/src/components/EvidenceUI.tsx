import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'
import { AlertCircle, CheckCircle2, Clock3, Loader2 } from 'lucide-react'

export type Tone = 'blue' | 'emerald' | 'amber' | 'red' | 'slate' | 'violet' | 'cyan'

const toneClasses: Record<Tone, { soft: string; solid: string; border: string; text: string }> = {
  blue: { soft: 'bg-blue-50 text-blue-700 ring-blue-200', solid: 'bg-blue-600 text-white', border: 'border-blue-200', text: 'text-blue-700' },
  emerald: { soft: 'bg-emerald-50 text-emerald-700 ring-emerald-200', solid: 'bg-emerald-600 text-white', border: 'border-emerald-200', text: 'text-emerald-700' },
  amber: { soft: 'bg-amber-50 text-amber-700 ring-amber-200', solid: 'bg-amber-500 text-white', border: 'border-amber-200', text: 'text-amber-700' },
  red: { soft: 'bg-red-50 text-red-700 ring-red-200', solid: 'bg-red-600 text-white', border: 'border-red-200', text: 'text-red-700' },
  slate: { soft: 'bg-slate-100 text-slate-700 ring-slate-200', solid: 'bg-slate-800 text-white', border: 'border-slate-200', text: 'text-slate-700' },
  violet: { soft: 'bg-violet-50 text-violet-700 ring-violet-200', solid: 'bg-violet-600 text-white', border: 'border-violet-200', text: 'text-violet-700' },
  cyan: { soft: 'bg-cyan-50 text-cyan-700 ring-cyan-200', solid: 'bg-cyan-600 text-white', border: 'border-cyan-200', text: 'text-cyan-700' },
}

export function toneForStatus(status?: string | null): Tone {
  const value = String(status || '').toLowerCase()
  if (['completed', 'done', 'analyzed', 'pass', 'ready'].some(item => value.includes(item))) return 'emerald'
  if (['running', 'queued', 'generating', 'writing', 'uploading'].some(item => value.includes(item))) return 'blue'
  if (['failed', 'error', 'blocked'].some(item => value.includes(item))) return 'red'
  if (['review', 'candidate', 'partial', 'waiting'].some(item => value.includes(item))) return 'amber'
  return 'slate'
}

export function EvidenceBadge({ children, tone = 'slate', className = '' }: { children: ReactNode; tone?: Tone; className?: string }) {
  return <span className={`inline-flex min-h-6 items-center rounded-md px-2 py-0.5 text-xs font-bold ring-1 ${toneClasses[tone].soft} ${className}`}>{children}</span>
}

export function EvidenceCard({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`rounded-lg border border-slate-200 bg-white shadow-sm shadow-slate-200/70 ${className}`}>{children}</section>
}

export function MetricTile({ label, value, helper, tone = 'slate', Icon }: { label: string; value: ReactNode; helper?: ReactNode; tone?: Tone; Icon?: LucideIcon }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm shadow-slate-200/70">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs font-bold uppercase tracking-wide text-slate-500">{label}</div>
          <div className="mt-2 break-words text-2xl font-black leading-none tracking-tight text-slate-950">{value}</div>
        </div>
        {Icon && <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ring-1 ${toneClasses[tone].soft}`}><Icon className="h-4 w-4" /></span>}
      </div>
      {helper && <div className="mt-2 text-xs font-medium text-slate-500">{helper}</div>}
    </div>
  )
}

export function PageHero({ eyebrow, title, description, actions, tabs }: { eyebrow?: ReactNode; title: ReactNode; description?: ReactNode; actions?: ReactNode; tabs?: ReactNode }) {
  return (
    <EvidenceCard className="overflow-hidden">
      <div className="p-5 sm:p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            {eyebrow && <div className="mb-2 text-sm font-semibold text-slate-500">{eyebrow}</div>}
            <h2 className="text-2xl font-black tracking-tight text-slate-950 sm:text-3xl">{title}</h2>
            {description && <p className="mt-2 max-w-3xl text-sm font-medium leading-6 text-slate-500">{description}</p>}
          </div>
          {actions && <div className="flex shrink-0 flex-wrap gap-2">{actions}</div>}
        </div>
      </div>
      {tabs && <div className="border-t border-slate-100 px-5 py-3 sm:px-6">{tabs}</div>}
    </EvidenceCard>
  )
}

export function ProgressStrip({ status, progress, message }: { status: string; progress: number; message?: ReactNode }) {
  const tone = toneForStatus(status)
  const pct = progress > 1 ? Math.round(progress) : Math.round(Math.max(0, Math.min(1, progress)) * 100)
  const Icon = tone === 'emerald' ? CheckCircle2 : tone === 'red' ? AlertCircle : tone === 'blue' ? Loader2 : Clock3
  return (
    <EvidenceCard className="px-5 py-4">
      <div className="mb-2 flex items-center justify-between gap-3 text-sm">
        <div className="flex min-w-0 items-center gap-2">
          <Icon className={`h-4 w-4 shrink-0 ${toneClasses[tone].text} ${tone === 'blue' ? 'animate-spin' : ''}`} />
          <span className="truncate font-bold text-slate-700">{message || status}</span>
        </div>
        <span className={`font-mono text-xs font-black ${toneClasses[tone].text}`}>{pct}%</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
        <div className={`h-full rounded-full transition-all duration-700 ${toneClasses[tone].solid}`} style={{ width: `${Math.max(pct, tone === 'blue' ? 2 : 0)}%` }} />
      </div>
    </EvidenceCard>
  )
}

export function EmptyEvidence({ title, description, action }: { title: ReactNode; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-6 text-sm text-slate-600">
      <div className="font-black text-slate-900">{title}</div>
      {description && <div className="mt-1 leading-6">{description}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function primaryButtonClass(tone: Tone = 'blue') {
  return `inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-bold shadow-lg transition hover:brightness-95 ${toneClasses[tone].solid}`
}

export function secondaryButtonClass(tone: Tone = 'slate') {
  return `inline-flex items-center justify-center gap-2 rounded-lg border bg-white px-4 py-2 text-sm font-bold transition hover:bg-slate-50 ${toneClasses[tone].border} ${toneClasses[tone].text}`
}
