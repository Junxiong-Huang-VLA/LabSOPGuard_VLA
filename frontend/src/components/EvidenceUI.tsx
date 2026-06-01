import type { HTMLAttributes, ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'
import { AlertCircle, CheckCircle2, Clock3, Loader2 } from 'lucide-react'

export type Tone =
  | 'primary'
  | 'slate'
  | 'success'
  | 'warning'
  | 'danger'
  | 'blue'
  | 'cyan'
  | 'emerald'
  | 'amber'
  | 'red'
  | 'violet'

const toneClasses: Record<Tone, { soft: string; solid: string; border: string; text: string }> = {
  primary: {
    soft: 'bg-[color:var(--ui-accent-soft)] text-[color:var(--ui-accent)] ring-[color:var(--ui-accent-weak)]',
    solid: 'bg-[color:var(--ui-accent)] text-white',
    border: 'border-[color:var(--ui-accent-soft)]',
    text: 'text-[color:var(--ui-accent)]',
  },
  slate: {
    soft: 'bg-[color:var(--ui-bg-muted)] text-[color:var(--ui-text-muted)] ring-[color:var(--ui-border)]',
    solid: 'bg-[color:var(--ui-border-dark)] text-white',
    border: 'border-[color:var(--ui-border)]',
    text: 'text-[color:var(--ui-text-muted)]',
  },
  success: {
    soft: 'bg-[color:var(--ui-success-soft)] text-[color:var(--ui-success)] ring-[color:var(--ui-success-weak)]',
    solid: 'bg-[color:var(--ui-success)] text-white',
    border: 'border-[color:var(--ui-success-soft)]',
    text: 'text-[color:var(--ui-success)]',
  },
  warning: {
    soft: 'bg-[color:var(--ui-warning-soft)] text-[color:var(--ui-warning)] ring-[color:var(--ui-warning-weak)]',
    solid: 'bg-[color:var(--ui-warning)] text-white',
    border: 'border-[color:var(--ui-warning-soft)]',
    text: 'text-[color:var(--ui-warning)]',
  },
  danger: {
    soft: 'bg-[color:var(--ui-danger-soft)] text-[color:var(--ui-danger)] ring-[color:var(--ui-danger-weak)]',
    solid: 'bg-[color:var(--ui-danger)] text-white',
    border: 'border-[color:var(--ui-danger-soft)]',
    text: 'text-[color:var(--ui-danger)]',
  },
  blue: {
    soft: 'bg-[color:var(--ui-accent-soft)] text-[color:var(--ui-accent)] ring-[color:var(--ui-accent-weak)]',
    solid: 'bg-[color:var(--ui-accent)] text-white',
    border: 'border-[color:var(--ui-accent-soft)]',
    text: 'text-[color:var(--ui-accent)]',
  },
  cyan: {
    soft: 'bg-[color:var(--ui-accent-soft)] text-[color:var(--ui-accent)] ring-[color:var(--ui-accent-weak)]',
    solid: 'bg-[color:var(--ui-accent)] text-white',
    border: 'border-[color:var(--ui-accent-soft)]',
    text: 'text-[color:var(--ui-accent)]',
  },
  emerald: {
    soft: 'bg-[color:var(--ui-success-soft)] text-[color:var(--ui-success)] ring-[color:var(--ui-success-weak)]',
    solid: 'bg-[color:var(--ui-success)] text-white',
    border: 'border-[color:var(--ui-success-soft)]',
    text: 'text-[color:var(--ui-success)]',
  },
  amber: {
    soft: 'bg-[color:var(--ui-warning-soft)] text-[color:var(--ui-warning)] ring-[color:var(--ui-warning-weak)]',
    solid: 'bg-[color:var(--ui-warning)] text-white',
    border: 'border-[color:var(--ui-warning-soft)]',
    text: 'text-[color:var(--ui-warning)]',
  },
  red: {
    soft: 'bg-[color:var(--ui-danger-soft)] text-[color:var(--ui-danger)] ring-[color:var(--ui-danger-weak)]',
    solid: 'bg-[color:var(--ui-danger)] text-white',
    border: 'border-[color:var(--ui-danger-soft)]',
    text: 'text-[color:var(--ui-danger)]',
  },
  violet: {
    soft: 'bg-[color:var(--ui-accent-soft)] text-[color:var(--ui-accent)] ring-[color:var(--ui-accent-weak)]',
    solid: 'bg-[color:var(--ui-accent)] text-white',
    border: 'border-[color:var(--ui-accent-soft)]',
    text: 'text-[color:var(--ui-accent)]',
  },
}

export function toneForStatus(status?: string | null): Tone {
  const value = String(status || '').toLowerCase()
  if (['completed', 'done', 'analyzed', 'pass', 'ready'].some(item => value.includes(item))) return 'success'
  if (['running', 'queued', 'generating', 'writing', 'uploading'].some(item => value.includes(item))) return 'primary'
  if (['failed', 'error', 'blocked'].some(item => value.includes(item))) return 'danger'
  if (['review', 'candidate', 'partial', 'waiting'].some(item => value.includes(item))) return 'warning'
  return 'slate'
}

export function EvidenceBadge({ children, tone = 'slate', className = '' }: { children: ReactNode; tone?: Tone; className?: string }) {
  return <span className={`inline-flex min-h-6 items-center rounded-md px-2 py-0.5 text-[11px] font-medium tracking-wide ring-1 ${toneClasses[tone].soft} ${className}`}>{children}</span>
}

export function EvidenceCard({ children, className = '', ...props }: { children: ReactNode; className?: string } & HTMLAttributes<HTMLElement>) {
  return <section className={`${evidenceCardClassName()} ${className}`} {...props}>{children}</section>
}

export function MetricTile({ label, value, helper, tone = 'slate', Icon }: { label: string; value: ReactNode; helper?: ReactNode; tone?: Tone; Icon?: LucideIcon }) {
  return (
    <div className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/85 p-4 shadow-[var(--ui-shadow-subtle)]">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-xs font-semibold uppercase tracking-wide text-[color:var(--ui-text-muted)]">{label}</div>
          <div className="mt-2 break-words text-2xl font-semibold leading-none tracking-tight text-[color:var(--ui-text)]">{value}</div>
        </div>
        {Icon && <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ring-1 ${toneClasses[tone].soft}`}><Icon className="h-4 w-4" /></span>}
      </div>
      {helper && <div className="mt-2 text-xs font-medium text-[color:var(--ui-text-muted)]">{helper}</div>}
    </div>
  )
}

export function PageHero({ eyebrow, title, description, actions, tabs }: { eyebrow?: ReactNode; title: ReactNode; description?: ReactNode; actions?: ReactNode; tabs?: ReactNode }) {
  return (
    <EvidenceCard className="overflow-hidden">
      <div className="rounded-[var(--ui-radius-xl)] bg-white/85 p-5 shadow-[var(--ui-shadow-card)] sm:p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            {eyebrow && <div className="mb-2 text-sm font-semibold text-[color:var(--ui-text-muted)]">{eyebrow}</div>}
            <h2 className="text-2xl font-semibold tracking-tight text-[color:var(--ui-text)] sm:text-3xl">{title}</h2>
            {description && <p className="mt-2 max-w-3xl text-sm font-medium leading-6 text-[color:var(--ui-text-muted)]">{description}</p>}
          </div>
          {actions && <div className="flex shrink-0 flex-wrap gap-2">{actions}</div>}
        </div>
        {tabs && <div className="mt-3 border-t border-[color:var(--ui-border)] pt-3">{tabs}</div>}
      </div>
    </EvidenceCard>
  )
}

export function ProgressStrip({
  status,
  progress,
  message,
  variant = 'default',
}: {
  status: string
  progress: number
  message?: ReactNode
  variant?: 'default' | 'health'
}) {
  const tone = toneForStatus(status)
  const pct = progress > 1 ? Math.round(progress) : Math.round(Math.max(0, Math.min(1, progress)) * 100)
  const Icon = tone === 'success' ? CheckCircle2 : tone === 'danger' ? AlertCircle : tone === 'warning' ? Clock3 : tone === 'primary' ? Loader2 : Clock3
  const barPercent = tone === 'danger' ? Math.min(99, Math.max(2, pct)) : Math.max(2, pct)

  if (variant === 'health') {
    const isFailed = tone === 'danger'
    const isPassed = tone === 'success'
    const healthPercent = isPassed ? 100 : isFailed ? 100 : Math.max(4, Math.min(100, pct))
    const healthLabel = isPassed ? '分析成功' : isFailed ? '分析失败' : '分析中'
    const healthFillClass = isFailed ? 'bg-red-500' : 'bg-emerald-500'
    const healthTextClass = isFailed ? 'text-red-700' : 'text-emerald-700'
    const healthSoftClass = isFailed ? 'border-red-200 bg-red-50' : 'border-emerald-200 bg-emerald-50'

    return (
      <EvidenceCard className={`px-5 py-4 ${healthSoftClass}`} data-smoke="analysis-health-progress">
        <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex min-w-0 items-center gap-2">
            <Icon className={`h-4 w-4 shrink-0 ${healthTextClass} ${tone === 'primary' ? 'animate-spin' : ''}`} />
            <span className={`text-sm font-semibold ${healthTextClass}`}>{healthLabel}</span>
            {message ? <span className="min-w-0 break-words text-sm font-medium text-[color:var(--ui-text-muted)]">{message}</span> : null}
          </div>
          <span className={`w-fit rounded-md bg-white/80 px-2 py-0.5 font-mono text-xs font-semibold ${healthTextClass}`}>{isPassed ? 100 : pct}%</span>
        </div>
        <div className="relative h-5 overflow-hidden rounded-md border border-white/70 bg-red-500 shadow-inner">
          <div className={`h-full rounded-[5px] transition-all duration-700 ${healthFillClass}`} style={{ width: `${healthPercent}%` }} />
          <div className="pointer-events-none absolute inset-0 grid grid-cols-10 divide-x divide-white/30" aria-hidden="true">
            {Array.from({ length: 10 }).map((_, index) => <span key={index} />)}
          </div>
        </div>
      </EvidenceCard>
    )
  }

  return (
    <EvidenceCard className="px-5 py-4">
      <div className="mb-2 flex items-center justify-between gap-3 text-sm">
        <div className="flex min-w-0 items-center gap-2">
          <Icon className={`h-4 w-4 shrink-0 ${toneClasses[tone].text} ${tone === 'primary' ? 'animate-spin' : ''}`} />
          <span className="truncate text-sm font-medium text-[color:var(--ui-text-muted)]">{message || status}</span>
        </div>
        <span className={`font-mono text-xs font-semibold ${toneClasses[tone].text}`}>{pct}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[color:var(--ui-bg-muted)]">
        <div className={`h-full rounded-full transition-all duration-700 ${toneClasses[tone].solid}`} style={{ width: `${barPercent}%` }} />
      </div>
    </EvidenceCard>
  )
}

export function EmptyEvidence({ title, description, action }: { title: ReactNode; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-[color:var(--ui-bg-muted)] p-6 text-sm text-[color:var(--ui-text-muted)]">
      <div className="text-sm font-semibold text-[color:var(--ui-text)]">{title}</div>
      {description && <div className="mt-1 leading-6">{description}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function primaryButtonClass(tone: Tone = 'primary') {
  return `inline-flex min-h-10 items-center justify-center gap-2 rounded-[var(--ui-radius-md)] px-4 py-2 text-sm font-medium transition hover:brightness-95 ${toneClasses[tone].solid}`
}

export function secondaryButtonClass(tone: Tone = 'slate') {
  return `inline-flex min-h-10 items-center justify-center gap-2 rounded-[var(--ui-radius-md)] border bg-white px-4 py-2 text-sm font-medium transition hover:bg-[color:var(--ui-bg-muted)] ${toneClasses[tone].border} ${toneClasses[tone].text}`
}

export function evidenceCardClassName() {
  return 'rounded-[var(--ui-radius-lg)] border border-[color:var(--ui-border)] bg-white/85 shadow-[var(--ui-shadow-card)]'
}

function resolveToneClassFromString(status: string | null | undefined): Tone {
  const value = String(status || '').toLowerCase()
  if (['success', 'ready', 'completed', 'confirmed', 'official', 'accepted', 'approved'].some(token => value.includes(token))) return 'success'
  if (['warning', 'review', 'pending', 'deferred'].some(token => value.includes(token))) return 'warning'
  if (['error', 'failed', 'rejected', 'blocking', 'invalid', 'blocked'].some(token => value.includes(token))) return 'danger'
  return 'primary'
}

export { resolveToneClassFromString }
