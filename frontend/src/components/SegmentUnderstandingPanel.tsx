import { useEffect, useMemo, useRef, useState } from 'react'
import { cleanDisplayText } from '../displayText'

const TYPEWRITER_CHARS_PER_TICK = 2
const TYPEWRITER_TICK_MS = 16

function prefersReducedMotion() {
  if (typeof window === 'undefined' || !window.matchMedia) return false
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/**
 * Progressive (typewriter) reveal of one segment's Qwen understanding text.
 * The full text is already fetched; this only animates how it appears.
 * Honors prefers-reduced-motion and offers a skip-to-full button.
 */
export default function SegmentUnderstandingPanel({
  text,
  source,
}: {
  text: string
  source?: string
}) {
  const cleanText = useMemo(() => cleanDisplayText(text || '', '').trim(), [text])
  const reduceMotion = useMemo(prefersReducedMotion, [])
  const [revealed, setRevealed] = useState(reduceMotion ? cleanText.length : 0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    setRevealed(reduceMotion ? cleanText.length : 0)
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    if (reduceMotion || !cleanText) return
    timerRef.current = setInterval(() => {
      setRevealed(prev => {
        const next = prev + TYPEWRITER_CHARS_PER_TICK
        if (next >= cleanText.length) {
          if (timerRef.current) {
            clearInterval(timerRef.current)
            timerRef.current = null
          }
          return cleanText.length
        }
        return next
      })
    }, TYPEWRITER_TICK_MS)
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [cleanText, reduceMotion])

  if (!cleanText) return null

  const isStreaming = revealed < cleanText.length
  const visible = cleanText.slice(0, revealed)
  const isQwen = source === 'qwen'

  const skip = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    setRevealed(cleanText.length)
  }

  return (
    <section
      className="mt-3 rounded-[var(--ui-radius-md)] border border-[color:var(--ui-border)] bg-[color:var(--ui-accent-soft)] p-3"
      data-smoke="segment-understanding-panel"
    >
      <header className="mb-1.5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="flex h-1.5 w-1.5 rounded-full bg-[color:var(--ui-accent)]" aria-hidden />
          <span className="text-xs font-black text-[color:var(--ui-text)]">实验理解</span>
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
              isQwen
                ? 'bg-[color:var(--ui-accent)] text-white'
                : 'bg-[color:var(--ui-bg-muted)] text-[color:var(--ui-text-muted)]'
            }`}
          >
            {isQwen ? 'Qwen 视觉理解' : '启发式摘要'}
          </span>
        </div>
        {isStreaming && (
          <button
            type="button"
            onClick={skip}
            className="text-[11px] font-semibold text-[color:var(--ui-accent)] hover:opacity-70"
          >
            跳过
          </button>
        )}
      </header>
      <p className="whitespace-pre-wrap text-xs leading-relaxed text-[color:var(--ui-text-muted)]">
        {visible}
        {isStreaming && (
          <span className="ml-0.5 inline-block h-3 w-1.5 animate-pulse bg-[color:var(--ui-accent-weak)] align-middle" />
        )}
      </p>
    </section>
  )
}
