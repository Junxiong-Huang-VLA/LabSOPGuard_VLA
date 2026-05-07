const BAD_TEXT_RE = /(?:锟|閿|脙|脗|忙|氓|猫|茅|\?{3,}|[\u0080-\u009f])/
const MOJIBAKE_RE = /[ÂÃÅÆæçèéêëìíîïðñòóôõöùúûüýāăąåã]/

function cjkCount(value: string) {
  return (value.match(/[\u3400-\u9fff]/g) || []).length
}

function repairUtf8Mojibake(value: string) {
  if (!MOJIBAKE_RE.test(value) && !/[\u0080-\u009f]/.test(value)) return value
  try {
    const bytes = new Uint8Array(Array.from(value, ch => ch.charCodeAt(0) & 0xff))
    const decoded = new TextDecoder('utf-8', { fatal: false }).decode(bytes)
    if (cjkCount(decoded) > cjkCount(value)) return decoded
  } catch {
    // Keep the original text if the browser cannot decode it.
  }
  return value
}

export function cleanDisplayText(value: unknown, fallback = '-'): string {
  const raw = String(value ?? '').replace(/\s+/g, ' ').trim()
  if (!raw) return fallback
  const text = repairUtf8Mojibake(raw)
  const compact = text.replace(/\s+/g, '')
  const questionRatio = compact.length > 0 ? (compact.match(/\?/g)?.length ?? 0) / compact.length : 0
  if (BAD_TEXT_RE.test(text) || questionRatio > 0.45) return fallback
  return text
}
