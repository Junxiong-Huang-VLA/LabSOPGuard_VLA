export function backendOrigin(): string {
  const configured = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_ORIGIN
  if (configured) {
    return String(configured).replace(/\/api\/v1\/?$/, '').replace(/\/$/, '')
  }
  if (window.location.port === '5173') {
    return window.location.origin
  }
  return window.location.origin
}

export function mediaUrl(url: string | null | undefined): string | undefined {
  if (!url) return undefined
  if (/^https?:\/\//i.test(url)) return url
  if (url.startsWith('/api/')) return `${backendOrigin()}${url}`
  return url
}

export function addCacheBust(url: string | undefined, token: string | undefined): string | undefined {
  if (!url) return undefined
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}v=${encodeURIComponent(token || 'latest')}`
}

export function experimentFileUrl(value: string | null | undefined, experimentId?: string | null, version?: string): string | undefined {
  if (!value) return undefined
  const normalized = String(value).trim().replace(/\\/g, '/')
  if (/^https?:\/\//i.test(normalized) || normalized.startsWith('/api/')) return addCacheBust(mediaUrl(normalized), version)
  if (normalized.includes('/outputs/experiments/')) {
    const marker = '/outputs/experiments/'
    const after = normalized.substring(normalized.indexOf(marker) + marker.length)
    const slash = after.indexOf('/')
    if (slash > 0) {
      const id = after.substring(0, slash)
      const rel = after.substring(slash + 1)
      return addCacheBust(mediaUrl(`/api/v1/experiments/${encodeURIComponent(id)}/files/${encodePath(rel)}`), version)
    }
  }
  if (normalized.startsWith('outputs/experiments/')) {
    const after = normalized.substring('outputs/experiments/'.length)
    const slash = after.indexOf('/')
    if (slash > 0) {
      const id = after.substring(0, slash)
      const rel = after.substring(slash + 1)
      return addCacheBust(mediaUrl(`/api/v1/experiments/${encodeURIComponent(id)}/files/${encodePath(rel)}`), version)
    }
  }
  if (experimentId) {
    return addCacheBust(mediaUrl(`/api/v1/experiments/${encodeURIComponent(experimentId)}/files/${encodePath(normalized)}`), version)
  }
  return addCacheBust(mediaUrl(normalized), version)
}

function encodePath(path: string): string {
  return path.split('/').map(part => encodeURIComponent(part)).join('/')
}
