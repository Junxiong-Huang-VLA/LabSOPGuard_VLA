import type { ErrorInfo, ReactNode } from 'react'
import { Component } from 'react'
import { AlertTriangle } from 'lucide-react'
import { EvidenceCard } from './EvidenceUI'

export default class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Frontend error boundary caught:', error, info)
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="mx-auto max-w-3xl p-6">
        <EvidenceCard className="border-red-200 bg-red-50 p-6 text-red-700">
          <div className="flex items-center gap-2 text-lg font-black">
            <AlertTriangle className="h-5 w-5" />
            页面渲染失败
          </div>
          <p className="mt-2 text-sm font-semibold">前端捕获到异常，当前页面没有继续渲染。</p>
          <pre className="mt-4 overflow-auto rounded-lg bg-white p-3 text-xs text-red-800">{this.state.error.message}</pre>
        </EvidenceCard>
      </div>
    )
  }
}
