import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { experimentApi } from '../api'
import ExperimentPageShell from '../components/ExperimentSideNav'
import type { StructuredExperimentResult } from '../types'

const EMPTY_STATE_HINT = {
  description: '未分析时返回空结构，不会伪造步骤、时间线或证据。',
}

const MAX_ARRAY_CHILDREN = 100

export default function JsonViewer() {
  const { id } = useParams<{ id: string }>()
  const [data, setData] = useState<StructuredExperimentResult | null>(null)
  const [view, setView] = useState<'tree' | 'raw'>('tree')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (id) void loadData()
  }, [id])

  const loadData = async () => {
    try {
      const jsonData = await experimentApi.getJson(id!)
      setData(jsonData)
    } catch (error) {
      console.error('Failed to load JSON:', error)
    } finally {
      setLoading(false)
    }
  }

  const copyToClipboard = () => {
    if (!data) return
    navigator.clipboard.writeText(JSON.stringify(data, null, 2))
    alert('JSON 已复制到剪贴板')
  }

  const downloadJson = () => {
    if (!data) return
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `experiment_${id}_structured.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (loading) {
    const content = <div className="py-12 text-center">加载中...</div>
    return id ? <ExperimentPageShell experimentId={id}>{content}</ExperimentPageShell> : content
  }
  if (!data) {
    const content = <div className="py-12 text-center text-red-600">JSON 数据加载失败</div>
    return id ? <ExperimentPageShell experimentId={id}>{content}</ExperimentPageShell> : content
  }

  const content = (
    <div>
      <div className="mb-6">
        <div className="flex items-center space-x-2 text-sm text-gray-500 mb-2">
          <Link to="/experiments" className="hover:text-gray-700">实验列表</Link>
          <span>/</span>
          <Link to={`/experiments/${id}/workspace`} className="hover:text-gray-700">实验详情</Link>
          <span>/</span>
          <span className="text-gray-900">结构化 JSON</span>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">结构化 JSON 查看器</h2>
            {data.analysis == null && <p className="text-sm text-gray-600 mt-1">{EMPTY_STATE_HINT.description}</p>}
          </div>
          <div className="flex space-x-2">
            <button onClick={() => setView('tree')} className={`px-4 py-2 rounded-md text-sm font-medium ${view === 'tree' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'}`}>树形视图</button>
            <button onClick={() => setView('raw')} className={`px-4 py-2 rounded-md text-sm font-medium ${view === 'raw' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'}`}>原始 JSON</button>
            <button onClick={copyToClipboard} className="px-4 py-2 bg-white text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50 text-sm">复制</button>
            <button onClick={downloadJson} className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 text-sm">下载</button>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow">
        {view === 'tree' ? (
          <div className="p-6"><JsonTree data={data} /></div>
        ) : (
          <pre className="p-6 overflow-auto text-sm font-mono">{JSON.stringify(data, null, 2)}</pre>
        )}
      </div>
    </div>
  )

  return id ? <ExperimentPageShell experimentId={id}>{content}</ExperimentPageShell> : content
}

function JsonTree({ data, level = 0 }: { data: any; level?: number }) {
  const [collapsed, setCollapsed] = useState<{ [key: string]: boolean }>({})

  const toggleCollapse = (key: string) => {
    setCollapsed(prev => ({ ...prev, [key]: !prev[key] }))
  }

  const renderValue = (key: string, value: any, index: number) => {
    const indent = level * 20

    if (value === null) {
      return <div key={index} style={{ marginLeft: `${indent}px` }} className="py-1"><span className="text-blue-600 font-medium">{key}:</span> <span className="text-gray-500">null</span></div>
    }

    if (typeof value === 'object' && !Array.isArray(value)) {
      const collapseKey = `${level}-${key}`
      const isCollapsed = collapsed[collapseKey] ?? true
      return (
        <div key={index} style={{ marginLeft: `${indent}px` }}>
          <div className="py-1">
            <button onClick={() => toggleCollapse(collapseKey)} className="text-blue-600 font-medium hover:text-blue-700 mr-2">{isCollapsed ? '▸' : '▾'}</button>
            <span className="text-blue-600 font-medium">{key}:</span> <span className="text-gray-500">{`{${Object.keys(value).length} keys}`}</span>
          </div>
          {!isCollapsed && <JsonTree data={value} level={level + 1} />}
        </div>
      )
    }

    if (Array.isArray(value)) {
      const collapseKey = `${level}-${key}`
      const isCollapsed = collapsed[collapseKey] ?? true
      const visibleItems = value.slice(0, MAX_ARRAY_CHILDREN)
      return (
        <div key={index} style={{ marginLeft: `${indent}px` }}>
          <div className="py-1">
            <button onClick={() => toggleCollapse(collapseKey)} className="text-blue-600 font-medium hover:text-blue-700 mr-2">{isCollapsed ? '▸' : '▾'}</button>
            <span className="text-blue-600 font-medium">{key}:</span> <span className="text-gray-500">{`[${value.length} items]`}</span>
          </div>
          {!isCollapsed && (
            <div>
              {visibleItems.map((item, idx) => <JsonTree key={idx} data={{ [`[${idx}]`]: item }} level={level + 1} />)}
              {value.length > visibleItems.length && (
                <div style={{ marginLeft: `${(level + 1) * 20}px` }} className="py-1 text-sm text-gray-500">
                  仅显示前 {MAX_ARRAY_CHILDREN} 项；完整内容请切换到“原始 JSON”。
                </div>
              )}
            </div>
          )}
        </div>
      )
    }

    return <div key={index} style={{ marginLeft: `${indent}px` }} className="py-1"><span className="text-blue-600 font-medium">{key}:</span> <span className={typeof value === 'string' ? 'text-green-600' : 'text-purple-600'}>{typeof value === 'string' ? `"${value}"` : String(value)}</span></div>
  }

  return <div>{Object.entries(data).map(([key, value], index) => renderValue(key, value, index))}</div>
}
