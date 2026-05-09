import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import ExperimentList from '../ExperimentList'
import { experimentApi } from '../../api'
import type { Experiment } from '../../types'

vi.mock('../../api', () => ({
  experimentApi: {
    list: vi.fn(),
    delete: vi.fn(),
    archive: vi.fn(),
    unarchive: vi.fn(),
  },
  prefetchExperimentRoute: vi.fn(),
}))

const mockedApi = experimentApi as unknown as {
  list: ReturnType<typeof vi.fn>
  delete: ReturnType<typeof vi.fn>
  archive: ReturnType<typeof vi.fn>
  unarchive: ReturnType<typeof vi.fn>
}

function experiment(overrides: Partial<Experiment> = {}): Experiment {
  const id = overrides.experiment_id || 'exp-1'
  return {
    experiment_id: id,
    title: `Experiment ${id}`,
    description: '',
    status: 'completed',
    created_at: '2026-05-05T00:00:00Z',
    completed_at: '2026-05-05T00:05:00Z',
    analyzed_at: '2026-05-05T00:05:00Z',
    total_steps: 1,
    inferred_steps: 0,
    avg_confidence: 1,
    evidence_count: 0,
    processing_stage: 'output_generation',
    models_used: [],
    ...overrides,
  }
}

function renderList(items: Experiment[]) {
  mockedApi.list.mockResolvedValue({ total: items.length, experiments: items })
  mockedApi.archive.mockImplementation((id: string) => {
    const found = items.find(item => item.experiment_id === id)
    return Promise.resolve({ ...(found || experiment({ experiment_id: id })), archived_at: '2026-05-06T00:00:00Z' })
  })
  mockedApi.unarchive.mockImplementation((id: string) => {
    const found = items.find(item => item.experiment_id === id)
    return Promise.resolve({ ...(found || experiment({ experiment_id: id })), archived_at: null })
  })
  return render(
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <ExperimentList />
    </MemoryRouter>,
  )
}

function renderPaginatedList(items: Experiment[]) {
  mockedApi.list.mockImplementation((params?: { limit?: number; offset?: number }) => {
    const offset = params?.offset ?? 0
    const limit = params?.limit ?? 100
    return Promise.resolve({ total: items.length, experiments: items.slice(offset, offset + limit) })
  })
  mockedApi.archive.mockImplementation((id: string) => {
    const found = items.find(item => item.experiment_id === id)
    return Promise.resolve({ ...(found || experiment({ experiment_id: id })), archived_at: '2026-05-06T00:00:00Z' })
  })
  mockedApi.unarchive.mockImplementation((id: string) => {
    const found = items.find(item => item.experiment_id === id)
    return Promise.resolve({ ...(found || experiment({ experiment_id: id })), archived_at: null })
  })
  return render(
    <MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <ExperimentList />
    </MemoryRouter>,
  )
}

describe('ExperimentList', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    window.localStorage.clear()
    mockedApi.delete.mockResolvedValue({})
  })

  it('deduplicates experiments and hides raw backend stage labels', async () => {
    renderList([
      experiment({ experiment_id: 'exp-1', title: '固体称量实验' }),
      experiment({ experiment_id: 'exp-1', title: '固体称量实验' }),
      experiment({ experiment_id: 'exp-2', title: '双视角实验', status: 'running', processing_stage: 'video_analysis', completed_at: null, analyzed_at: null }),
    ])

    await waitFor(() => expect(screen.getByText('固体称量实验')).toBeInTheDocument())
    expect(screen.getAllByText('固体称量实验')).toHaveLength(1)
    expect(screen.queryByText(/output_generation/)).not.toBeInTheDocument()
    expect(screen.getAllByText('已完成').length).toBeGreaterThan(0)
  })

  it('keeps the recovered experiment-list surface copy', async () => {
    renderList([experiment({ experiment_id: 'exp-1', title: '固体称量实验' })])

    await waitFor(() => expect(screen.getByText('实验列表 Experiment List')).toBeInTheDocument())
    expect(screen.getByText('从这里创建实验、上传视频并启动分析。')).toBeInTheDocument()
    expect(screen.getByText('新建实验 New')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('搜索实验标题或 ID')).toBeInTheDocument()
    expect(screen.queryByText('实验证据队列')).not.toBeInTheDocument()
    expect(screen.queryByText('Experiment Evidence Queue')).not.toBeInTheDocument()
  })

  it('archives experiments before permanent deletion', async () => {
    renderList([
      experiment({ experiment_id: 'exp-1', title: '固体称量实验' }),
      experiment({ experiment_id: 'exp-2', title: '双视角实验' }),
    ])

    await waitFor(() => expect(screen.getByText('固体称量实验')).toBeInTheDocument())
    fireEvent.click(screen.getByLabelText('更多操作 固体称量实验'))
    fireEvent.click(screen.getByText('归档 Archive'))

    await waitFor(() => expect(mockedApi.archive).toHaveBeenCalledWith('exp-1'))
    await waitFor(() => expect(screen.queryByText('固体称量实验')).not.toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: /已归档 Archived/ }))
    await waitFor(() => expect(screen.getByText('固体称量实验')).toBeInTheDocument())
  })

  it('clears backend archived state when unarchiving', async () => {
    renderList([
      experiment({ experiment_id: 'exp-archived', title: '已归档实验', archived_at: '2026-05-06T00:00:00Z' }),
      experiment({ experiment_id: 'exp-active', title: '活跃实验' }),
    ])

    await waitFor(() => expect(screen.getByText('活跃实验')).toBeInTheDocument())
    expect(screen.queryByText('已归档实验')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /已归档 Archived/ }))
    await waitFor(() => expect(screen.getByText('已归档实验')).toBeInTheDocument())
    fireEvent.click(screen.getByLabelText('更多操作 已归档实验'))
    fireEvent.click(screen.getByText('取消归档 Unarchive'))

    await waitFor(() => expect(mockedApi.unarchive).toHaveBeenCalledWith('exp-archived'))
    await waitFor(() => expect(screen.queryByText('已归档实验')).not.toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: /全部 All/ }))
    await waitFor(() => expect(screen.getByText('已归档实验')).toBeInTheDocument())
  })

  it('loads additional backend pages before filtering and counting', async () => {
    const items = Array.from({ length: 105 }, (_, index) => experiment({
      experiment_id: `page-exp-${index}`,
      title: `分页实验 ${index}`,
      status: index >= 100 ? 'failed' : 'completed',
      created_at: `2026-05-${String((index % 9) + 1).padStart(2, '0')}T00:00:00Z`,
    }))
    renderPaginatedList(items)

    await waitFor(() => expect(screen.getByText('共 105 项')).toBeInTheDocument())
    expect(mockedApi.list).toHaveBeenCalledWith({ limit: 100 }, { force: true })
    expect(mockedApi.list).toHaveBeenCalledWith({ limit: 100, offset: 100 }, { force: true })

    fireEvent.click(screen.getByRole('button', { name: /失败 Failed/ }))
    await waitFor(() => expect(screen.getByText('共 5 项')).toBeInTheDocument())
  })

  it('shows the first page and loads more experiments on demand', async () => {
    const items = Array.from({ length: 22 }, (_, index) => experiment({
      experiment_id: `exp-${index}`,
      title: `实验 ${index}`,
      created_at: `2026-05-${String((index % 9) + 1).padStart(2, '0')}T00:00:00Z`,
    }))
    renderList(items)

    await waitFor(() => expect(screen.getByText('显示 18 / 22')).toBeInTheDocument())
    fireEvent.click(screen.getByText('加载更多'))
    expect(screen.getByText('显示 22 / 22')).toBeInTheDocument()
  })

  it('shows unknown confidence as empty and searches repaired Chinese titles', async () => {
    renderList([
      experiment({ experiment_id: 'exp-mojibake', title: 'ç°åºæ¼ç¤ºå®éª 2026-04-29', avg_confidence: null }),
      experiment({ experiment_id: 'exp-other', title: '另一个实验', avg_confidence: 0.88 }),
    ])

    await waitFor(() => expect(screen.getByText('现场演示实验 2026-04-29')).toBeInTheDocument())
    expect(screen.getByText('-')).toBeInTheDocument()

    fireEvent.change(screen.getByPlaceholderText('搜索实验标题或 ID'), { target: { value: '现场演示' } })
    expect(screen.getByText('现场演示实验 2026-04-29')).toBeInTheDocument()
    expect(screen.queryByText('另一个实验')).not.toBeInTheDocument()
  })

  it('keeps permanent delete behind confirmation', async () => {
    renderList([experiment({ experiment_id: 'exp-1', title: '固体称量实验' })])

    await waitFor(() => expect(screen.getByText('固体称量实验')).toBeInTheDocument())
    fireEvent.click(screen.getByLabelText('更多操作 固体称量实验'))
    fireEvent.click(screen.getByText('删除 Delete'))

    expect(screen.getByRole('dialog', { name: '删除实验' })).toBeInTheDocument()
    expect(mockedApi.delete).not.toHaveBeenCalled()
  })
})
