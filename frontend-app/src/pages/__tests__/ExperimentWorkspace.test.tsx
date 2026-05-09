import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import ExperimentWorkspace from '../ExperimentWorkspace'
import { experimentApi } from '../../api'
import type { AnalysisOverview, StepRecord } from '../../types'

vi.mock('../../api', () => ({
  experimentApi: {
    getAnalysisOverview: vi.fn(),
    peekAnalysisOverview: vi.fn(),
    getPublishedMaterials: vi.fn(),
    peekPublishedMaterials: vi.fn(),
    uploadVideo: vi.fn(),
    process: vi.fn(),
  },
  invalidateExperimentCache: vi.fn(),
  prefetchExperimentRoute: vi.fn(),
}))

const mockedApi = experimentApi as unknown as {
  getAnalysisOverview: ReturnType<typeof vi.fn>
  peekAnalysisOverview: ReturnType<typeof vi.fn>
  getPublishedMaterials: ReturnType<typeof vi.fn>
  peekPublishedMaterials: ReturnType<typeof vi.fn>
  uploadVideo: ReturnType<typeof vi.fn>
  process: ReturnType<typeof vi.fn>
}

const step: StepRecord = {
  step_id: 'step-1',
  experiment_id: 'exp-1',
  step_index: 1,
  step_name: 'Candidate transfer step',
  step_description: 'candidate step',
  status: 'candidate',
  start_time_sec: 1,
  end_time_sec: 2,
  duration_sec: 1,
  confidence: 0.8,
  step_confidence: 'high',
  completed_by_inference: true,
  evidence_refs: [],
  parameters: [],
  created_at: '2026-04-20T00:00:00Z',
  updated_at: '2026-04-20T00:00:00Z',
}

function overview(overrides: Partial<AnalysisOverview> = {}): AnalysisOverview {
  return {
    schema_version: 'analysis_overview.v1',
    experiment: { experiment_id: 'exp-1', experiment_name: 'Experiment A' },
    run: { run_id: 'run-1', result_version: 'v1', status: 'completed', stage: 'completed', progress: 1, updated_at: '2026-04-20T00:00:00Z' },
    readiness: { summary_ready: true, steps_ready: true, alerts_ready: true, artifacts_ready: true, annotated_video_ready: true, writeback_ready: true },
    summary: { frame_count: 1, detection_count: 2, alert_count: 0, official_step_count: 0, candidate_step_count: 1, confirmed_step_count: 0, inferred_step_count: 0, avg_confidence: 0.8, model_name: 'best.pt' },
    steps: { official: [], candidate: [step], inferred: [] },
    scene_summary: {
      description: 'A hand transfers liquid.',
      activities: ['transfer'],
      objects: ['pipette', 'tube'],
      step_indicators: ['transfer'],
      ppe_assessment: { gloves: true },
      alerts: [],
      detections: [{ class_name: 'pipette' }],
    },
    alerts: [],
    markers: { steps: [{ id: 'step-1', label: 'Candidate transfer step', timestamp_sec: 1, kind: 'step' }], alerts: [], evidence: [] },
    artifacts: {
      source_video: { name: 'source_video', ready: true, kind: 'mp4', size_bytes: 10, url: '/video.mp4' },
      annotated_video: { name: 'annotated_video', ready: true, kind: 'mp4', size_bytes: 10, url: '/annotated.mp4' },
    },
    debug: { trace_id: 'trace-1' },
    ...overrides,
  }
}

function renderWorkspace(data: AnalysisOverview) {
  mockedApi.getAnalysisOverview.mockResolvedValue(data)
  mockedApi.getPublishedMaterials.mockResolvedValue({ items: [] })
  return render(
    <MemoryRouter
      initialEntries={[`/experiments/${data.experiment.experiment_id}/workspace`]}
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      <Routes><Route path="/experiments/:id/workspace" element={<ExperimentWorkspace />} /></Routes>
    </MemoryRouter>,
  )
}

describe('ExperimentWorkspace analysis overview contract', () => {
  beforeEach(() => vi.resetAllMocks())

  it('renders candidate steps when official is empty', async () => {
    renderWorkspace(overview())
    await waitFor(() => expect(screen.getAllByText('Candidate transfer step').length).toBeGreaterThan(0))
    expect(screen.getAllByText('candidate').length).toBeGreaterThan(0)
  })

  it('auto selects the first visible step', async () => {
    renderWorkspace(overview())
    await waitFor(() => expect(screen.getByText('confidence: 0.8000')).toBeInTheDocument())
  })

  it('renders structured scene sections instead of empty labels', async () => {
    renderWorkspace(overview())
    await waitFor(() => expect(screen.getByText(/activities:/)).toBeInTheDocument())
    expect(screen.getAllByText(/transfer/).length).toBeGreaterThan(0)
    expect(screen.getByText(/objects:/)).toBeInTheDocument()
    expect(screen.getByText(/pipette, tube/)).toBeInTheDocument()
    expect(screen.getByText(/step_indicators:/)).toBeInTheDocument()
  })

  it('shows generating state when steps are not ready', async () => {
    renderWorkspace(overview({
      run: { run_id: 'run-1', result_version: 'v1', status: 'generating_outputs', stage: 'output_generation', progress: 0.7 },
      readiness: { summary_ready: true, steps_ready: false, alerts_ready: true, artifacts_ready: true, annotated_video_ready: false, writeback_ready: false },
      steps: { official: [], candidate: [], inferred: [] },
      summary: { frame_count: 1, detection_count: 2, alert_count: 0, official_step_count: 0, candidate_step_count: 0, confirmed_step_count: 0, inferred_step_count: 0, avg_confidence: null, model_name: 'best.pt' },
    }))
    await waitFor(() => expect(screen.getByText('Generating structured steps...')).toBeInTheDocument())
  })

  it('loads material preview with the six item API limit', async () => {
    mockedApi.getAnalysisOverview.mockResolvedValue(overview())
    mockedApi.getPublishedMaterials.mockResolvedValue({
      total: 12,
      returned: 1,
      items: [{
        event_id: 'evt-preview',
        display_name: 'Limited preview material',
        event_type: 'liquid_transfer',
        time_start: 3,
        time_end: 5,
        published_paths: { clip: '/outputs/experiments/exp-1/published_materials/operator/clip.mp4' },
      }],
    })

    render(
      <MemoryRouter
        initialEntries={['/experiments/exp-1/workspace']}
        future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
      >
        <Routes><Route path="/experiments/:id/workspace" element={<ExperimentWorkspace />} /></Routes>
      </MemoryRouter>,
    )

    await waitFor(() => expect(screen.getByText('Limited preview material')).toBeInTheDocument())
    expect(mockedApi.getPublishedMaterials).toHaveBeenCalledWith('exp-1', { limit: 6 }, { force: false })
    expect(mockedApi.getPublishedMaterials.mock.calls.every(call => call[1]?.limit === 6)).toBe(true)
  })
})
