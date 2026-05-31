import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import ExperimentWorkspace from '../ExperimentWorkspace'
import { experimentApi } from '../../api'
import type { AnalysisOverview, ExperimentSubExperimentSegment, ExperimentSubExperimentsResponse } from '../../types'

vi.mock('../../api', () => ({
  experimentApi: {
    getAnalysisOverview: vi.fn(),
    getSubExperiments: vi.fn(),
    getPublishedMaterials: vi.fn(),
    getVideoCalibration: vi.fn(),
    updateVideoCalibration: vi.fn(),
    resetVideoCalibration: vi.fn(),
    uploadVideo: vi.fn(),
    process: vi.fn(),
    peekAnalysisOverview: vi.fn(),
    peekPublishedMaterials: vi.fn(),
  },
  invalidateExperimentCache: vi.fn(),
  prefetchExperimentRoute: vi.fn(),
}))

const mockedApi = experimentApi as unknown as {
  getAnalysisOverview: ReturnType<typeof vi.fn>
  getSubExperiments: ReturnType<typeof vi.fn>
  getPublishedMaterials: ReturnType<typeof vi.fn>
  getVideoCalibration: ReturnType<typeof vi.fn>
}

const baseSegmentResponse: ExperimentSubExperimentSegment = {
  segment_id: 'seg-1',
  index: 0,
  start_sec: 0,
  end_sec: 120,
  duration_sec: 120,
  experiment_window_duration_s: 118,
  preview_duration_s: 30,
  preview_mode: 'realtime',
  playback_speed_ratio: 1.0,
  actual_experiment_start: true,
  activity_segment_count: 1,
  third_person_video_url: null,
  first_person_video_url: null,
  third_person_annotated_video_url: null,
  first_person_annotated_video_url: null,
  third_person_poster_url: null,
  first_person_poster_url: null,
  preview_poster_url: null,
}

function subExperiments(overrides: Partial<ExperimentSubExperimentSegment> = {}, response: Partial<ExperimentSubExperimentsResponse> = {}) {
  return {
    total: 1,
    official_total: 0,
    candidate_total: 1,
    segment_preview_ready_count: 1,
    segments: [{ ...baseSegmentResponse, ...overrides }],
    sub_experiments: [],
    ...response,
  } as ExperimentSubExperimentsResponse
}

function overview(overrides: Partial<AnalysisOverview> = {}): AnalysisOverview {
  return {
    schema_version: 'analysis_overview.v1',
    experiment: { experiment_id: 'exp-1', experiment_name: 'Experiment A' },
    run: { run_id: 'run-1', result_version: 'v1', status: 'completed', stage: 'completed', progress: 1 },
    readiness: { summary_ready: true, steps_ready: true, alerts_ready: true, artifacts_ready: true, annotated_video_ready: true, writeback_ready: true },
    summary: { frame_count: 1, detection_count: 2, alert_count: 0, official_step_count: 0, candidate_step_count: 1, confirmed_step_count: 0, inferred_step_count: 0, avg_confidence: 0.8, model_name: 'best.pt' },
    steps: { official: [], candidate: [], inferred: [] },
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
    markers: { steps: [], alerts: [], evidence: [] },
    artifacts: {
      source_video: { name: 'source_video', ready: true, kind: 'mp4', size_bytes: 10, url: '/video.mp4' },
      annotated_video: { name: 'annotated_video', ready: true, kind: 'mp4', size_bytes: 10, url: '/annotated.mp4' },
    },
    debug: { trace_id: 'trace-1' },
    ...overrides,
  }
}

function renderWorkspace(
  subSegments: ExperimentSubExperimentsResponse = subExperiments(),
  overviewOverrides: Partial<AnalysisOverview> = {},
) {
  mockedApi.getAnalysisOverview.mockResolvedValue(overview(overviewOverrides))
  mockedApi.getSubExperiments.mockResolvedValue(subSegments)
  mockedApi.getPublishedMaterials.mockResolvedValue({ items: [] })
  mockedApi.getVideoCalibration.mockResolvedValue({
    schema_version: 'video_playback_calibration.v1',
    experiment_id: 'exp-1',
    offset_adjust_sec: 0,
    history: [],
  })

  return render(
    <MemoryRouter initialEntries={['/experiments/exp-1/workspace']} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        <Route path="/experiments/:id/workspace" element={<ExperimentWorkspace />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('ExperimentWorkspace segment preview comparison layout', () => {
  beforeEach(() => vi.resetAllMocks())

  it('uses only third_view_realtime_preview in the left third-person panel', async () => {
    const { container } = renderWorkspace(subExperiments({
      side_by_side_realtime_preview: '/legacy-dual.mp4',
      third_view_realtime_preview: '/third.mp4',
      first_view_realtime_preview: '/first.mp4',
    }))

    await waitFor(() => {
      const thirdPanel = container.querySelector('[data-smoke="segment-third-view-preview"]')
      const thirdVideo = thirdPanel?.querySelector('[data-smoke="segment-third-person-preview-video"]') as HTMLVideoElement | null
      expect(thirdVideo?.getAttribute('src')).toContain('/third.mp4')
      expect(thirdVideo?.getAttribute('src')).not.toContain('/legacy-dual.mp4')
    })
  })

  it('uses only first_view_realtime_preview in the right first-person panel', async () => {
    const { container } = renderWorkspace(subExperiments({
      side_by_side_realtime_preview: '/legacy-dual.mp4',
      third_view_realtime_preview: '/third.mp4',
      first_view_realtime_preview: '/first.mp4',
    }))

    await waitFor(() => {
      const firstPanel = container.querySelector('[data-smoke="segment-first-view-preview"]')
      const firstVideo = firstPanel?.querySelector('[data-smoke="segment-first-person-preview-video"]') as HTMLVideoElement | null
      expect(firstVideo?.getAttribute('src')).toContain('/first.mp4')
      expect(firstVideo?.getAttribute('src')).not.toContain('/legacy-dual.mp4')
    })
  })

  it('prefers explicit realtime preview URL aliases over path fields', async () => {
    const { container } = renderWorkspace(subExperiments({
      side_by_side_realtime_preview_url: '/legacy-dual.mp4',
      third_view_realtime_preview_url: '/third-url.mp4',
      third_view_realtime_preview: 'D:\\LabCapability\\LabSOPGuard\\outputs\\experiments\\exp-1\\windows\\w1\\third.mp4',
      first_view_realtime_preview_url: '/first-url.mp4',
      first_view_realtime_preview: 'D:\\LabCapability\\LabSOPGuard\\outputs\\experiments\\exp-1\\windows\\w1\\first.mp4',
    }))

    await waitFor(() => {
      const thirdPanel = container.querySelector('[data-smoke="segment-third-view-preview"]')
      const firstPanel = container.querySelector('[data-smoke="segment-first-view-preview"]')
      const thirdVideo = thirdPanel?.querySelector('[data-smoke="segment-third-person-preview-video"]') as HTMLVideoElement | null
      const firstVideo = firstPanel?.querySelector('[data-smoke="segment-first-person-preview-video"]') as HTMLVideoElement | null
      expect(thirdVideo?.getAttribute('src')).toContain('/third-url.mp4')
      expect(firstVideo?.getAttribute('src')).toContain('/first-url.mp4')
      expect(thirdVideo?.getAttribute('src')).not.toContain('/legacy-dual.mp4')
      expect(firstVideo?.getAttribute('src')).not.toContain('/legacy-dual.mp4')
    })
  })

  it('uses legacy single-view alias fields before showing placeholders', async () => {
    const { container } = renderWorkspace(subExperiments({
      third_view_realtime_preview: null,
      third_view_preview_url: '/third-alias.mp4',
      first_view_realtime_preview: null,
      first_preview_url: '/first-alias.mp4',
      side_by_side_realtime_preview_url: '/legacy-dual.mp4',
    }))

    await waitFor(() => {
      const thirdPanel = container.querySelector('[data-smoke="segment-third-view-preview"]')
      const firstPanel = container.querySelector('[data-smoke="segment-first-view-preview"]')
      const thirdVideo = thirdPanel?.querySelector('[data-smoke="segment-third-person-preview-video"]') as HTMLVideoElement | null
      const firstVideo = firstPanel?.querySelector('[data-smoke="segment-first-person-preview-video"]') as HTMLVideoElement | null
      expect(thirdVideo?.getAttribute('src')).toContain('/third-alias.mp4')
      expect(firstVideo?.getAttribute('src')).toContain('/first-alias.mp4')
    })
  })

  it('does not render a separate side-by-side preview section', async () => {
    renderWorkspace(subExperiments({
      side_by_side_realtime_preview: '/legacy-dual.mp4',
      first_view_realtime_preview: '/first.mp4',
      third_view_realtime_preview: '/third.mp4',
    }))

    await waitFor(() => {
      const legacyDualSection = document.querySelector('[data-smoke="segment-dual-view-preview"]')
      expect(legacyDualSection).toBeNull()
      expect(document.querySelector('[data-smoke="segment-dual-view-comparison"]')).toBeTruthy()
    })
  })

  it('shows first-person placeholder when first-person preview is missing', async () => {
    const { container } = renderWorkspace(subExperiments({
      first_view_realtime_preview: null,
      first_view_realtime_preview_url: null,
      first_person_video_url: null,
      first_person_annotated_video_url: null,
      first_person_poster_url: null,
      third_view_realtime_preview: '/third.mp4',
      side_by_side_realtime_preview_url: '/legacy-dual.mp4',
      fast_preview_url: '/fast-dual.mp4',
    }))

    await waitFor(() => {
      const firstPanel = container.querySelector('[data-smoke="segment-first-view-preview"]')
      expect(firstPanel?.textContent).toContain('第一人称预览待生成')
      expect(firstPanel?.querySelector('video')).toBeNull()
      expect(firstPanel?.textContent).not.toContain('/legacy-dual.mp4')
    })
  })

  it('shows third-person placeholder when third-person preview is missing', async () => {
    const { container } = renderWorkspace(subExperiments({
      third_view_realtime_preview: null,
      third_view_realtime_preview_url: null,
      third_person_video_url: null,
      third_person_annotated_video_url: null,
      third_person_poster_url: null,
      first_view_realtime_preview: '/first.mp4',
      side_by_side_realtime_preview_url: '/legacy-dual.mp4',
      fast_preview_url: '/fast-dual.mp4',
    }))

    await waitFor(() => {
      const thirdPanel = container.querySelector('[data-smoke="segment-third-view-preview"]')
      expect(thirdPanel?.textContent).toContain('第三人称预览待生成')
      expect(thirdPanel?.querySelector('video')).toBeNull()
      expect(thirdPanel?.textContent).not.toContain('/legacy-dual.mp4')
    })
  })

  it('renders a two-column synchronized comparison layout', async () => {
    const { container } = renderWorkspace(subExperiments({
      first_view_realtime_preview: '/first.mp4',
      third_view_realtime_preview: '/third.mp4',
    }))

    await waitFor(() => {
      const comparisonGrid = container.querySelector('[data-smoke="segment-dual-preview-grid"]')
      expect(comparisonGrid).toBeTruthy()
      expect(comparisonGrid?.className).toContain('md:grid-cols-2')
      expect(container.querySelector('[data-smoke="segment-third-view-preview"]')).toBeTruthy()
      expect(container.querySelector('[data-smoke="segment-first-view-preview"]')).toBeTruthy()
    })
  })

  it('renders synchronized controls and wires play/pause to both videos', async () => {
    const playSpy = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined as never)
    const pauseSpy = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined)

    const { container } = renderWorkspace(subExperiments({
      first_view_realtime_preview: '/first.mp4',
      third_view_realtime_preview: '/third.mp4',
    }))

    await waitFor(() => {
      expect(container.querySelector('[data-smoke="segment-dual-view-comparison-play"]')).toBeTruthy()
      expect(container.querySelector('[data-smoke="segment-dual-view-comparison-replay"]')).toBeTruthy()
      expect(container.querySelector('[data-smoke="segment-dual-view-comparison-seek"]')).toBeTruthy()
    })

    const playButton = container.querySelector('[data-smoke="segment-dual-view-comparison-play"]') as HTMLButtonElement | null
    const replayButton = container.querySelector('[data-smoke="segment-dual-view-comparison-replay"]') as HTMLButtonElement | null
    const seekInput = container.querySelector('[data-smoke="segment-dual-view-comparison-seek"]') as HTMLInputElement | null

    expect(playButton).toBeTruthy()
    expect(replayButton).toBeTruthy()
    expect(seekInput).toBeTruthy()

    fireEvent.click(playButton!)
    expect(playSpy).toHaveBeenCalledTimes(2)
    fireEvent.click(replayButton!)
    expect(playSpy).toHaveBeenCalledTimes(4)
    expect(pauseSpy).toHaveBeenCalledTimes(2)

    fireEvent.change(seekInput!, { target: { value: '1' } })
    fireEvent.blur(seekInput!)

    playSpy.mockRestore()
    pauseSpy.mockRestore()
  })

  it('does not render release-blocking or debug detail text on the default workspace', async () => {
    renderWorkspace(subExperiments({
      first_view_realtime_preview: '/first.mp4',
      third_view_realtime_preview: '/third.mp4',
    }))

    await waitFor(() => {
      expect(document.querySelector('[data-smoke="experiment-segments"]')).toBeTruthy()
      expect(document.body.textContent).not.toContain('调试详情')
      expect(document.body.textContent).not.toContain('正式发布状态')
      expect(document.body.textContent).not.toContain('实验片段发布被阻断')
      expect(document.body.textContent).not.toContain('当前结果仅用于复核')
    })
  })

  it('uses backend display_stages for runtime summary values', async () => {
    renderWorkspace(subExperiments({
      first_view_realtime_preview: '/first.mp4',
      third_view_realtime_preview: '/third.mp4',
    }), {
      run: {
        run_id: 'run-1',
        result_version: 'v1',
        status: 'completed',
        stage: 'completed',
        progress: 1,
        timing: {
          display_stages: [
            { stage: 'total_elapsed', label_zh: '总耗时', duration_sec: 123.4, available: true },
            { stage: 'time_alignment', label_zh: '视频时间戳对齐', duration_sec: 2, available: true },
            { stage: 'coarse_scan', label_zh: '长视频并行粗扫', duration_sec: null, available: false },
          ],
        },
      },
    })

    await waitFor(() => {
      expect(document.querySelector('[data-smoke="analysis-timing-summary"]')).toBeTruthy()
      expect(document.body.textContent).toContain('2.0 秒')
      expect(document.body.textContent).toContain('未记录')
      expect(document.body.textContent).not.toContain('0.0 秒')
    })
  })

  it('shows 未记录 instead of 0.0 seconds when runtime values are missing', async () => {
    renderWorkspace(subExperiments({
      first_view_realtime_preview: '/first.mp4',
      third_view_realtime_preview: '/third.mp4',
    }))

    await waitFor(() => {
      expect(document.querySelector('[data-smoke="analysis-timing-summary"]')).toBeTruthy()
      expect(document.body.textContent).toContain('未记录')
      expect(document.body.textContent).not.toContain('0.0 秒')
    })
  })
})
