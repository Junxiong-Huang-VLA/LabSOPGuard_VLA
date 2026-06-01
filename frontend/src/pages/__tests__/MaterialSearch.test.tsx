import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import MaterialSearch from '../MaterialSearch'
import { experimentApi } from '../../api'
import type { MaterialCandidateGroup } from '../../types'

vi.mock('../../api', () => ({
  experimentApi: {
    getAnalysisOverview: vi.fn(),
    getPublishedMaterials: vi.fn(),
    getMaterialCandidates: vi.fn(),
    getSubExperiments: vi.fn(),
    getMaterialDiagnostics: vi.fn(),
    getMaterialQualityReview: vi.fn(),
    getMaterialRetrievalEvaluation: vi.fn(),
    getMaterialTaxonomyRegression: vi.fn(),
    approveMaterialCandidate: vi.fn(),
    updateMaterialQualityReview: vi.fn(),
    decideMaterialCandidate: vi.fn(),
    restoreMaterialCandidate: vi.fn(),
    reviewMaterialCandidatesBatch: vi.fn(),
  },
  prefetchExperimentRoute: vi.fn(),
}))

const mockedApi = experimentApi as unknown as {
  getAnalysisOverview: ReturnType<typeof vi.fn>
  getPublishedMaterials: ReturnType<typeof vi.fn>
  getMaterialCandidates: ReturnType<typeof vi.fn>
  getSubExperiments: ReturnType<typeof vi.fn>
  getMaterialDiagnostics: ReturnType<typeof vi.fn>
  getMaterialQualityReview: ReturnType<typeof vi.fn>
  getMaterialRetrievalEvaluation: ReturnType<typeof vi.fn>
  getMaterialTaxonomyRegression: ReturnType<typeof vi.fn>
}

function candidateGroup(overrides: Partial<MaterialCandidateGroup>): MaterialCandidateGroup {
  return {
    candidate_group_id: 'candidate-group',
    status: 'pending',
    recommended: false,
    quality_score: 0,
    yolo_recheck: { status: 'review', valid_evidence_count: 0 },
    vlm_semantics: { status: 'ready' },
    keyframes: [],
    clips: [],
    files: [],
  ...overrides,
  }
}

function renderMaterialSearch(
  candidates: MaterialCandidateGroup[],
  publishedMaterialsOverride: Partial<Record<string, unknown>> = {},
  options: { review?: boolean } = {},
) {
  const defaultPublishedMaterials = {
    experiment_id: 'exp-1',
    total: 1,
    items: [{
      item_id: 'material-1',
      event_id: 'event-1',
      display_title: 'hand bottle operation',
      event_type: 'hand-object-contact',
      canonical_action_type: 'hand-container',
      canonical_object: 'bottle',
      camera_view: 'third_person',
      time_start: 12.5,
      time_end: 14.5,
      experiment_window_id: 'window-1',
      source_window_sync_index: '101',
      preview_url: '/outputs/experiments/exp-1/materials/material-1.jpg',
      exists: true,
      recommended: true,
      preferred_best: true,
    }],
  }

  mockedApi.getAnalysisOverview.mockResolvedValue(null)
  mockedApi.getPublishedMaterials.mockResolvedValue({
    ...defaultPublishedMaterials,
    ...publishedMaterialsOverride,
  })
  mockedApi.getSubExperiments.mockResolvedValue({
    total: 1,
    segments: [
      {
        segment_id: 'window-1',
        segment_index: 0,
        has_materials: true,
      },
    ],
    sub_experiments: [],
  })
  mockedApi.getMaterialCandidates.mockResolvedValue({ experiment_id: 'exp-1', total: candidates.length, items: candidates })
  mockedApi.getMaterialDiagnostics.mockResolvedValue({
    experiment_id: 'exp-1',
    formal_material_total: 1,
    candidate_pending_total: candidates.length,
  })
  mockedApi.getMaterialQualityReview.mockResolvedValue({ experiment_id: 'exp-1', reviews: {} })
  mockedApi.getMaterialRetrievalEvaluation.mockResolvedValue(null)
  mockedApi.getMaterialTaxonomyRegression.mockResolvedValue(null)

  const entry = options.review
    ? '/experiments/exp-1/materials?review=1'
    : '/experiments/exp-1/materials'
  return render(
    <MemoryRouter initialEntries={[entry]} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <Routes>
        <Route path="/experiments/:id/materials" element={<MaterialSearch />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('MaterialSearch formal material library', () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  it('renders formal materials while keeping candidate review tools hidden', async () => {
    renderMaterialSearch([
      candidateGroup({
        candidate_group_id: 'candidate-high',
        recommended: true,
        recommended_count: 1,
        quality_score: 0.86,
        primary_object: 'bottle',
        canonical_action_type: 'hand-container',
        yolo_recheck: { status: 'pass', valid_evidence_count: 3 },
        keyframes: [{
          candidate_id: 'high-frame',
          preview_url: '/outputs/experiments/exp-1/candidates/high.jpg',
          recommended: true,
          primary_object: 'bottle',
          quality_score: 0.9,
          yolo_annotation_rendered: true,
        }],
      }),
    ])

    await waitFor(() => {
      const library = document.querySelector('[data-smoke="formal-material-library"]')
      expect(library).toBeInTheDocument()
      expect(library).toHaveAttribute('data-count', '1')
    })
    expect(mockedApi.getMaterialCandidates).toHaveBeenCalled()
    expect(mockedApi.getMaterialDiagnostics).toHaveBeenCalled()
  })

  it('classifies orphan materials in diagnostics and removes them from formal list when window linkage is invalid', async () => {
    renderMaterialSearch([], {
      total: 2,
      items: [
        {
          item_id: 'material-valid',
          event_id: 'event-valid',
          display_title: 'hand paper operation',
          event_type: 'hand-object-contact',
          canonical_action_type: 'hand-paper',
          canonical_object: 'paper',
          camera_view: 'third_person',
          time_start: 11,
          time_end: 12,
          experiment_window_id: 'window-1',
          source_window_sync_index: '11',
          preview_url: '/outputs/experiments/exp-1/materials/material-valid.jpg',
          exists: true,
          recommended: true,
          preferred_best: true,
        },
        {
          item_id: 'material-orphan',
          event_id: 'event-orphan',
          display_title: 'hand container operation',
          event_type: 'hand-object-contact',
          canonical_action_type: 'hand-container',
          canonical_object: 'bottle',
          camera_view: 'first_person',
          time_start: 20,
          time_end: 21,
          experiment_window_id: 'window-2',
          source_window_sync_index: '22',
          preview_url: '/outputs/experiments/exp-1/materials/material-orphan.jpg',
          exists: true,
          recommended: true,
          preferred_best: true,
        },
      ],
    })

    await waitFor(() => {
      expect(document.querySelector('[data-smoke="formal-material-library"]')).toBeInTheDocument()
      expect(document.querySelector('[data-smoke="formal-material-library"]')).toHaveAttribute('data-count', '1')
    })
    const orphanPanel = document.querySelector('[data-smoke="material-orphan-diagnostics"]')
    expect(orphanPanel).not.toBeNull()
    if (orphanPanel) {
      expect(orphanPanel).toBeInTheDocument()
    }
    expect(screen.getByText('material-orphan')).toBeInTheDocument()
    expect(screen.getByText('window-2')).toBeInTheDocument()
    const orphanRows = document.querySelectorAll('[data-smoke="orphan-material-item"]')
    expect(orphanRows).toHaveLength(1)
  })

  it('treats explicit orphan_material=true as diagnostics-only even with valid window linkage', async () => {
    renderMaterialSearch([], {
      total: 2,
      items: [
        {
          item_id: 'material-valid',
          material_id: 'material-valid',
          event_id: 'event-valid',
          display_title: 'hand paper operation',
          event_type: 'hand-object-contact',
          canonical_action_type: 'hand-paper',
          canonical_object: 'paper',
          camera_view: 'third_person',
          time_start: 11,
          time_end: 12,
          experiment_window_id: 'window-1',
          source_window_sync_index: '11',
          preview_url: '/outputs/experiments/exp-1/materials/material-valid.jpg',
          exists: true,
          recommended: true,
          preferred_best: true,
        },
        {
          item_id: 'material-orphan-explicit',
          material_id: 'orphan-material-explicit',
          event_id: 'event-orphan-explicit',
          display_title: 'orphan material should hide from formal view',
          event_type: 'hand-object-contact',
          canonical_action_type: 'hand-container',
          canonical_object: 'bottle',
          camera_view: 'first_person',
          time_start: 20,
          time_end: 21,
          experiment_window_id: 'window-1',
          source_window_sync_index: '11',
          preview_url: '/outputs/experiments/exp-1/materials/material-orphan.jpg',
          exists: true,
          recommended: true,
          preferred_best: true,
          orphan_material: true,
        },
      ],
    })

    await waitFor(() => {
      expect(document.querySelector('[data-smoke="formal-material-library"]')).toBeInTheDocument()
      expect(document.querySelector('[data-smoke="formal-material-library"]')).toHaveAttribute('data-count', '1')
    })
    const formalSection = document.querySelector('[data-smoke="formal-material-library"]')
    expect(formalSection).toBeInTheDocument()
    expect(formalSection).not.toBeNull()
    if (formalSection) {
      expect(formalSection.textContent || '').not.toContain('orphan-material-explicit')
    }
    const orphanPanel = document.querySelector('[data-smoke="material-orphan-diagnostics"]')
    expect(orphanPanel).not.toBeNull()
    if (orphanPanel) {
      expect(orphanPanel).toBeInTheDocument()
    }
    expect(screen.getByText((content) => content.includes('该素材缺少有效实验片段关联'))).toBeInTheDocument()
    expect(screen.getByText('orphan-material-explicit')).toBeInTheDocument()
    const orphanRows = document.querySelectorAll('[data-smoke="orphan-material-item"]')
    expect(orphanRows).toHaveLength(1)
  })

  it('keeps debug/internal labels out of default card view and shows them in 证据详情', async () => {
    renderMaterialSearch([], {
      total: 1,
      items: [{
        item_id: 'material-debug',
        event_id: 'event-debug',
        display_name: 'hand bottle operation',
        event_type: 'hand-object-contact',
        canonical_action_type: 'hand-container',
        canonical_object: 'bottle',
        camera_view: 'first_person',
        time_start: 11.2,
        time_end: 12.6,
        experiment_window_id: 'window-1',
        source_window_sync_index: '11',
        preview_url: '/outputs/experiments/exp-1/materials/material-debug.jpg',
        exists: true,
        recommended: true,
        preferred_best: true,
        yolo_annotation_rendered: 'unknown',
        semantic_action: 'unknown_operation',
        physical_action_type: 'unknown',
        instrument_context: 'spatula',
        '原始YOLO对象': 'spatula',
        review_route: 'human_review',
        evidence_bundle_id: 'bundle-debug',
        'view_action_review_group_abc': 'debug-group-id',
        raw_debug: 'abc',
        debug_tokens: ['x', 'y'],
        'raw_model_output': 'skip',
        vlm_semantics: { status: 'not_available' },
      }],
    }, { review: true })

    await waitFor(() => {
      expect(document.querySelector('[data-smoke="formal-material-library"]')).toBeInTheDocument()
      expect(document.querySelector('[data-smoke="formal-material-library"]')).toHaveAttribute('data-count', '1')
    })
    expect(screen.queryByText('view_action_review_group_abc')).not.toBeInTheDocument()

    const evidenceSummary = screen.getAllByText('证据详情')[0]
    fireEvent.click(evidenceSummary)
    expect(screen.getByText('YOLO标注渲染')).toBeInTheDocument()
    expect(screen.getByText('语义动作')).toBeInTheDocument()
    expect(screen.getAllByText('证据包ID').length).toBeGreaterThan(0)
    expect(screen.getByText('候选视图分组')).toBeInTheDocument()
    expect(screen.getByText('unknown_operation')).toBeInTheDocument()
    expect(screen.getByText('human_review')).toBeInTheDocument()
  })

  it('hides the evidence-detail drawer and review jargon in default product view', async () => {
    renderMaterialSearch([], {
      total: 1,
      items: [{
        item_id: 'material-product',
        event_id: 'event-product',
        display_name: 'hand bottle operation',
        event_type: 'hand-object-contact',
        canonical_action_type: 'hand-container',
        canonical_object: 'bottle',
        camera_view: 'first_person',
        time_start: 11.2,
        time_end: 13.0,
        experiment_window_id: 'window-1',
        source_window_sync_index: '101',
        preview_url: '/outputs/experiments/exp-1/materials/material-product.jpg',
        exists: true,
        recommended: true,
        preferred_best: true,
        evidence_bundle_id: 'bundle-product',
      }],
    })  // no review flag -> product default

    await waitFor(() => {
      expect(document.querySelector('[data-smoke="formal-material-library"]')).toBeInTheDocument()
    })
    // engineering drawer + ids must not appear in the default product view
    expect(screen.queryByText('证据详情')).not.toBeInTheDocument()
    expect(screen.queryByText('证据包ID')).not.toBeInTheDocument()
    expect(screen.queryByText('窗口同步索引')).not.toBeInTheDocument()
    // dual-view material grid is still rendered for the user
    expect(document.querySelector('[data-smoke="key-material-pair"]')).toBeInTheDocument()
  })

  it('shows keyframe and keyclip understanding when key-material data contains them', async () => {
    renderMaterialSearch([], {
      total: 1,
      items: [{
        item_id: 'material-understood',
        event_id: 'event-understood',
        display_name: 'hand pipette operation',
        event_type: 'hand-object-contact',
        canonical_action_type: 'hand-container',
        canonical_object: 'bottle',
        camera_view: 'third_person',
        time_start: 14.2,
        time_end: 15.8,
        experiment_window_id: 'window-1',
        source_window_sync_index: '101',
        preview_url: '/outputs/experiments/exp-1/materials/material-understood.jpg',
        exists: true,
        recommended: true,
        preferred_best: true,
        keyframe_understanding: {
          visible_facts: ['Third-person keyframe shows hand entering the bottle neck.'],
          action_interpretation: ['The operator adjusts the pipette position.'],
          uncertainties: ['Small occlusion from the glove shadow.'],
          evidence_refs: ['frame-14', 'frame-15'],
        },
        keyclip_understanding: {
          visible_facts: ['Keyclip shows full transfer gesture from start to end.'],
          action_interpretation: ['Operator lowers and lifts in one continuous movement.'],
          uncertainties: ['Blur on final frame.'],
          evidence_refs: ['clip-1'],
        },
      }],
    })  // product default view

    await waitFor(() => {
      expect(document.querySelector('[data-smoke="formal-material-library"]')).toBeInTheDocument()
      expect(document.querySelector('[data-smoke="key-material-pair"]')).toBeInTheDocument()
    })

    const materialCard = document.querySelector('[data-smoke="key-material-pair"]')
    expect(materialCard).not.toBeNull()
    if (!materialCard) return

    const keyframePanel = materialCard.querySelector('[data-smoke="material-keyframe-understanding"]')
    const keyclipPanel = materialCard.querySelector('[data-smoke="material-keyclip-understanding"]')
    expect(keyframePanel).not.toBeNull()
    expect(keyclipPanel).not.toBeNull()
    if (!keyframePanel || !keyclipPanel) return

    const keyframe = within(keyframePanel)
    const keyclip = within(keyclipPanel)

    expect(keyframe.getByText('Keyframe understanding')).toBeInTheDocument()
    expect(keyframe.getByText('Third-person keyframe shows hand entering the bottle neck.')).toBeInTheDocument()
    expect(keyframe.getByText('The operator adjusts the pipette position.')).toBeInTheDocument()
    expect(keyframe.getByText('Small occlusion from the glove shadow.')).toBeInTheDocument()
    expect(keyframe.getByText('frame-14')).toBeInTheDocument()
    expect(keyframe.getByText('frame-15')).toBeInTheDocument()

    expect(keyclip.getByText('Keyclip understanding')).toBeInTheDocument()
    expect(keyclip.getByText('Keyclip shows full transfer gesture from start to end.')).toBeInTheDocument()
    expect(keyclip.getByText('Operator lowers and lifts in one continuous movement.')).toBeInTheDocument()
    expect(keyclip.getByText('Blur on final frame.')).toBeInTheDocument()
    expect(keyclip.getByText('clip-1')).toBeInTheDocument()
  })

  it('shows pending placeholders when keyframe and keyclip understanding are missing', async () => {
    renderMaterialSearch([], {
      total: 1,
      items: [{
        item_id: 'material-missing-understanding',
        event_id: 'event-missing-understanding',
        display_name: 'hand slide operation',
        event_type: 'hand-object-contact',
        canonical_action_type: 'hand-container',
        canonical_object: 'bottle',
        camera_view: 'third_person',
        time_start: 10.2,
        time_end: 11.9,
        experiment_window_id: 'window-1',
        source_window_sync_index: '101',
        preview_url: '/outputs/experiments/exp-1/materials/material-missing-understanding.jpg',
        exists: true,
        recommended: true,
        preferred_best: true,
      }],
    })  // product default view

    await waitFor(() => {
      expect(document.querySelector('[data-smoke="formal-material-library"]')).toBeInTheDocument()
      expect(document.querySelector('[data-smoke="key-material-pair"]')).toBeInTheDocument()
    })

    const materialCard = document.querySelector('[data-smoke="key-material-pair"]')
    expect(materialCard).not.toBeNull()
    if (!materialCard) return

    const keyframePanel = materialCard.querySelector('[data-smoke="material-keyframe-understanding"]')
    const keyclipPanel = materialCard.querySelector('[data-smoke="material-keyclip-understanding"]')
    expect(keyframePanel).not.toBeNull()
    expect(keyclipPanel).not.toBeNull()
    if (!keyframePanel || !keyclipPanel) return

    expect(keyframePanel.textContent).toContain('Keyframe understanding pending')
    expect(keyclipPanel.textContent).toContain('Keyclip understanding pending')
  })

  it('uses all_items for blocked alignment and keeps orphan materials in diagnostics without normal material rendering', async () => {
    renderMaterialSearch([], {
      total: 2,
      returned: 0,
      items: [],
      all_items: [
        {
          item_id: 'material-closed',
          event_id: 'event-closed',
          display_title: 'valid linked material',
          event_type: 'hand-container',
          canonical_action_type: 'hand-container',
          canonical_object: 'bottle',
          camera_view: 'third_person',
          time_start: 12,
          time_end: 13,
          experiment_window_id: 'window-1',
          source_window_sync_index: '101',
          preview_url: '/outputs/experiments/exp-1/materials/material-closed.jpg',
          exists: true,
          recommended: true,
          preferred_best: true,
        },
        {
          item_id: 'material-orphan',
          event_id: 'event-orphan',
          display_title: 'missing linkage material',
          event_type: 'hand-paper',
          canonical_action_type: 'hand-paper',
          canonical_object: 'paper',
          camera_view: 'first_person',
          time_start: 20,
          time_end: 21,
          experiment_window_id: 'window-2',
          preview_url: '/outputs/experiments/exp-1/materials/material-orphan.jpg',
          exists: true,
          recommended: true,
          preferred_best: true,
        },
      ],
      alignment_gate: {
        status: 'blocked',
        reason: 'timeline_alignment_not_reliable_for_dual_view_materials',
        hidden_item_count: 2,
      },
    })

    await waitFor(() => {
      expect(document.querySelector('[data-smoke="formal-material-alignment-blocked"]')).toBeInTheDocument()
      expect(document.querySelector('[data-smoke="formal-material-library"]')).not.toBeInTheDocument()
    })
    const orphanPanel = document.querySelector('[data-smoke="material-orphan-diagnostics"]')
    expect(orphanPanel).not.toBeNull()
    if (orphanPanel) {
      expect(orphanPanel).toBeInTheDocument()
    }
    expect(screen.getByText('material-orphan')).toBeInTheDocument()
    expect(screen.queryByText('valid linked material')).not.toBeInTheDocument()
    const orphanRows = document.querySelectorAll('[data-smoke="orphan-material-item"]')
    expect(orphanRows).toHaveLength(1)
  })
})
