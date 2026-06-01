# Key Material Page Review Package

> Scope: Key material page (MaterialSearch experiment view), related APIs/types/tests only. No code changes made in this package.

## 1) Primary source files

- `frontend/src/pages/MaterialSearch.tsx`
  - Core page implementation for key material review.
  - Handles route parsing (`experimentId`, `onlyKeyMaterials`, `onlyConfident`), lifecycle loading, and UI composition.
  - Implements material understanding parsing/rendering, keyframe/keyclip display, review toggles, orphan diagnostics, and candidate review controls.
- `frontend/src/api.ts`
  - Defines experiment/material endpoints used by MaterialSearch:
    - `getAnalysisOverview`, `getSubExperiments`, `getPublishedMaterials`, `getMaterialCandidates`, `getMaterialDiagnostics`
    - Candidate actions: `approveMaterialCandidate`, `confirmMaterialCandidate`, `renameMaterialCandidate`, `decideMaterialCandidate`, `decideMaterialCandidates`, `addMaterialCandidate`, `removeMaterialCandidate`
    - Workspace material helpers (`getPublishedMaterials`, `getPublishedHealth`) used for related material metadata context.
- `frontend/src/types.ts`
  - Material model/types consumed by MaterialSearch:
    - `MaterialSearchItem`, `MaterialSearchResponse`, `WorkspacePublishedMaterialsResponse`
    - `MaterialDiagnosticsResponse`, `MaterialDiagnosticsEvidenceItem`
    - `MaterialCandidateGroup`, `MaterialCandidateFile`, `MaterialCandidatesResponse`

## 2) Test coverage files

- `frontend/src/pages/__tests__/MaterialSearch.test.tsx`
  - Covers MaterialSearch rendering and behavior for:
    - formal and candidate loading paths
    - orphan classification and diagnostics rendering
    - understanding panel behavior
    - blocked-alignment branch behavior
    - debug label/evidence visibility

## 3) Related but non-primary context

- `frontend/src/pages/WorkspaceMaterials.tsx`
  - Workspace-level material listing page; related to materials domain but not the key material page.

## 4) Current build/test status (no new changes executed in this turn)

- Build:
  - `npm run build` (from `frontend`) failed
  - TypeScript parse errors in `MaterialSearch.tsx` at approximate lines 945, 947, 951 (`TS1381: Unexpected token`), blocking full build.

- Tests:
  - `npm run test -- src/pages/__tests__/MaterialSearch.test.tsx` reported 8 tests total with 4 failures.
  - Existing failures are mostly text/assertion mismatches (expected string vs rendered output / content variation) in MaterialSearch test cases.

## 5) Quick technical note (for review focus)

- `MaterialSearch.tsx` is the main orchestrator for the key-material workflow.
- Data model relies on `MaterialSearchItem` fields for evidence + keyframe/keyclip references:
  - `first_keyframe`, `third_keyframe`, `first_keyclip`, `third_keyclip`, `side_by_side_keyclip`, `source_window_sync_index`, `review_status`, `vlm_semantics`, etc.
- Orphan flow and diagnostics:
  - `isOrphanMaterial` and `isOrphanCandidateMaterial` style predicates used for orphan filtering/flagging.
  - Diagnostic rendering depends on `getMaterialDiagnostics` with evidence grouped by window and type, and includes optional collapsed/expanded details logic.
- Candidate flow includes explicit actions for approval/confirmation/renaming/decision plus multi-decision and add/remove operations.

## 6) Review status summary

- Scope collected with intentional boundaries.
- No edits were made to application logic; this is documentation-only synthesis for review.
