import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import ErrorBoundary from './components/ErrorBoundary'
import Layout from './components/Layout'

const ExperimentList = lazy(() => import('./pages/ExperimentList'))
const ExperimentWorkspace = lazy(() => import('./pages/ExperimentWorkspace'))
const ExperimentTimeline = lazy(() => import('./pages/ExperimentTimeline'))
const ExperimentReport = lazy(() => import('./pages/ExperimentReport'))
const VideoAnalysis = lazy(() => import('./pages/VideoAnalysis'))
const StepDetail = lazy(() => import('./pages/StepDetail'))
const StepReview = lazy(() => import('./pages/StepReview'))
const JsonViewer = lazy(() => import('./pages/JsonViewer'))
const Upload = lazy(() => import('./pages/Upload'))
const MaterialSearch = lazy(() => import('./pages/MaterialSearch'))
const WorkspaceMaterials = lazy(() => import('./pages/WorkspaceMaterials'))
const MaterialTimelineView = lazy(() => import('./pages/MaterialTimelineView'))
const KeyActionIndex = lazy(() => import('./pages/KeyActionIndex'))
const KeyActionReviewQueue = lazy(() => import('./pages/KeyActionReviewQueue'))

function PageFallback() {
  return <div className="py-12 text-center text-gray-500">加载中 Loading...</div>
}

export default function App() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/experiments" replace />} />
            <Route path="experiments" element={<ExperimentList />} />
            <Route path="materials" element={<WorkspaceMaterials />} />
            <Route path="experiments/:id/workspace" element={<ExperimentWorkspace />} />
            <Route path="experiments/:id/video-analysis" element={<VideoAnalysis />} />
            <Route path="experiments/:id/report" element={<ExperimentReport />} />
            <Route path="experiments/:id/timeline" element={<ExperimentTimeline />} />
            <Route path="experiments/:id/materials" element={<MaterialSearch />} />
            <Route path="experiments/:id/materials/review" element={<MaterialSearch />} />
            <Route path="experiments/:id/materials/timeline" element={<MaterialTimelineView />} />
            <Route path="experiments/:id/steps/:stepId" element={<StepDetail />} />
            <Route path="experiments/:id/steps/:stepId/review" element={<StepReview />} />
            <Route path="experiments/:id/json" element={<JsonViewer />} />
            <Route path="experiments/:id/key-actions" element={<KeyActionIndex />} />
            <Route path="experiments/:id/key-actions/review" element={<KeyActionReviewQueue />} />
            <Route path="upload" element={<Upload />} />
            <Route path="*" element={<Navigate to="/experiments" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </ErrorBoundary>
  )
}
