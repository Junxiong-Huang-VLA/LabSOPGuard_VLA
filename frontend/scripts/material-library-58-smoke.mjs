import { mkdir } from 'node:fs/promises'
import path from 'node:path'
import { spawn, spawnSync } from 'node:child_process'
import net from 'node:net'
import { chromium } from 'playwright'

const experimentId = process.env.MATERIAL58_SMOKE_EXPERIMENT_ID || 'solid-weighing-dual-view-20260508-153648'
const requestedPort = Number(process.env.MATERIAL58_SMOKE_PORT || 4178)
const mockApi = process.env.MATERIAL58_SMOKE_MOCK !== '0'
const fallbackBaseUrl = process.env.MATERIAL58_SMOKE_BASE_URL

function normalizeBaseUrl(value) {
  return (value || '').replace(/\/$/, '')
}

function isPortFree(port) {
  return new Promise((resolve) => {
    const server = net.createServer()
    server.once('error', () => resolve(false))
    server.once('listening', () => {
      server.close(() => resolve(true))
    })
    server.listen(port, '127.0.0.1')
  })
}

async function resolvePreviewPort() {
  if (Number.isFinite(requestedPort) && requestedPort > 0) {
    if (await isPortFree(requestedPort)) {
      return requestedPort
    }
    if (process.env.MATERIAL58_SMOKE_PORT) {
      return requestedPort
    }
  }
  const startPort = Number.isFinite(requestedPort) && requestedPort > 0 ? requestedPort : 4178
  for (let port = startPort; port <= startPort + 50; port += 1) {
    if (await isPortFree(port)) {
      return port
    }
  }
  return startPort
}

const canonicalItems = [
  ['hand-paper', 'paper', 'Hand-paper evidence', 'keyframe'],
  ['hand-bottle', 'bottle', 'Hand-bottle evidence', 'keyframe'],
  ['hand-spatula', 'spatula', 'Hand-spatula evidence', 'keyframe'],
  ['hand-balance', 'balance', 'Hand-balance evidence', 'keyframe'],
  ['hand-container', 'container', 'Hand-container evidence', 'clip'],
]

const published = {
  schema_version: 'published_materials.approved_material_references.v1',
  total: canonicalItems.length,
  returned: canonicalItems.length,
  taxonomy: canonicalItems.map(([action]) => action),
  items: canonicalItems.map(([action, object, displayName, kind], index) => ({
    item_id: `${action}-${kind}`,
    material_id: `${action}-${kind}`,
    experiment_id: experimentId,
    event_id: `${action}-event`,
    display_name: displayName,
    event_type: action,
    canonical_action_type: action,
    canonical_object: object,
    sop_phase: `${object}-phase`,
    interaction_family: 'hand-object',
    time_start: index + 1,
    time_end: index + 1.5,
    review_status: 'accepted',
    object_labels: [object],
    actions: [action],
    best_score: 0.9 - index * 0.03,
    best_reason: `${action} representative YOLO-backed evidence`,
    published_paths: {},
  })),
  index_lifecycle: {
    schema_version: 'workspace_published_materials_lifecycle.v1',
    status: 'ok',
    sqlite_count: canonicalItems.length,
    expected_indexable_count: canonicalItems.length,
    warnings: [],
  },
}

const candidates = {
  schema_version: 'material_candidates.api.v1',
  experiment_id: experimentId,
  total: 4,
  file_total: 4,
  pending_total: 0,
  approved_total: 1,
  rejected_total: 1,
  deferred_total: 1,
  not_selected_total: 1,
  items: [
    { candidate_group_id: 'approved', status: 'approved', review_status: 'approved', canonical_action_type: 'hand-paper', files: [{ candidate_id: 'a', candidate_status: 'approved' }], keyframes: [], clips: [] },
    { candidate_group_id: 'rejected', status: 'rejected', review_status: 'rejected', canonical_action_type: 'hand-bottle', files: [{ candidate_id: 'r', candidate_status: 'rejected' }], keyframes: [], clips: [] },
    { candidate_group_id: 'deferred', status: 'deferred', review_status: 'deferred', canonical_action_type: 'hand-spatula', files: [{ candidate_id: 'd', candidate_status: 'deferred' }], keyframes: [], clips: [] },
    { candidate_group_id: 'not-selected', status: 'not_selected', review_status: 'not_selected', canonical_action_type: 'hand-container', files: [{ candidate_id: 'n', candidate_status: 'not_selected' }], keyframes: [], clips: [] },
  ],
}

const diagnostics = {
  schema_version: 'material_diagnostics.v1',
  experiment_id: experimentId,
  published_total: canonicalItems.length,
  formal_material_total: canonicalItems.length,
  best_material_total: canonicalItems.length,
  candidate_total: 4,
  candidate_pending_total: 0,
  candidate_approved_total: 1,
  candidate_rejected_total: 1,
  candidate_deferred_total: 1,
  candidate_not_selected_total: 1,
  file_access_status: 'ok',
  evidence_items: [],
  taxonomy_calibration: { per_action: {} },
}

function fail(message, detail) {
  const suffix = detail === undefined ? '' : `\n${JSON.stringify(detail, null, 2)}`
  throw new Error(`${message}${suffix}`)
}

async function waitForServer(url, timeoutMs = 30_000) {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(2_000) })
      if (response.ok) return
    } catch {
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }
  fail(`Preview server did not become ready: ${url}`)
}

async function startPreview() {
  if (process.env.MATERIAL58_SMOKE_EXTERNAL_SERVER === '1') return null
  const port = await resolvePreviewPort()
  const baseUrl = normalizeBaseUrl(fallbackBaseUrl || `http://127.0.0.1:${port}`)
  const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'
  const child = spawn(npm, ['run', 'preview', '--', '--host', '127.0.0.1', '--port', String(port)], {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, BROWSER: 'none' },
    shell: process.platform === 'win32',
  })
  child.stdout.on('data', chunk => process.stdout.write(chunk))
  child.stderr.on('data', chunk => process.stderr.write(chunk))
  await waitForServer(baseUrl)
  return { child, baseUrl }
}

function stopPreview(child) {
  if (!child) return
  if (process.platform === 'win32') {
    spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore' })
  } else {
    child.kill('SIGTERM')
  }
}

async function installMockRoutes(page) {
  if (!mockApi) return
  await page.route('**/api/v1/**', async route => {
    const url = new URL(route.request().url())
    const pathname = url.pathname
    let body
    if (pathname === '/api/v1/materials/published/health') body = published.index_lifecycle
    else if (pathname === '/api/v1/materials/published') body = published
    else if (pathname.endsWith('/materials/published')) body = published
    else if (pathname.endsWith('/materials/candidates')) body = candidates
    else if (pathname.endsWith('/materials/diagnostics')) body = diagnostics
    else if (pathname.endsWith('/materials/professional-reports')) body = { schema_version: 'professional_report_materials.v1', total: 1, items: [{ file_name: 'professional_report_qwen36max.pdf', grid_policy: 'professional_report_only' }] }
    else body = {}
    await route.fulfill({
      status: 200,
      contentType: 'application/json; charset=utf-8',
      body: JSON.stringify(body),
    })
  })
}

async function run() {
  const preview = await startPreview()
  const server = preview?.child ?? null
  const baseUrl = preview?.baseUrl || normalizeBaseUrl(fallbackBaseUrl || `http://127.0.0.1:${requestedPort}`)
  const browser = await chromium.launch()
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  try {
    await installMockRoutes(page)
    await page.goto(`${baseUrl}/materials?experiment_id=${encodeURIComponent(experimentId)}`)
    await page.waitForSelector('[data-smoke="workspace-material-grid"]', { timeout: 30_000 })
    const workspaceCount = Number(await page.getAttribute('[data-smoke="workspace-material-grid"]', 'data-filtered-count'))
    if (workspaceCount !== canonicalItems.length) fail('Unexpected /materials grid count', { workspaceCount })
    const workspaceText = await page.textContent('body')
    for (const [action] of canonicalItems) {
      const label = action.replace('hand-', 'Hand-')
      if (!workspaceText?.includes(label)) fail(`/materials missing canonical label ${label}`)
    }
    if (workspaceText?.includes('professional_report_qwen36max.pdf')) fail('Professional PDF leaked into /materials grid')

    await page.goto(`${baseUrl}/materials/review?experiment_id=${encodeURIComponent(experimentId)}`)
    await page.waitForURL(`**/experiments/${encodeURIComponent(experimentId)}/materials/review`, { timeout: 30_000 })
    await page.waitForSelector('[data-smoke="material-review-metrics"]', { timeout: 30_000 })
    const reviewText = await page.textContent('body')
    if (!reviewText?.includes('rejected 1 / deferred 1')) fail('/materials/review missing rejected/deferred summary')
    const reviewGrid = await page.$('[data-smoke="material-review-grid"]')
    if (reviewGrid) fail('Default pending review queue should be empty in 5.8 smoke fixture')

    console.log(JSON.stringify({
      status: 'passed',
      baseUrl,
      experimentId,
      routes: ['/materials', '/materials/review'],
      canonical_actions: canonicalItems.map(([action]) => action),
      default_pending_queue: 0,
      professional_report_grid_policy: 'excluded',
    }, null, 2))
  } catch (error) {
    await mkdir(path.join('output', 'playwright'), { recursive: true })
    await page.screenshot({ path: path.join('output', 'playwright', 'material-library-58-smoke-failure.png'), fullPage: true }).catch(() => undefined)
    throw error
  } finally {
    await browser.close()
    stopPreview(server)
  }
}

run().catch(error => {
  console.error(error?.stack || error)
  process.exit(1)
})
