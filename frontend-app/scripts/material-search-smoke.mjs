import { mkdir } from 'node:fs/promises'
import path from 'node:path'
import { chromium, request as playwrightRequest } from 'playwright'

const baseUrl = (process.env.MATERIAL_SMOKE_BASE_URL || 'http://127.0.0.1:5173').replace(/\/$/, '')
const experimentId = process.env.MATERIAL_SMOKE_EXPERIMENT_ID || '2190fe06-3619-45fc-96ef-1bb8afb9bdf9'
const operatorRole = process.env.MATERIAL_SMOKE_OPERATOR_ROLE || 'admin'
const shouldReindex = process.env.MATERIAL_SMOKE_REINDEX !== '0'
const expected = {
  materialCount: Number(process.env.MATERIAL_SMOKE_EXPECTED_MATERIALS || 4),
  imageCount: Number(process.env.MATERIAL_SMOKE_EXPECTED_IMAGES || 2),
  videoCount: Number(process.env.MATERIAL_SMOKE_EXPECTED_VIDEOS || 2),
}
const semanticTerms = [
  ['烧杯', 2],
  ['容器操作', 2],
  ['戴手套操作', 4],
  ['pouring liquid', 2],
]

function fail(message, detail) {
  const suffix = detail === undefined ? '' : `\n${JSON.stringify(detail, null, 2)}`
  throw new Error(`${message}${suffix}`)
}

function assertEqual(actual, wanted, message) {
  if (actual !== wanted) fail(`${message}: expected ${wanted}, got ${actual}`)
}

function assertAtLeast(actual, wanted, message) {
  if (actual < wanted) fail(`${message}: expected at least ${wanted}, got ${actual}`)
}

async function getJson(api, url) {
  const response = await api.get(url, { headers: { 'X-Operator-Role': operatorRole } })
  if (!response.ok()) fail(`GET ${url} failed`, { status: response.status(), body: await response.text() })
  return response.json()
}

async function post(api, url) {
  const response = await api.post(url, { headers: { 'X-Operator-Role': operatorRole } })
  if (!response.ok()) fail(`POST ${url} failed`, { status: response.status(), body: await response.text() })
  return response.json()
}

async function waitForLoadedMedia(page, selector, count, kind) {
  await page.waitForFunction(
    ({ selector: mediaSelector, count: expectedCount, kind: mediaKind }) => {
      const nodes = Array.from(document.querySelectorAll(mediaSelector))
      if (nodes.length !== expectedCount) return false
      if (mediaKind === 'image') return nodes.every(node => node.complete && node.naturalWidth > 0 && node.naturalHeight > 0)
      return nodes.every(node => node.readyState >= 1 && !node.error)
    },
    { selector, count, kind },
    { timeout: 30_000 },
  )
}

async function workspaceSnapshot(page) {
  return page.evaluate(() => {
    const metrics = document.querySelector('[data-smoke="workspace-material-metrics"]')
    const grid = document.querySelector('[data-smoke="workspace-material-grid"]')
    const images = Array.from(document.querySelectorAll('img[data-smoke="workspace-formal-image"]')).map(img => ({
      src: img.currentSrc || img.src,
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
    }))
    const videos = Array.from(document.querySelectorAll('video[data-smoke="workspace-formal-video"]')).map(video => ({
      src: video.currentSrc || video.src,
      readyState: video.readyState,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
    }))
    return {
      total: Number(metrics?.getAttribute('data-total') || 0),
      filtered: Number(metrics?.getAttribute('data-filtered') || grid?.getAttribute('data-filtered-count') || 0),
      healthStatus: metrics?.getAttribute('data-health-status'),
      images,
      videos,
    }
  })
}

async function experimentSnapshot(page) {
  return page.evaluate(() => {
    const metrics = document.querySelector('[data-smoke="experiment-material-metrics"]')
    const diagnostics = document.querySelector('[data-smoke="material-diagnostics-panel"]')
    const rows = Array.from(document.querySelectorAll('[data-smoke="material-diagnostics-row"]')).map(row => ({
      fileExists: row.getAttribute('data-file-exists'),
      urlAccessible: row.getAttribute('data-url-accessible'),
    }))
    const images = Array.from(document.querySelectorAll('img[data-smoke="experiment-formal-image"]')).map(img => ({
      src: img.currentSrc || img.src,
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
    }))
    const videos = Array.from(document.querySelectorAll('video[data-smoke="experiment-formal-video"]')).map(video => ({
      src: video.currentSrc || video.src,
      readyState: video.readyState,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
    }))
    return {
      total: Number(metrics?.getAttribute('data-total') || 0),
      pending: Number(metrics?.getAttribute('data-pending') || 0),
      clips: Number(metrics?.getAttribute('data-clips') || 0),
      diagnosticsCount: Number(diagnostics?.getAttribute('data-count') || 0),
      diagnosticsAccessible: Number(diagnostics?.getAttribute('data-url-accessible') || 0),
      rows,
      images,
      videos,
    }
  })
}

async function run() {
  const api = await playwrightRequest.newContext({
    baseURL: baseUrl,
    extraHTTPHeaders: { 'X-Operator-Role': operatorRole },
  })
  if (shouldReindex) await post(api, '/api/v1/materials/published/reindex')

  const published = await getJson(api, '/api/v1/materials/published?limit=500')
  const currentExperimentItems = (published.items || []).filter(item => item.experiment_id === experimentId)
  assertEqual(currentExperimentItems.length, expected.materialCount, 'workspace index current experiment material count')
  assertEqual(currentExperimentItems.filter(item => item.preview_url).length, expected.imageCount, 'workspace index preview_url count')
  assertEqual(currentExperimentItems.filter(item => item.clip_url).length, expected.videoCount, 'workspace index clip_url count')

  for (const [term, expectedHits] of semanticTerms) {
    const result = await getJson(api, `/api/v1/materials/published?limit=500&sort_by=relevance&text=${encodeURIComponent(term)}`)
    const hits = (result.items || []).filter(item => item.experiment_id === experimentId).length
    assertAtLeast(hits, expectedHits, `semantic search "${term}" current experiment hits`)
  }

  const browser = await chromium.launch()
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } })
  try {
    await page.goto(`${baseUrl}/materials?experiment_id=${encodeURIComponent(experimentId)}`)
    await page.waitForSelector('[data-smoke="workspace-material-grid"]', { timeout: 30_000 })
    await waitForLoadedMedia(page, 'img[data-smoke="workspace-formal-image"]', expected.imageCount, 'image')
    await waitForLoadedMedia(page, 'video[data-smoke="workspace-formal-video"]', expected.videoCount, 'video')
    const workspace = await workspaceSnapshot(page)
    assertEqual(workspace.filtered, expected.materialCount, 'workspace global page filtered material count')
    assertEqual(workspace.images.length, expected.imageCount, 'workspace global page image count')
    assertEqual(workspace.videos.length, expected.videoCount, 'workspace global page video count')

    for (const [term, expectedHits] of semanticTerms) {
      await page.fill('[data-smoke="workspace-material-search"]', term)
      await page.click('[data-smoke="workspace-material-search-submit"]')
      await page.waitForFunction(
        ({ wanted }) => Number(document.querySelector('[data-smoke="workspace-material-grid"]')?.getAttribute('data-filtered-count') || 0) >= wanted,
        { wanted: expectedHits },
        { timeout: 30_000 },
      )
    }

    await page.goto(`${baseUrl}/experiments/${encodeURIComponent(experimentId)}/materials`)
    await page.waitForSelector('[data-smoke="experiment-material-grid"]', { timeout: 30_000 })
    await waitForLoadedMedia(page, 'img[data-smoke="experiment-formal-image"]', expected.imageCount, 'image')
    await waitForLoadedMedia(page, 'video[data-smoke="experiment-formal-video"]', expected.videoCount, 'video')
    await page.waitForSelector('[data-smoke="material-diagnostics-panel"]', { timeout: 30_000 })
    const experiment = await experimentSnapshot(page)
    assertEqual(experiment.total, expected.materialCount, 'experiment MaterialSearch material count')
    assertEqual(experiment.pending, 0, 'experiment MaterialSearch pending count')
    assertEqual(experiment.images.length, expected.imageCount, 'experiment MaterialSearch formal image count')
    assertEqual(experiment.videos.length, expected.videoCount, 'experiment MaterialSearch formal video count')
    assertEqual(experiment.diagnosticsCount, expected.materialCount, 'diagnostics row count')
    assertEqual(experiment.diagnosticsAccessible, expected.materialCount, 'diagnostics URL accessible count')
    if (!experiment.rows.every(row => row.fileExists === 'true' && row.urlAccessible === 'true')) {
      fail('diagnostics contains inaccessible formal material', experiment.rows)
    }
  } catch (error) {
    await mkdir(path.join('output', 'playwright'), { recursive: true })
    await page.screenshot({ path: path.join('output', 'playwright', 'material-search-smoke-failure.png'), fullPage: true }).catch(() => undefined)
    throw error
  } finally {
    await browser.close()
    await api.dispose()
  }

  console.log(JSON.stringify({
    status: 'passed',
    baseUrl,
    experimentId,
    workspace_materials: expected.materialCount,
    formal_images: expected.imageCount,
    formal_videos: expected.videoCount,
    pending: 0,
    semantic_terms: semanticTerms.map(([term, hits]) => ({ term, expected_min_hits: hits })),
  }, null, 2))
}

run().catch(error => {
  console.error(error?.stack || error)
  process.exit(1)
})
