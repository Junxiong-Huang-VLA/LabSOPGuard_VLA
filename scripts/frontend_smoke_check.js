const fs = require('fs');
const path = require('path');
let chromium;
try {
  chromium = require('playwright').chromium;
} catch (error) {
  chromium = require(path.join(process.cwd(), 'node_modules', 'playwright')).chromium;
}

const frontendUrl = (process.env.FRONTEND_URL || 'http://127.0.0.1:5173').replace(/\/$/, '');
const experimentId = process.env.EXPERIMENT_ID;
const outputPath = process.env.FRONTEND_SMOKE_OUTPUT || '';
const screenshotDir = process.env.FRONTEND_SMOKE_SCREENSHOTS || path.resolve(process.cwd(), '..', '..', '.codex_runtime', 'frontend-smoke');

if (!experimentId) {
  console.error('EXPERIMENT_ID is required.');
  process.exit(2);
}

fs.mkdirSync(screenshotDir, { recursive: true });

const pages = [
  ['experiments', '/experiments'],
  ['workspace', `/experiments/${experimentId}/workspace`],
  ['key-actions', `/experiments/${experimentId}/key-actions`],
  ['materials', `/experiments/${experimentId}/materials`],
  ['report', `/experiments/${experimentId}/report`],
];

function isBenignAbort(item) {
  if (item.reason !== 'net::ERR_ABORTED') return false;
  return /\.mp4(\?|$)/i.test(item.url) || item.url.endsWith('/video') || item.url.endsWith('/key-actions/results');
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  const consoleErrors = [];
  const pageErrors = [];
  const failedRequests = [];
  const badResponses = [];
  const keyframeResponses = [];

  page.on('console', msg => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', err => pageErrors.push(String(err.message || err)));
  page.on('requestfailed', req => {
    const failure = req.failure();
    failedRequests.push({ url: req.url(), reason: failure ? failure.errorText : 'unknown' });
  });
  page.on('response', res => {
    const status = res.status();
    const url = res.url();
    if (status >= 400) badResponses.push({ status, url });
    if (url.includes('/files/key_action_index/keyframes')) keyframeResponses.push({ status, url });
  });

  const pageSummaries = [];
  for (const [name, route] of pages) {
    await page.goto(`${frontendUrl}${route}`, { waitUntil: 'domcontentloaded', timeout: 45000 });
    await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => null);
    await page.waitForTimeout(name === 'key-actions' || name === 'report' ? 6000 : 3000);
    if (name === 'key-actions') {
      const advanced = await page.$$('button');
      const advancedButton = await Promise.all(
        advanced.map(async button => ((await button.textContent()) || '').includes('\u9ad8\u7ea7') ? button : null)
      ).then(items => items.find(Boolean));
      if (advancedButton) {
        await advancedButton.click();
        await page.waitForTimeout(1000);
      }
    }
    const fileRefs = await page.$$eval('img,video,a', elements => elements
      .map(element => element.currentSrc || element.src || element.href || '')
      .filter(url => /^file:/i.test(url)));
    const apiRefCount = await page.$$eval('img,video,a', elements => elements
      .map(element => element.currentSrc || element.src || element.href || '')
      .filter(url => url.includes('/api/v1/')).length);
    const screenshot = path.join(screenshotDir, `${name}.png`);
    await page.screenshot({ path: screenshot, fullPage: true });
    const textLength = await page.locator('body').innerText({ timeout: 5000 }).then(text => text.length).catch(() => 0);
    pageSummaries.push({ name, route, textLength, fileRefCount: fileRefs.length, apiRefCount, screenshot });
  }

  const apiCheck = await page.evaluate(async (experimentId) => {
    const resultsRes = await fetch(`/api/v1/experiments/${experimentId}/key-actions/results`, { cache: 'no-store' });
    const keyData = await resultsRes.json();
    const queryRes = await fetch(`/api/v1/experiments/${experimentId}/key-actions/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'balance weighing', top_k: 2, index_level: 'all' }),
    });
    const queryData = await queryRes.json();
    const pdfRes = await fetch(`/api/v1/experiments/${experimentId}/artifacts/professional_report_pdf`, { cache: 'no-store' });
    const pdfBytes = new Uint8Array(await pdfRes.arrayBuffer());
    const pdfHeader = Array.from(pdfBytes.slice(0, 5)).map(value => String.fromCharCode(value)).join('');
    const first = keyData.segments?.[0] || {};
    return {
      keyStatus: resultsRes.status,
      queryStatus: queryRes.status,
      querySource: queryData.validation_summary?.source,
      pdfStatus: pdfRes.status,
      pdfContentType: pdfRes.headers.get('content-type'),
      pdfHeader,
      segments: Array.isArray(keyData.segments) ? keyData.segments.length : null,
      microSegments: Array.isArray(keyData.micro_segments) ? keyData.micro_segments.length : null,
      vectorMetadata: Array.isArray(keyData.vector_metadata) ? keyData.vector_metadata.length : null,
      firstSegmentId: first.segment_id || null,
      firstStart: first.cv_detection?.start_sec ?? first.first_person?.local_start_sec ?? first.third_person?.local_start_sec ?? null,
      firstEnd: first.cv_detection?.end_sec ?? first.first_person?.local_end_sec ?? first.third_person?.local_end_sec ?? null,
      queryRows: Array.isArray(queryData.results) ? queryData.results.length : null,
      firstQueryId: queryData.results?.[0]?.micro_segment_id || queryData.results?.[0]?.segment_id || null,
    };
  }, experimentId);

  await browser.close();

  const relevantFailed = failedRequests.filter(item => !isBenignAbort(item));
  const relevantBadResponses = badResponses.filter(item => !item.url.includes('/favicon.ico'));
  const summary = {
    schema_version: 'frontend_smoke.v1',
    frontend_url: frontendUrl,
    experiment_id: experimentId,
    pages: pageSummaries,
    apiCheck,
    consoleErrors,
    pageErrors,
    failedRequests: relevantFailed,
    badResponses: relevantBadResponses,
    keyframeResponseCount: keyframeResponses.length,
    keyframeBadResponses: keyframeResponses.filter(item => item.status >= 400),
  };
  const keyActionPage = pageSummaries.find(item => item.name === 'key-actions') || {};
  const reportPage = pageSummaries.find(item => item.name === 'report') || {};

  if (outputPath) {
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, JSON.stringify(summary, null, 2), 'utf8');
  }
  console.log(JSON.stringify(summary, null, 2));

  const failed = (
    consoleErrors.length ||
    pageErrors.length ||
    relevantFailed.length ||
    relevantBadResponses.length ||
    pageSummaries.some(item => item.fileRefCount > 0) ||
    summary.keyframeBadResponses.length ||
    keyframeResponses.length < 1 ||
    (keyActionPage.textLength || 0) < 1000 ||
    (reportPage.textLength || 0) < 400 ||
    apiCheck.pdfHeader !== '%PDF-' ||
    apiCheck.queryRows < 1
  );
  if (failed) process.exitCode = 1;
})().catch(error => {
  console.error(error);
  process.exit(1);
});
