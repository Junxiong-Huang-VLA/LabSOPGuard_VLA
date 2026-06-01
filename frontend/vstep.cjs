const { chromium } = require('playwright');
const BASE = process.argv[2] || 'http://127.0.0.1:8000';
const EID = 'benchmark-weighing-pipetting-2026-05-22-fastlocate';
const URL = `${BASE}/experiments/${EID}/materials`;
(async () => {
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 2600 }, deviceScaleFactor: 1 })).newPage();
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.waitForSelector('[data-smoke="key-material-pair"]', { timeout: 20000 }).catch(()=>console.log('WARN no card'));
  await page.waitForTimeout(3000);
  const cards = await page.locator('[data-smoke="key-material-pair"]').count();
  console.log('CARD_COUNT', cards);
  // For each video: is the poster ACTUALLY painted? check computed + naturalWidth of poster img + bounding box
  const vis = await page.evaluate(() => {
    const out = [];
    document.querySelectorAll('video').forEach((el, i) => {
      const r = el.getBoundingClientRect();
      const cs = getComputedStyle(el);
      out.push({
        i,
        poster: (el.getAttribute('poster')||'').split('/').slice(-3).join('/'),
        boxW: Math.round(r.width), boxH: Math.round(r.height),
        visible: r.width>0 && r.height>0 && cs.display!=='none' && cs.visibility!=='hidden' && cs.opacity!=='0',
        videoW: el.videoWidth, videoH: el.videoHeight, readyState: el.readyState,
      });
    });
    return out;
  });
  console.log('VIDEO_VIS', JSON.stringify(vis,null,1));
  // screenshot first card close-up
  const first = page.locator('[data-smoke="key-material-pair"]').first();
  await first.scrollIntoViewIfNeeded();
  await page.waitForTimeout(800);
  await first.screenshot({ path: 'card_closeup.png' });
  await page.screenshot({ path: 'materials_full2.png', fullPage: true });
  console.log('SHOT card_closeup.png materials_full2.png');
  await browser.close();
})().catch(e=>{console.error('ERR',e);process.exit(1);});
