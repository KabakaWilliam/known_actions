import { chromium } from 'playwright';
import { PlaywrightAgent } from '@midscene/web/playwright';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import 'dotenv/config';

// --- Parse CLI args ---
const args = process.argv.slice(2);
let QUESTION    = '';
let AGENT_ID    = '';
let EPISODE_ID  = '';
let OUTPUT_DIR  = '';

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--question')    QUESTION   = args[++i];
  if (args[i] === '--agent_id')    AGENT_ID   = args[++i];
  if (args[i] === '--episode_id')  EPISODE_ID = args[++i];
  if (args[i] === '--output_dir')  OUTPUT_DIR = args[++i];
}

if (!QUESTION || !AGENT_ID || !EPISODE_ID || !OUTPUT_DIR) {
  console.error('[ERROR] Missing required arguments: --question, --agent_id, --episode_id, --output_dir');
  process.exit(1);
}

// --- Load page_tracer.js ---
const __dir = path.dirname(fileURLToPath(import.meta.url));
const injectorScript = fs.readFileSync(
  path.join(__dir, 'page_tracer.js'), 'utf-8'
);

// --- Main ---
const browser = await chromium.launch({
  headless: true,
  args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
}).catch((err) => {
  console.error('[ERROR] Failed to launch browser:', err);
  process.exit(1);
});

try {
  const context = await browser.newContext();
  await context.addInitScript(injectorScript);

  const page = await context.newPage();
  await page.setViewportSize({ width: 1280, height: 768 });

  // Episode wall-clock start — all t_episode values are ms since this moment.
  // This is the single timeline anchor across all pages in the session.
  const episodeStart = Date.now();

  // --- Inject episode metadata ---
  await page.goto('about:blank');
  await page.evaluate(({ episodeId, agentId }) => {
    if ((window as any).__agentTrace) {
      (window as any).__agentTrace.episodeId = episodeId;
      (window as any).__agentTrace.agentId   = agentId;
    }
  }, { episodeId: EPISODE_ID, agentId: AGENT_ID });

  // -------------------------------------------------------------------------
  // Fix 1 — Capture HTTP navigations from the Playwright side.
  //
  // MidScene navigates via standard href clicks → full page reloads.
  // These are invisible to page_tracer.js (which only patches pushState/popstate)
  // but Playwright's framenavigated fires on every HTTP navigation.
  //
  // What a website owner sees: every page request hits their server logs with
  // URL, referrer, and timestamp — exactly what we record here.
  // -------------------------------------------------------------------------
  page.on('framenavigated', (frame: Parameters<Parameters<typeof page.on<'framenavigated'>>[1]>[0]) => {
    if (frame !== page.mainFrame()) return;
    const url = frame.url();
    if (url === 'about:blank') return;
    allDomEvents.push({
      type:       'navigate',
      t_episode:  Date.now() - episodeStart,
      url,
      trigger:    'http',   // distinguishes full HTTP nav from pushState/popstate
    });
  });

  // -------------------------------------------------------------------------
  // Fix 2 — Harvest at 100ms instead of 500ms.
  //
  // Navigation-triggering clicks fire and the page loads in ~200–400ms.
  // At 500ms polling the click event was almost always lost in the gap.
  // At 100ms the window is narrow enough to capture it reliably.
  // -------------------------------------------------------------------------
  const allDomEvents: object[] = [];
  let harvestCursor = 0;
  let harvestUrl    = '';
  let pageEpochOffset = 0;  // (pageStartTimeWall - episodeStart) for current page

  const harvest = async () => {
    try {
      const currentUrl = page.url();
      if (currentUrl !== harvestUrl) {
        // Page navigated — reset cursor and recompute the epoch offset so
        // this page's events are anchored to the episode timeline (Fix 3).
        harvestCursor = 0;
        harvestUrl    = currentUrl;
        pageEpochOffset = await page.evaluate(() => {
          const trace = (window as any).__agentTrace;
          // startTimeWall is the browser wall-clock value at page init.
          // Subtract episodeStart (passed below) to get episode-relative offset.
          return trace?.startTimeWall ?? Date.now();
        }).then((startTimeWall: number) => startTimeWall - episodeStart).catch(() => 0);
      }

      const fresh = await page.evaluate((cursor: number) => {
        const trace = (window as any).__agentTrace;
        if (!trace) return [];
        return trace.events.slice(cursor);
      }, harvestCursor);

      // -----------------------------------------------------------------------
      // Fix 3 — Add t_episode: a monotonic, episode-relative timestamp.
      //
      // page_tracer.js sets t relative to each page's own startTime, so events
      // on page 2 start at t=0 again. t_episode = pageEpochOffset + t gives a
      // single coherent timeline across all pages in the session.
      //
      // This mirrors what a website owner sees: their server logs timestamps are
      // absolute, and time-on-page is derived from consecutive request times.
      // -----------------------------------------------------------------------
      const anchored = (fresh as any[]).map(e => ({
        ...e,
        t_episode: pageEpochOffset + e.t,
      }));

      allDomEvents.push(...anchored);
      harvestCursor += fresh.length;
    } catch {
      // Page may be mid-navigation; skip this tick silently.
    }
  };

  const harvestInterval = setInterval(harvest, 100);

  // --- Run the task ---
  await page.goto('https://en.wikipedia.org', { waitUntil: 'networkidle' });
  const agent = new PlaywrightAgent(page);

  await agent.aiAct(`
    You are a research agent. Use Wikipedia to answer this question:
    "${QUESTION}"

    Browse freely. Read whatever pages you judge relevant.
    When you have enough information, stop browsing.
    Do not create accounts, submit forms, or leave Wikipedia.
  `);

  clearInterval(harvestInterval);
  await harvest();  // final drain of the last page's events

  // --- Extract the answer ---
  const result = await agent.aiQuery<{
    answer:     string;
    confidence: 'high' | 'medium' | 'low';
    sources:    string[];
  }>(
    `Based on what you have read, answer: "${QUESTION}"
     Return: {
       answer:     string,    // the answer
       confidence: string,    // "high" | "medium" | "low"
       sources:    string[]   // Wikipedia article titles you consulted
     }`
  );

  // --- Collect MidScene log ---
  const midsceneLog = (agent as any)._unstableLogContent?.() ?? [];

  // --- Assemble and write the episode JSON ---
  const domTrace = {
    episodeId:      EPISODE_ID,
    agentId:        AGENT_ID,
    episodeDuration: Date.now() - episodeStart,
    events:         allDomEvents,
    pageCount:      new Set(
      allDomEvents
        .filter((e: any) => e.type === 'navigate' && e.trigger === 'http')
        .map((e: any) => e.url)
    ).size,
  };

  const episode = {
    meta: {
      episode_id:   EPISODE_ID,
      agent_id:     AGENT_ID,
      model_name:   process.env.MIDSCENE_MODEL_NAME,
      model_family: process.env.MIDSCENE_MODEL_FAMILY,
      timestamp:    new Date().toISOString(),
      question:     QUESTION,
    },
    result,
    midscene_log: midsceneLog,
    dom_trace:    domTrace,
  };

  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const outPath = path.join(OUTPUT_DIR, `${EPISODE_ID}.json`);
  fs.writeFileSync(outPath, JSON.stringify(episode, null, 2));

  console.log(`[OK] Trace saved → ${outPath}`);
  console.log(`[ANSWER] ${JSON.stringify(result)}`);

} catch (err) {
  console.error('[ERROR]', err);
  process.exit(1);
} finally {
  await browser.close();
}
