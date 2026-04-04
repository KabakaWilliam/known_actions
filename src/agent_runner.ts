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

  // Episode wall-clock start — all t_episode values are ms since this moment.
  const episodeStart = Date.now();
  const allDomEvents: object[] = [];
  let harvestCursor = 0;

  // -------------------------------------------------------------------------
  // Event-driven collection via exposeFunction.
  //
  // page_tracer.js calls window.__pushTraceEvent(event) for every recorded
  // interaction. Playwright routes the call over CDP to this handler — no
  // polling needed. harvestCursor stays in sync so the backstop harvest()
  // below only reads events that didn't make it through this bridge.
  // -------------------------------------------------------------------------
  await context.exposeFunction('__pushTraceEvent', (event: any) => {
    allDomEvents.push({ ...event, t_episode: Date.now() - episodeStart });
    harvestCursor++;
  });

  await context.addInitScript(injectorScript);

  const page = await context.newPage();
  await page.setViewportSize({ width: 1280, height: 768 });

  // -------------------------------------------------------------------------
  // Capture HTTP navigations from the Playwright side.
  //
  // MidScene navigates via standard href clicks → full page reloads.
  // These are invisible to page_tracer.js (which only patches pushState/popstate)
  // but Playwright's framenavigated fires on every HTTP navigation.
  //
  // Also reset harvestCursor: each new page gets a fresh __agentTrace.events[],
  // so the index must restart from 0.
  // -------------------------------------------------------------------------
  page.on('framenavigated', (frame: Parameters<Parameters<typeof page.on<'framenavigated'>>[1]>[0]) => {
    if (frame !== page.mainFrame()) return;
    const url = frame.url();
    if (url === 'about:blank') return;
    harvestCursor = 0;
    allDomEvents.push({
      type:      'navigate',
      t_episode: Date.now() - episodeStart,
      url,
      trigger:   'http',
    });
  });

  // -------------------------------------------------------------------------
  // Backstop harvest: reads any events that didn't arrive via exposeFunction.
  //
  // The main case is the beforeunload event: the CDP bridge call is in-flight
  // when the page tears down and may not complete. A single harvest() call
  // at episode end recovers those events from the in-page array.
  // -------------------------------------------------------------------------
  const harvest = async () => {
    try {
      const fresh = await page.evaluate((cursor: number) => {
        const trace = (window as any).__agentTrace;
        if (!trace) return [];
        return trace.events.slice(cursor);
      }, harvestCursor);
      const now = Date.now();
      allDomEvents.push(...(fresh as any[]).map(e => ({ ...e, t_episode: now - episodeStart })));
      harvestCursor += (fresh as any[]).length;
    } catch {
      // Page may be mid-navigation; skip silently.
    }
  };

  // --- Run the task ---
  await page.goto('https://en.wikipedia.org', { waitUntil: 'networkidle' });
  const agent = new PlaywrightAgent(page);

  let aiActError: string | null = null;
  try {
    await agent.aiAct(`
      You are a research agent. Use Wikipedia to answer this question:
      "${QUESTION}"

      Browse freely. Read whatever pages you judge relevant.
      When you have enough information, stop browsing.
      Do not create accounts, submit forms, or leave Wikipedia.
    `);
  } catch (err) {
    aiActError = String(err);
  }

  await harvest();  // backstop: catch any events that missed the push bridge

  // --- Extract the answer (skip if aiAct failed) ---
  let result: { answer: string; confidence: 'high' | 'medium' | 'low'; sources: string[] } | null = null;
  if (!aiActError) {
    result = await agent.aiQuery<{
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
  }

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
    error:        aiActError,
    midscene_log: midsceneLog,
    dom_trace:    domTrace,
  };

  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const outPath = path.join(OUTPUT_DIR, `${EPISODE_ID}.json`);
  fs.writeFileSync(outPath, JSON.stringify(episode, null, 2));

  if (aiActError) {
    console.error(`[ERROR] aiAct failed — partial trace saved → ${outPath}`);
    process.exit(1);
  }

  console.log(`[OK] Trace saved → ${outPath}`);
  console.log(`[ANSWER] ${JSON.stringify(result)}`);

} catch (err) {
  console.error('[ERROR]', err);
  process.exit(1);
} finally {
  await browser.close();
}
