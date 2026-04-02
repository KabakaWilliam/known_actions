import 'dotenv/config';
import { createAgent } from '../lib/agent';

async function main() {
  const agent = await createAgent();

  console.log('[task] Opening Chrome...');
  await agent.aiAct('open a terminal and run: google-chrome --no-sandbox https://google.com &');
  await agent.aiWaitFor('a Chrome browser window is visible', { timeoutMs: 15000 });

  console.log('[task] Browser is open, navigating...');
  await agent.aiAct('click the address bar and type "https://example.com" then press Enter');
  await agent.aiWaitFor('the page has loaded', { timeoutMs: 10000 });

  const pageInfo = await agent.aiQuery(
    '{ title: string, hasContent: boolean }, describe what is on the screen'
  );
  console.log('[task] Page info:', pageInfo);

  await agent.aiAssert('a webpage is visible with content');
  console.log('[task] Done!');
}

main().catch(console.error);
