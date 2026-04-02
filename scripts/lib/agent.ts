import 'dotenv/config';
import { agentFromAiServiceBinding } from '@midscene/computer';

const MIDSCENE_SERVICE_URL = process.env.MIDSCENE_SERVICE_URL || 'http://localhost:3333';

export async function createAgent() {
  console.log(`[agent] Connecting to midscene-pc at ${MIDSCENE_SERVICE_URL}`);

  const agent = await agentFromAiServiceBinding({
    serviceUrl: MIDSCENE_SERVICE_URL,
  });

  console.log('[agent] Connected.');
  return agent;
}
