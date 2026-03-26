import { db } from './db';
import type { Session, StartSessionBody } from './types';

export async function createSession(
  body: StartSessionBody,
  userAgent: string,
): Promise<Session> {
  const session: Session = {
    session_id: body.session_id,
    run_id: body.run_id,
    task_id: body.task_id,
    started_at: new Date().toISOString(),
    user_agent: userAgent,
  };
  await db.createSession(session);
  return session;
}

export async function endSession(session_id: string): Promise<void> {
  await db.endSession(session_id, new Date().toISOString());
}
