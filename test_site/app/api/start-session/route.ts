import { NextRequest, NextResponse } from 'next/server';
import { validateRunnerSecret, signSessionCookie, buildSetCookieHeader } from '@/lib/auth';
import { createSession } from '@/lib/session';
import type { StartSessionBody } from '@/lib/types';

export async function POST(req: NextRequest) {
  // 1. Validate runner secret
  const secret = req.headers.get('x-runner-secret');
  if (!validateRunnerSecret(secret)) {
    return NextResponse.json({ ok: false, error: 'Unauthorized' }, { status: 401 });
  }

  // 2. Parse body
  let body: Partial<StartSessionBody>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: 'Invalid JSON' }, { status: 400 });
  }

  const { experiment_id, run_id, session_id, task_id } = body;
  if (!experiment_id || !run_id || !session_id || !task_id) {
    return NextResponse.json(
      { ok: false, error: 'Missing required fields: experiment_id, run_id, session_id, task_id' },
      { status: 400 },
    );
  }

  // 3. Create session record
  const userAgent = req.headers.get('user-agent') ?? '';
  await createSession({ experiment_id, run_id, session_id, task_id }, userAgent);

  // 4. Sign cookie
  const token = await signSessionCookie({ session_id, task_id });

  // 5. Return response with cookie
  return NextResponse.json(
    { ok: true, session_id },
    {
      status: 200,
      headers: { 'Set-Cookie': buildSetCookieHeader(token) },
    },
  );
}
