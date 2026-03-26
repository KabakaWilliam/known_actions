import { NextRequest, NextResponse } from 'next/server';
import { verifySessionCookie, SESSION_COOKIE } from '@/lib/auth';
import { db } from '@/lib/db';
import type { ApiEvent, Event } from '@/lib/types';
import { randomUUID } from 'crypto';

export async function POST(req: NextRequest) {
  // 1. Validate session cookie — anonymous traffic is never logged as experiment data
  const token = req.cookies.get(SESSION_COOKIE)?.value;
  if (!token) {
    return NextResponse.json({ ok: false, error: 'No session' }, { status: 401 });
  }

  const claims = await verifySessionCookie(token);
  if (!claims) {
    return NextResponse.json({ ok: false, error: 'Invalid session' }, { status: 401 });
  }

  // 2. Parse event body
  let body: Partial<ApiEvent>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: 'Invalid JSON' }, { status: 400 });
  }

  const { session_id, timestamp_ms, event_type, page, target, referrer, delta_ms } = body;

  // Ensure the cookie's session_id matches the event's session_id
  if (!session_id || !timestamp_ms || !event_type || !page) {
    return NextResponse.json({ ok: false, error: 'Missing required event fields' }, { status: 400 });
  }
  if (session_id !== claims.session_id) {
    return NextResponse.json({ ok: false, error: 'Session mismatch' }, { status: 403 });
  }

  // 3. Insert event
  const event: Event = {
    event_id: randomUUID(),
    session_id,
    timestamp_ms,
    event_type,
    page,
    ...(target !== undefined && { target }),
    ...(referrer !== undefined && { referrer }),
    ...(delta_ms !== undefined && { delta_ms }),
  };

  await db.insertEvent(event);

  return NextResponse.json({ ok: true });
}
