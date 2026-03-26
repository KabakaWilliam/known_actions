import { NextRequest, NextResponse } from 'next/server';
import { verifySessionCookie, SESSION_COOKIE, buildClearCookieHeader } from '@/lib/auth';
import { endSession } from '@/lib/session';

export async function POST(req: NextRequest) {
  const token = req.cookies.get(SESSION_COOKIE)?.value;
  if (!token) {
    return NextResponse.json({ ok: false, error: 'No session' }, { status: 401 });
  }

  const claims = await verifySessionCookie(token);
  if (!claims) {
    return NextResponse.json({ ok: false, error: 'Invalid session' }, { status: 401 });
  }

  await endSession(claims.session_id);

  return NextResponse.json(
    { ok: true },
    { headers: { 'Set-Cookie': buildClearCookieHeader() } },
  );
}
