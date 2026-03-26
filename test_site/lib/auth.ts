import { SignJWT, jwtVerify } from 'jose';
import type { SessionClaims } from './types';

// ─── Secrets (server-side only) ───────────────────────────────────────────────
const RUNNER_SECRET = process.env.RUNNER_SECRET ?? '';
const COOKIE_SECRET = process.env.COOKIE_SECRET ?? 'dev-cookie-secret-change-me!!';

export const SESSION_COOKIE = 'exp_session';

// Encode cookie secret as Uint8Array for jose
function cookieKey(): Uint8Array {
  return new TextEncoder().encode(COOKIE_SECRET);
}

// ─── Runner secret validation ─────────────────────────────────────────────────

export function validateRunnerSecret(headerValue: string | null): boolean {
  if (!RUNNER_SECRET) return false; // fail closed if secret not configured
  if (!headerValue) return false;
  return headerValue === RUNNER_SECRET;
}

// ─── Signed session cookie (HS256 JWT) ───────────────────────────────────────

export async function signSessionCookie(claims: SessionClaims): Promise<string> {
  return new SignJWT({ ...claims })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('24h')
    .sign(cookieKey());
}

export async function verifySessionCookie(token: string): Promise<SessionClaims | null> {
  try {
    const { payload } = await jwtVerify(token, cookieKey());
    if (
      typeof payload.session_id === 'string' &&
      typeof payload.task_id === 'string'
    ) {
      return { session_id: payload.session_id, task_id: payload.task_id };
    }
    return null;
  } catch {
    return null;
  }
}

// ─── Cookie header helpers ────────────────────────────────────────────────────

export function buildSetCookieHeader(token: string): string {
  return `${SESSION_COOKIE}=${token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=86400`;
}

export function buildClearCookieHeader(): string {
  return `${SESSION_COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0`;
}
