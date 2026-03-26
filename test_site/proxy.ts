import { NextRequest, NextResponse } from "next/server";
import { verifySessionCookie, SESSION_COOKIE } from "@/lib/auth";

export async function proxy(req: NextRequest) {
  const token = req.cookies.get(SESSION_COOKIE)?.value ?? null;

  if (token) {
    const claims = await verifySessionCookie(token);
    if (claims) {
      // Valid experiment session — log the request server-side
      // TODO: Replace console.log with a lightweight DB write (e.g. request_log table)
      console.log("[exp]", {
        method: req.method,
        path: req.nextUrl.pathname,
        session_id: claims.session_id,
        task_id: claims.task_id,
        ts: Date.now(),
      });
    }
  }

  // All routes remain publicly accessible — experimental logging is additive only
  return NextResponse.next();
}

export const config = {
  // Run on all routes except static assets and Next.js internals
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
