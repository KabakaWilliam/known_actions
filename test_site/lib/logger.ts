// Client-side event logger — browser only.
// Call initLogger() once per page, then use logEvent / helpers freely.

import type { EventType, ApiEvent } from './types';

let _sessionId: string | null = null;
let _lastEventTs: number | null = null;

/** Call once on mount when a valid exp_session cookie is present. */
export function initLogger(sessionId: string): void {
  _sessionId = sessionId;
  _lastEventTs = Date.now();

  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', () => {
      logEvent('page_exit');
    });
  }
}

/** Fire-and-forget POST to /api/log. Silently drops on failure. */
export function logEvent(
  type: EventType,
  opts: {
    target?: string;
    referrer?: string;
  } = {},
): void {
  if (!_sessionId) return;

  const now = Date.now();
  const delta_ms = _lastEventTs !== null ? now - _lastEventTs : undefined;
  _lastEventTs = now;

  const body: ApiEvent = {
    session_id: _sessionId,
    timestamp_ms: now,
    event_type: type,
    page: typeof window !== 'undefined' ? window.location.pathname : '',
    ...(opts.target !== undefined && { target: opts.target }),
    ...(opts.referrer !== undefined && { referrer: opts.referrer }),
    ...(delta_ms !== undefined && { delta_ms }),
  };

  // Use sendBeacon for page_exit to survive unload; fetch otherwise
  if (type === 'page_exit' && typeof navigator !== 'undefined' && navigator.sendBeacon) {
    navigator.sendBeacon('/api/log', JSON.stringify(body));
  } else {
    fetch('/api/log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      keepalive: true,
    }).catch(() => {/* silently ignore */});
  }
}

export function logClick(target: string): void {
  logEvent('click', { target });
}

export function logScroll(): void {
  logEvent('scroll');
}

export function logInteraction(target: string): void {
  logEvent('interaction', { target });
}
