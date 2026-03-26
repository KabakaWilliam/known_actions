'use client';

import { useEffect, useRef } from 'react';
import { usePathname } from 'next/navigation';
import { initLogger, logEvent, logScroll } from '@/lib/logger';

interface EventLoggerProps {
  /** The session_id decoded server-side from the exp_session cookie.
   *  Null if no valid session is active — logger will be a no-op. */
  sessionId: string | null;
}

export default function EventLogger({ sessionId }: EventLoggerProps) {
  const pathname = usePathname();
  const initialized = useRef(false);

  useEffect(() => {
    if (!sessionId) return;

    if (!initialized.current) {
      initLogger(sessionId);
      initialized.current = true;
    }

    // Log page_load on each navigation
    logEvent('page_load', { referrer: document.referrer || undefined });
  }, [pathname, sessionId]);

  useEffect(() => {
    if (!sessionId) return;

    let ticking = false;
    function onScroll() {
      if (!ticking) {
        requestAnimationFrame(() => {
          logScroll();
          ticking = false;
        });
        ticking = true;
      }
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, [sessionId]);

  return null; // no UI
}
