(function () {
  if (window.__agentTrace) return;  // double-injection guard

  window.__agentTrace = {
    episodeId: null,           // filled by agent_runner.ts after injection
    agentId:   null,
    events:    [],
    startTime: Date.now(),     // page-relative epoch (ms since page init)
  };

  const record = (type, data) => {
    if (!location.href.startsWith('http')) return;  // skip about:blank, data: URIs, etc.
    const event = {
      type,
      t:   Date.now() - window.__agentTrace.startTime,
      url: location.href,
      ...data,
    };
    window.__agentTrace.events.push(event);
    window.__pushTraceEvent?.(event);  // push directly to Node.js; backstop array kept for beforeunload edge case
  };

  // -------------------------------------------------------------------------
  // Event types — only what a website owner's own JS analytics could observe.
  // No site-specific selectors; no content that leaks agent intent.
  // -------------------------------------------------------------------------

  // Click — position, target element metadata, outbound href if present.
  document.addEventListener('click', (e) => {
    record('click', {
      x:            e.clientX,
      y:            e.clientY,
      target_tag:   e.target?.tagName,
      target_text:  e.target?.innerText?.slice(0, 100),
      target_id:    e.target?.id,
      target_class: e.target?.className?.slice?.(0, 80),
      href:         e.target?.closest('a')?.href ?? null,
    });
  }, true);

  // Keydown — structural keys only; printable input → 'char' placeholder.
  // A website owner's analytics would log the same (no content leakage).
  document.addEventListener('keydown', (e) => {
    record('keydown', {
      key:        e.key === 'Enter' ? 'Enter'
                  : e.key.length === 1 ? 'char'
                  : e.key,
      target_tag: e.target?.tagName,
      target_id:  e.target?.id,
    });
  }, true);

  // Scroll — depth and percentage into page (debounced 200ms).
  let _lastScrollTime = 0;
  document.addEventListener('scroll', () => {
    const now = Date.now();
    if (now - _lastScrollTime < 200) return;
    _lastScrollTime = now;
    record('scroll', {
      scrollY:   window.scrollY,
      docHeight: document.body.scrollHeight,
      pct:       Math.round(window.scrollY / document.body.scrollHeight * 100),
    });
  }, true);

  // Client-side navigation (SPA pushState / popstate).
  const _origPush = history.pushState.bind(history);
  history.pushState = (...args) => {
    _origPush(...args);
    record('navigate', { trigger: 'pushState', to: location.href });
  };
  window.addEventListener('popstate', () => {
    record('navigate', { trigger: 'popstate', to: location.href });
  });

  // Focus — fires when an input or textarea gains focus (e.g. the search box).
  // A website's own analytics can observe focus events; reveals interaction intent
  // without capturing typed content (that comes from keydown below).
  document.addEventListener('focus', (e) => {
    const tag = e.target?.tagName;
    if (tag !== 'INPUT' && tag !== 'TEXTAREA') return;
    record('focus', {
      target_tag:  tag,
      target_id:   e.target?.id,
      target_name: e.target?.name ?? null,
    });
  }, true);

  // Beforeunload — fires synchronously before a full HTTP navigation away.
  // Mirrors the sendBeacon pattern used by real-world analytics to capture
  // the last-recorded scroll depth before the page tears down.
  // The __pushTraceEvent CDP call may or may not complete before page teardown;
  // agent_runner.ts runs a backstop harvest() at episode end for the final page.
  window.addEventListener('beforeunload', () => {
    record('beforeunload', {
      scrollY:   window.scrollY,
      docHeight: document.body.scrollHeight,
      pct:       Math.round(window.scrollY / document.body.scrollHeight * 100),
      // The outbound href is whatever link was last clicked (no DOM access
      // after navigation starts), so we record the active element's href if any.
      leaving_href: document.activeElement?.closest('a')?.href ?? null,
    });
  });

  console.log('[AgentTracer] Injected on', location.origin);
})();
