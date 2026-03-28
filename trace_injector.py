"""Trace injection and collection for Playwright pages.

Trace Event Type Schema:
-----------------------
All events have these base fields:
  - timestamp (float): Unix timestamp in milliseconds
  - type (str): Event category
  - url (str): URL where event occurred
  - relativeTime (int, optional): ms since trace started
  - latency (int, optional): ms since last event

Event Types:
  - navigate: {url, reason} - Page navigation
  - keypress: {key, code, ctrlKey, shiftKey, altKey, metaKey} - User keyboard input
  - click: {x, y, button, target, targetId, targetClass} - User mouse click
  - scroll: {scrollX, scrollY, scrollWidth, scrollHeight} - User scroll
  - input: {target, value} - Form input change
  - screenshot: {phase?, iteration?} - Screenshot captured
  - action_start: {actionSequence, numActionsInSequence} - Start of model's action sequence
  - action_end: {actionSequence, success} - End of model's action sequence
  - response_received: {actionSequence} - Model response received from API
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlparse


def _extract_origin(url: str) -> Optional[str]:
    """Safely extract origin (netloc) from URL.
    
    Args:
        url: URL string
        
    Returns:
        Origin string (e.g., 'www.wikipedia.org') or None if parse fails
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc or None
    except Exception:
        return None


TRACE_COLLECTOR_JS = """
(function() {
  // Initialize trace storage with per-origin tracking
  window.__trace = {
    events: [],
    currentOrigin: window.location.origin,
    originStartTime: Date.now(),
    lastEventTime: Date.now(),
  };

  // Helper to record event with per-origin relativeTime
  function recordEvent(eventType, details = {}) {
    const now = Date.now();
    const event = {
      timestamp: now,
      relativeTime: now - window.__trace.originStartTime,
      latency: now - window.__trace.lastEventTime,
      type: eventType,
      url: window.location.href,
      origin: window.location.origin,
      ...details,
    };
    window.__trace.events.push(event);
    window.__trace.lastEventTime = now;
  }

  // Track origin changes
  function onOriginChange() {
    const newOrigin = window.location.origin;
    if (newOrigin !== window.__trace.currentOrigin) {
      window.__trace.currentOrigin = newOrigin;
      window.__trace.originStartTime = Date.now();
      window.__trace.lastEventTime = Date.now();
    }
  }

  // Capture keydown events
  document.addEventListener('keydown', (e) => {
    onOriginChange();
    recordEvent('keydown', {
      key: e.key,
      code: e.code,
      ctrlKey: e.ctrlKey,
      shiftKey: e.shiftKey,
      altKey: e.altKey,
      metaKey: e.metaKey,
    });
  });

  // Capture keypress events
  document.addEventListener('keypress', (e) => {
    onOriginChange();
    recordEvent('keypress', {
      key: e.key,
      code: e.code,
    });
  });

  // Capture click events
  document.addEventListener('click', (e) => {
    onOriginChange();
    recordEvent('click', {
      x: e.clientX,
      y: e.clientY,
      button: e.button,
      target: e.target.tagName,
      targetId: e.target.id || null,
      targetClass: e.target.className || null,
    });
  }, true);

  // Capture double-click events
  document.addEventListener('dblclick', (e) => {
    onOriginChange();
    recordEvent('dblclick', {
      x: e.clientX,
      y: e.clientY,
      target: e.target.tagName,
    });
  }, true);

  // Capture mousemove events (throttled to every 100ms)
  let lastMouseMoveTime = 0;
  document.addEventListener('mousemove', (e) => {
    const now = Date.now();
    if (now - lastMouseMoveTime > 100) {
      onOriginChange();
      recordEvent('mousemove', {
        x: e.clientX,
        y: e.clientY,
      });
      lastMouseMoveTime = now;
    }
  });

  // Capture scroll events
  window.addEventListener('scroll', (e) => {
    onOriginChange();
    recordEvent('scroll', {
      scrollX: window.scrollX,
      scrollY: window.scrollY,
      scrollWidth: document.documentElement.scrollWidth,
      scrollHeight: document.documentElement.scrollHeight,
    });
  });

  // Capture input/change events
  document.addEventListener('input', (e) => {
    onOriginChange();
    recordEvent('input', {
      target: e.target.tagName,
      targetType: e.target.type,
      value: e.target.value?.substring(0, 100) || '',  // Limit to 100 chars
    });
  });

  // Capture focus events
  document.addEventListener('focus', (e) => {
    onOriginChange();
    recordEvent('focus', {
      target: e.target.tagName,
      targetType: e.target.type,
    });
  }, true);

  // Persist trace to sessionStorage before navigation
  window.addEventListener('beforeunload', (e) => {
    try {
      const stored = JSON.parse(sessionStorage.getItem('__trace_backup') || '[]');
      sessionStorage.setItem('__trace_backup', JSON.stringify([...stored, ...window.__trace.events]));
    } catch (err) {
      console.error('[TraceCollector] Failed to persist trace:', err);
    }
  });

  // Expose function to retrieve traces (includes backed-up traces)
  window.getTrace = function() {
    let events = window.__trace.events || [];
    try {
      const backedUp = JSON.parse(sessionStorage.getItem('__trace_backup') || '[]');
      events = [...backedUp, ...events];
      // Don't clear backup - let Python side manage it
    } catch (err) {
      console.error('[TraceCollector] Failed to restore trace:', err);
    }
    return events;
  };

  // Expose function to clear traces
  window.clearTrace = function() {
    window.__trace.events = [];
    window.__trace.originStartTime = Date.now();
    window.__trace.lastEventTime = Date.now();
    try {
      sessionStorage.setItem('__trace_backup', '[]');
    } catch (err) {}
  };

  // Expose function to get current origin
  window.getTraceOrigin = function() {
    return window.__trace.currentOrigin || window.location.origin;
  };

  console.log('[TraceCollector] Initialized');
})();
"""


class TraceCollector:
    """Manages trace collection for a Playwright page with client-observable events only."""

    def __init__(self, artifacts_dir: Path):
        """Initialize trace collector.
        
        Args:
            artifacts_dir: Directory to store trace.jsonl
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.trace_file = self.artifacts_dir / "trace.jsonl"
        self.events = []
        self.page = None
        self.current_origin = None
        self.origin_start_time = None

    def inject_trace_collector(self, page) -> None:
        """Inject trace collector JavaScript into page and setup navigation handler.
        
        Args:
            page: Playwright page object
        """
        self.page = page
        self.current_origin = _extract_origin(page.url)
        # Don't set origin_start_time here - let first event set it for accurate relative timing
        self.origin_start_time = None
        
        page.evaluate(TRACE_COLLECTOR_JS)
        
        # Re-inject trace collector after navigation
        page.on("framenavigated", lambda frame: self._on_frame_navigated(frame))

    def _on_frame_navigated(self, frame) -> None:
        """Handle frame navigation by re-injecting trace collector.
        
        Args:
            frame: Playwright frame object
        """
        # Only inject in the main frame
        if frame and self.page and frame == self.page.main_frame:
            try:
                self.page.evaluate(TRACE_COLLECTOR_JS)
                # Update origin tracking
                new_origin = _extract_origin(self.page.url)
                if new_origin and new_origin != self.current_origin:
                    self.current_origin = new_origin
                    self.origin_start_time = datetime.now().timestamp() * 1000
            except Exception as e:
                # Silently ignore errors during re-injection
                pass

    def measure_page_load(self, page) -> float:
        """Measure page load timing using Performance API.
        
        Args:
            page: Playwright page object
            
        Returns:
            Page load time in milliseconds, or 0 if measurement fails
        """
        try:
            page_load_ms = page.evaluate("""
                () => {
                  const perfData = window.performance.timing;
                  if (perfData.loadEventEnd && perfData.navigationStart) {
                    return perfData.loadEventEnd - perfData.navigationStart;
                  }
                  // Fallback: check if document is interactive
                  if (perfData.domInteractive && perfData.navigationStart) {
                    return perfData.domInteractive - perfData.navigationStart;
                  }
                  return 0;
                }
            """)
            return float(page_load_ms)
        except Exception:
            return 0.0

    def add_navigation_event(self, page, url: str, reason: str = "navigate", prev_origin: Optional[str] = None, page_load_time: Optional[float] = None) -> None:
        """Record a navigation event (client-observable).
        
        Args:
            page: Playwright page object for URL extraction
            url: Target URL
            reason: Navigation reason (e.g., "initial", "navigate", "action")
            prev_origin: Previous origin (for cross-origin navigation tracking)
            page_load_time: Page load duration in ms (optional, will be measured if not provided)
        """
        timestamp = datetime.now().timestamp() * 1000
        origin = _extract_origin(url)  # Extract domain from URL safely
        if not origin:
            return  # Skip navigation events without valid origin
        
        # Check if this is a cross-origin navigation
        is_cross_origin = prev_origin is not None and prev_origin != origin
        
        # Measure page load time if not provided
        if page_load_time is None:
            page_load_time = self.measure_page_load(page)
        
        # Create per-origin relative time
        if is_cross_origin:
            relative_time = 0.0  # Reset relativeTime on origin change
            self.origin_start_time = timestamp
        else:
            relative_time = timestamp - (self.origin_start_time or timestamp)
        
        event = {
            "timestamp": timestamp,
            "type": "navigate",
            "relativeTime": relative_time,
            "url": url,
            "origin": origin,
            "reason": reason,
        }
        
        if page_load_time and page_load_time > 0:
            event["pageLoadTime"] = page_load_time
        
        if is_cross_origin:
            event["prevOrigin"] = prev_origin
        
        self.events.append(event)
        self.current_origin = origin


    def _wait_for_trace_function(self, page, timeout: float = 5.0) -> bool:
        """Wait for window.getTrace to be available (indicating JS injection is ready).
        
        Args:
            page: Playwright page object
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if function becomes available, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if getTrace function exists
                exists = page.evaluate("typeof window.getTrace === 'function'")
                if exists:
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def collect_page_trace(self, page, retry_count: int = 3, retry_delay: float = 0.2) -> list[dict[str, Any]]:
        """Retrieve trace events from page with retry logic.
        
        Args:
            page: Playwright page object
            retry_count: Number of retries on context destruction
            retry_delay: Delay between retries in seconds
            
        Returns:
            List of trace events captured by the page (normalized with proper relativeTime)
        """
        # First, ensure the JavaScript context is ready
        if not self._wait_for_trace_function(page, timeout=3.0):
            # Re-inject if function isn't available
            try:
                page.evaluate(TRACE_COLLECTOR_JS)
                if not self._wait_for_trace_function(page, timeout=2.0):
                    import sys
                    print("[TraceCollector] ERROR: window.getTrace still not available after re-injection", file=sys.stderr)
                    return []
            except Exception as e:
                import sys
                print(f"[TraceCollector] ERROR: Failed to re-inject JavaScript: {e}", file=sys.stderr)
                return []
        
        for attempt in range(retry_count):
            try:
                trace_events = page.evaluate("window.getTrace()")
                if trace_events is None:
                    return []
                
                if not trace_events:
                    return []
                
                # Find the earliest timestamp to use as origin baseline
                min_timestamp = min(e.get('timestamp', float('inf')) for e in trace_events)
                
                # Normalize all events: relativeTime should be relative to earliest event of that origin
                normalized_events = []
                for event in trace_events:
                    event_origin = event.get('origin')
                    # For same-origin events, reset baseline to min timestamp
                    if event_origin == self.current_origin or self.current_origin is None:
                        event['relativeTime'] = event['timestamp'] - min_timestamp
                    normalized_events.append(event)
                
                # Update origin_start_time to the earliest event we collected
                if not self.origin_start_time:
                    self.origin_start_time = min_timestamp
                    
                return normalized_events
            except Exception as e:
                error_msg = str(e)
                # If context was destroyed, wait a bit and retry
                if "Execution context was destroyed" in error_msg:
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        # Re-inject on context destruction
                        try:
                            page.evaluate(TRACE_COLLECTOR_JS)
                            self._wait_for_trace_function(page, timeout=1.0)
                        except Exception:
                            pass
                        continue
                # For ANY error, log it but try to recover gracefully
                import sys
                print(f"[TraceCollector] Warning: Failed to collect trace (attempt {attempt+1}/{retry_count}): {error_msg}", file=sys.stderr)
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
                    continue
        print("[TraceCollector] Warning: Could not collect trace events after retries", file=sys.stderr)
        return []

    def save_trace(self) -> None:
        """Save collected events to trace.jsonl file (JSONL format, one event per line)."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.trace_file, "w", encoding="utf-8") as f:
            for event in self.events:
                # Ensure all floats in timestamps are properly serialized
                f.write(json.dumps(event) + "\n")

    def clear_page_trace(self, page) -> None:
        """Clear trace events on the page.
        
        Args:
            page: Playwright page object
        """
        try:
            page.evaluate("window.clearTrace()")
        except Exception:
            # Silently ignore errors when clearing
            pass

    def add_page_events_to_trace(self, page) -> None:
        """Collect page trace events and add them to the trace.
        
        Args:
            page: Playwright page object
        """
        page_events = self.collect_page_trace(page)
        self.events.extend(page_events)

