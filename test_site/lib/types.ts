// ─── Data model types ─────────────────────────────────────────────────────────

export type EventType =
  | 'page_load'
  | 'click'
  | 'scroll'
  | 'interaction'
  | 'page_exit';

/** A top-level experiment run, grouping multiple sessions */
export interface Run {
  run_id: string;
  experiment_id: string;
  created_at: string; // ISO 8601
  meta?: Record<string, unknown>;
}

/** One agent session within a run, assigned a specific task */
export interface Session {
  session_id: string;
  run_id: string;
  task_id: string;
  started_at: string; // ISO 8601
  ended_at?: string;  // ISO 8601
  user_agent?: string;
}

/** A single browser event recorded during a session */
export interface Event {
  event_id: string;       // UUID assigned on insert
  session_id: string;
  timestamp_ms: number;
  event_type: EventType;
  page: string;           // pathname e.g. "/facts/pricing"
  target?: string;        // e.g. element label or href
  referrer?: string;
  delta_ms?: number;      // ms since previous event
}

// ─── API payload types ────────────────────────────────────────────────────────

/** Body accepted by POST /api/start-session */
export interface StartSessionBody {
  experiment_id: string;
  run_id: string;
  session_id: string;
  task_id: string;
}

/** Body accepted by POST /api/log */
export interface ApiEvent {
  session_id: string;
  timestamp_ms: number;
  event_type: EventType;
  page: string;
  target?: string;
  referrer?: string;
  delta_ms?: number;
}

/** Claims stored in the signed exp_session JWT */
export interface SessionClaims {
  session_id: string;
  task_id: string;
}
