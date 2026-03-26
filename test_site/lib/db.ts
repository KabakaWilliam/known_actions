import type { Session, Event } from './types';

// ─── Database interface ───────────────────────────────────────────────────────
// Swap InMemoryDb for a real implementation (e.g. SupabaseDb) without changing
// call-sites. All methods are async so the interface is Postgres-compatible.

export interface Db {
  createSession(session: Session): Promise<void>;
  endSession(session_id: string, ended_at: string): Promise<void>;
  insertEvent(event: Event): Promise<void>;
  getSession(session_id: string): Promise<Session | null>;
}

// ─── In-memory stub ───────────────────────────────────────────────────────────
// TODO: Replace with SupabaseDb that calls supabase.from('sessions').insert(...)
//       and supabase.from('events').insert(...) once a Postgres backend is ready.

class InMemoryDb implements Db {
  private sessions = new Map<string, Session>();
  private events: Event[] = [];

  async createSession(session: Session): Promise<void> {
    // TODO: INSERT INTO sessions (...) VALUES (...)
    this.sessions.set(session.session_id, session);
  }

  async endSession(session_id: string, ended_at: string): Promise<void> {
    // TODO: UPDATE sessions SET ended_at = $1 WHERE session_id = $2
    const s = this.sessions.get(session_id);
    if (s) this.sessions.set(session_id, { ...s, ended_at });
  }

  async insertEvent(event: Event): Promise<void> {
    // TODO: INSERT INTO events (...) VALUES (...)
    this.events.push(event);
  }

  async getSession(session_id: string): Promise<Session | null> {
    // TODO: SELECT * FROM sessions WHERE session_id = $1 LIMIT 1
    return this.sessions.get(session_id) ?? null;
  }
}

// Singleton — survives hot-reload in dev via module cache
const globalForDb = globalThis as typeof globalThis & { __db?: Db };
export const db: Db = globalForDb.__db ?? (globalForDb.__db = new InMemoryDb());
