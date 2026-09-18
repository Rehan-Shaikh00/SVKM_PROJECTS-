/**
 * Typed client for the NMIMS Dhule voice-assistant API.
 *
 * Everything is same-origin: in production FastAPI serves this bundle from
 * backend/static/dashboard, and in `npm run dev` Vite proxies /api to :8000.
 * Never hard-code a host — the browser must not reach for localhost.
 */

export interface Citation {
  record_id: string
  chunk_id: string
  title: string
  category: string
  score: number
  verified: boolean
  academic_year: string | null
  source: string | null
  citation: string
}

export interface FollowUp {
  channel?: string
  title?: string
  items?: string[]
}

export interface QueryResponse {
  question: string
  language: string
  answer: string
  grounded: boolean
  confidence: number
  intent: string
  needs_escalation: boolean
  escalation_reason: string | null
  citations: Citation[]
  followup: FollowUp | null
  provider: string
  fallback_used: boolean
  latency_ms: number
  retrieval_ms: number
  llm_ms: number
  warnings: string[]
}

export interface KBRecord {
  id: string
  slug: string
  category: string
  subcategory: string | null
  title: string
  body: string
  structured: Record<string, unknown>
  language: string
  tags: string[]
  aliases: string[]
  academic_year: string | null
  source: string | null
  source_uri: string | null
  verified: boolean
  verified_by: string | null
  verified_at: string | null
  status: string
  revision: number
  stale: boolean
  created_at: string | null
  updated_at: string | null
  chunk_count: number | null
}

export interface KBRecordDetail extends KBRecord {
  chunks: {
    id: string
    position: number
    text: string
    language: string
    tokens: number
    /** present on the chunking-preview endpoint, absent on stored chunks */
    characters?: number
    kind?: string
  }[]
  revisions: { revision: number; changed_by: string | null; change_note: string | null; created_at: string | null }[]
}

export interface RecordList {
  total: number
  limit: number
  offset: number
  items: KBRecord[]
}

export interface KBStats {
  records: {
    records: number
    chunks: number
    verified: number
    unverified: number
    /** verified from the university website at ingest, never signed off by a person */
    awaiting_signoff: number
    stale: number
    by_category: Record<string, number>
    by_status: Record<string, number>
  }
  index: { chunks: number; records: number; embedding: string; dimensions: number; min_score: number; top_k: number }
  seed_dir: string
  google_sheet_configured: boolean
  sync_interval_minutes: number
  academic_year: string
  staleness_days: number
}

export interface CallSummary {
  id: string
  call_id: string
  provider: string
  status: string
  from_number: string | null
  to_number: string | null
  started_at: string | null
  ended_at: string | null
  duration_seconds: number
  detected_language: string | null
  language_confidence: number | null
  primary_intent: string | null
  turn_count: number
  escalated: boolean
  resolved: boolean
  end_reason: string | null
  barge_in_count: number
  avg_response_latency_ms: number | null
  time_to_first_audio_ms: number | null
  satisfaction: number | null
  recording_consent: boolean | null
}

async function unwrap<T = Record<string, any>>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export const api = {
  config: () => http<Record<string, any>>('/api/config'),
  status: () => http<Record<string, any>>('/api/status'),
  health: () => http<Record<string, any>>('/health'),
  providers: () => http<Record<string, any>>('/health/providers'),
  simulatorConfig: () => http<Record<string, any>>('/api/simulator/config'),

  query: (question: string, language = 'en-IN') =>
    http<QueryResponse>('/api/assistant/query', {
      method: 'POST',
      body: JSON.stringify({ question, language }),
    }),
  languages: () => http<Record<string, any>>('/api/assistant/languages'),

  kbStats: () => http<KBStats>('/api/kb/stats'),
  kbCategories: () => http<Record<string, any>>('/api/kb/categories'),
  kbRecords: (params: Record<string, string | number | boolean | undefined>) => {
    const qs = new URLSearchParams()
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== '') qs.set(k, String(v))
    return http<RecordList>(`/api/kb/records?${qs.toString()}`)
  },
  kbRecord: (id: string) => http<KBRecordDetail>(`/api/kb/records/${id}`),
  kbCreate: (body: Record<string, unknown>) =>
    http<{ record: KBRecord }>('/api/kb/records', { method: 'POST', body: JSON.stringify(body) }),
  kbUpdate: (id: string, body: Record<string, unknown>) =>
    http<{ record: KBRecord }>(`/api/kb/records/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
  kbDelete: (id: string) => http<{ deleted: string }>(`/api/kb/records/${id}`, { method: 'DELETE' }),
  /** `changeNote` is what the sign-off is kept against in the audit trail. */
  kbVerify: (id: string, verifiedBy?: string, changeNote?: string) =>
    http<{ record: KBRecord }>(`/api/kb/records/${id}/verify`, {
      method: 'POST',
      body: JSON.stringify({ verified_by: verifiedBy ?? '', change_note: changeNote ?? '' }),
    }),
  kbBulkVerify: (ids: string[], verifiedBy?: string, changeNote?: string) =>
    http<Record<string, any>>('/api/kb/bulk-verify', {
      method: 'POST',
      body: JSON.stringify({
        record_ids: ids,
        verified_by: verifiedBy ?? '',
        change_note: changeNote ?? '',
      }),
    }),
  kbReindex: () => http<Record<string, any>>('/api/kb/reindex', { method: 'POST' }),
  /** These two take form-encoded bodies, not JSON. */
  kbSyncSheet: (url = '') => {
    const fd = new FormData()
    fd.set('url', url)
    return fetch('/api/kb/sync-sheet', { method: 'POST', body: fd }).then((r) => unwrap(r))
  },
  kbPurge: () => {
    const fd = new FormData()
    fd.set('confirm', 'PURGE')
    return fetch('/api/kb/purge', { method: 'POST', body: fd }).then((r) => unwrap(r))
  },
  kbImport: (file: File | null, text: string, dryRun: boolean) => {
    const fd = new FormData()
    if (file) fd.append('file', file)
    if (text) fd.append('text', text)
    fd.append('source', 'dashboard-import')
    fd.append('dry_run', dryRun ? 'true' : 'false')
    return fetch('/api/kb/import', { method: 'POST', body: fd }).then((r) => unwrap(r))
  },
  kbPreviewChunking: (body: Record<string, unknown>) =>
    http<Record<string, any>>('/api/kb/preview-chunking', { method: 'POST', body: JSON.stringify(body) }),

  analytics: {
    overview: (days = 30) => http<Record<string, any>>(`/api/analytics/overview?days=${days}`),
    languages: (days = 30) => http<Record<string, any>>(`/api/analytics/languages?days=${days}`),
    intents: (days = 30) => http<Record<string, any>>(`/api/analytics/intents?days=${days}`),
    topQueries: (days = 30, limit = 15) =>
      http<Record<string, any>>(`/api/analytics/top-queries?days=${days}&limit=${limit}`),
    unanswered: (limit = 50) => http<Record<string, any>>(`/api/analytics/unanswered?limit=${limit}`),
    latency: (days = 30) => http<Record<string, any>>(`/api/analytics/latency?days=${days}`),
    escalations: (days = 30) => http<Record<string, any>>(`/api/analytics/escalations?days=${days}`),
    quality: (days = 30) => http<Record<string, any>>(`/api/analytics/quality?days=${days}`),
    resolveUnanswered: (id: string, resolution: string, note: string) =>
      http<Record<string, any>>(`/api/analytics/unanswered/${id}/resolve`, {
        method: 'POST',
        body: JSON.stringify({ resolution, note }),
      }),
  },

  calls: (params: Record<string, string | number | undefined> = {}) => {
    const qs = new URLSearchParams()
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== '') qs.set(k, String(v))
    return http<Record<string, any>>(`/api/calls?${qs.toString()}`)
  },
  liveCalls: () => http<Record<string, any>>('/api/calls/live'),
  call: (id: string) => http<Record<string, any>>(`/api/calls/${id}`),
  callReview: (id: string, satisfaction: string, note: string) =>
    http<Record<string, any>>(
      `/api/calls/${id}/review?satisfaction=${encodeURIComponent(satisfaction)}&note=${encodeURIComponent(note)}`,
      { method: 'POST' },
    ),
  handoffBrief: (id: string) => http<Record<string, any>>(`/api/calls/${id}/handoff-brief`),
  transcriptUrl: (id: string) => `/api/calls/${id}/transcript.txt`,
  csvExportUrl: '/api/kb/export.csv',
  csvTemplateUrl: '/api/kb/template.csv',
}

/** Browser WebSocket URL for the phone simulator. */
export function simulatorSocket(): WebSocket {
  const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return new WebSocket(`${scheme}://${window.location.host}/api/simulator/ws`)
}
