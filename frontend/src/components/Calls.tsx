import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'

/**
 * Call log and QA review.
 *
 * Every call keeps a full transcript with per-turn latency, grounding, citations
 * and tool calls, plus the escalation record and the hand-off brief that a human
 * agent would have received. This is what supervisors use to spot-check quality.
 */

interface Turn {
  seq: number
  role: string
  text: string
  language: string | null
  interrupted: boolean
  grounded: boolean | null
  confidence: number | null
  timings: { asr_ms: number | null; retrieval_ms: number | null; llm_ms: number | null; tts_ms: number | null; total_ms: number | null }
  citations: { title?: string; score?: number; verified?: boolean }[]
  tool_calls: unknown[]
  at: string | null
}

const SATISFACTION = ['resolved_ok', 'resolved_partial', 'caller_unhappy', 'escalated_ok', 'escalated_bad', 'not_reviewed']

export default function Calls() {
  const [rows, setRows] = useState<any[]>([])
  const [live, setLive] = useState<any[]>([])
  const [total, setTotal] = useState(0)
  const [status, setStatus] = useState('')
  const [escalated, setEscalated] = useState('')
  const [language, setLanguage] = useState('')
  const [days, setDays] = useState(30)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [openId, setOpenId] = useState<string | null>(null)
  const [detail, setDetail] = useState<Record<string, any> | null>(null)
  const [brief, setBrief] = useState<Record<string, any> | null>(null)
  const [satisfaction, setSatisfaction] = useState('resolved_ok')
  const [note, setNote] = useState('')

  const load = useCallback(async () => {
    try {
      const res = await api.calls({ status, escalated: escalated || undefined, language, days, limit: 50 })
      setRows(res.items || [])
      setTotal(res.total || 0)
    } catch (e) {
      setError(String((e as Error).message || e))
    }
    try {
      const l = await api.liveCalls()
      setLive(l.calls || l.items || [])
    } catch {
      setLive([])
    }
  }, [status, escalated, language, days])

  useEffect(() => {
    void load()
    const id = window.setInterval(() => void load(), 8000)
    return () => window.clearInterval(id)
  }, [load])

  const openCall = useCallback(async (callId: string) => {
    setOpenId(callId)
    setDetail(null)
    setBrief(null)
    try {
      setDetail(await api.call(callId))
    } catch (e) {
      setError(String((e as Error).message || e))
    }
    try {
      setBrief(await api.handoffBrief(callId))
    } catch {
      setBrief(null)
    }
  }, [])

  const submitReview = useCallback(async () => {
    if (!openId) return
    try {
      await api.callReview(openId, satisfaction, note)
      setNotice('Review saved.')
      setNote('')
      window.setTimeout(() => setNotice(''), 4000)
      await load()
    } catch (e) {
      setError(String((e as Error).message || e))
    }
  }, [openId, satisfaction, note, load])

  const turns: Turn[] = (detail?.transcript || []) as Turn[]
  const record = detail?.record || detail || {}

  return (
    <div className="grid">
      {error ? <div className="error-banner">{error}</div> : null}
      {notice ? <div className="notice">{notice}</div> : null}

      {live.length > 0 ? (
        <div className="card">
          <h3>Live calls</h3>
          <div className="row">
            {live.map((c: any) => (
              <span className="pill ok" key={c.call_id}>
                {c.call_id} · {c.state} · {c.language || 'detecting'} · {c.turns} turns
                {c.escalated ? ' · escalated' : ''}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <div className="card">
        <div className="row">
          <h2 style={{ margin: 0 }}>Call log</h2>
          <div className="spacer" />
          <select value={days} onChange={(e) => setDays(Number(e.target.value))} style={{ width: 150 }}>
            <option value={1}>Today</option>
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
          <select value={status} onChange={(e) => setStatus(e.target.value)} style={{ width: 160 }}>
            <option value="">Any status</option>
            {['completed', 'escalated', 'abandoned', 'failed'].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <select value={escalated} onChange={(e) => setEscalated(e.target.value)} style={{ width: 150 }}>
            <option value="">Escalated: any</option>
            <option value="true">Escalated only</option>
            <option value="false">Not escalated</option>
          </select>
          <select value={language} onChange={(e) => setLanguage(e.target.value)} style={{ width: 140 }}>
            <option value="">Any language</option>
            {['en-IN', 'hi-IN', 'mr-IN', 'gu-IN', 'ta-IN', 'bn-IN'].map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <button onClick={() => void load()}>Refresh</button>
        </div>

        <table className="mt">
          <thead>
            <tr>
              <th>When</th>
              <th>Call</th>
              <th>Language</th>
              <th>Outcome</th>
              <th style={{ textAlign: 'right' }}>Turns</th>
              <th style={{ textAlign: 'right' }}>Length</th>
              <th style={{ textAlign: 'right' }}>Avg resp.</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={8}>
                  <div className="empty">
                    No calls recorded yet. Place a call in the Phone simulator tab and it will appear
                    here with its full transcript.
                  </div>
                </td>
              </tr>
            ) : null}
            {rows.map((r) => (
              <tr key={r.id}>
                <td className="small">{r.started_at ? new Date(r.started_at).toLocaleString() : '—'}</td>
                <td className="mono small">
                  {r.id}
                  <div className="muted">{r.from_number || 'unknown caller'}</div>
                </td>
                <td>
                  {r.detected_language ? <span className="pill brand">{r.detected_language}</span> : <span className="muted">—</span>}
                  {r.language_confidence ? (
                    <div className="small muted">{Math.round(Number(r.language_confidence) * 100)}%</div>
                  ) : null}
                </td>
                <td>
                  {r.escalated ? <span className="pill warn">escalated</span> : null}
                  {r.resolved ? <span className="pill ok">resolved</span> : null}
                  {!r.escalated && !r.resolved ? <span className="pill">{r.status}</span> : null}
                  <div className="small muted">{r.end_reason || ''}</div>
                </td>
                <td style={{ textAlign: 'right' }}>{r.turn_count}</td>
                <td style={{ textAlign: 'right' }}>{Number(r.duration_seconds || 0).toFixed(0)}s</td>
                <td style={{ textAlign: 'right' }}>{Number(r.avg_response_latency_ms || 0).toFixed(0)} ms</td>
                <td style={{ textAlign: 'right' }}>
                  <button className="btn sm" onClick={() => void openCall(r.id)}>
                    Open
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="small muted mt">{total} calls match these filters</div>
      </div>

      {openId ? (
        <>
          <div className="scrim" onClick={() => setOpenId(null)} />
          <div className="drawer">
            <div className="row">
              <h2 style={{ margin: 0 }}>Call {openId}</h2>
              <div className="spacer" />
              <a className="btn sm" href={api.transcriptUrl(openId)} target="_blank" rel="noreferrer">
                Transcript .txt
              </a>
              <button className="btn sm" onClick={() => setOpenId(null)}>
                Close
              </button>
            </div>

            <div className="grid cols-4 mt">
              <div>
                <div className="label muted small">Status</div>
                <div>{record.status || '—'}</div>
              </div>
              <div>
                <div className="label muted small">Language</div>
                <div>
                  {record.detected_language || '—'}{' '}
                  <span className="muted small">via {record.language_method || '—'}</span>
                </div>
              </div>
              <div>
                <div className="label muted small">Duration</div>
                <div>{Number(record.duration_seconds || 0).toFixed(0)}s</div>
              </div>
              <div>
                <div className="label muted small">Barge-ins</div>
                <div>{record.barge_in_count ?? 0}</div>
              </div>
            </div>

            <p className="small muted mt">
              Recording consent announced: {record.recording_consent ? 'yes' : 'no'} ·
              {record.recording_uri ? ` recording stored` : ' no audio stored (privacy by default)'} ·
              satisfaction: <b>{record.satisfaction || 'not reviewed'}</b>
            </p>

            {brief?.summary ? (
              <div className="card mt" style={{ background: '#0e1628' }}>
                <h3>Hand-off brief given to the human agent</h3>
                <p style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{brief.summary}</p>
                {brief.whisper ? <p className="small muted mt">Whisper: {brief.whisper}</p> : null}
                {((brief.unresolved || []) as string[]).length > 0 ? (
                  <div className="mt">
                    <div className="label muted small">Left unresolved</div>
                    <ul className="small">
                      {((brief.unresolved || []) as string[]).map((u, i) => (
                        <li key={i}>{u}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            ) : null}

            <h3 className="mt">Transcript ({turns.length} turns)</h3>
            {turns.length === 0 ? <div className="empty">No turns recorded.</div> : null}
            {turns.map((t) => (
              <div key={t.seq} className={`bubble ${t.role === 'assistant' ? 'bot' : 'me'}`} style={{ maxWidth: '100%', marginBottom: 8 }}>
                <span className="tag">
                  #{t.seq} · {t.role} · {t.language || '—'}
                  {t.interrupted ? ' · interrupted' : ''}
                  {t.grounded === false ? ' · NOT grounded' : t.grounded ? ' · grounded' : ''}
                  {t.confidence != null ? ` · conf ${Number(t.confidence).toFixed(2)}` : ''}
                  {t.timings?.total_ms ? ` · ${Number(t.timings.total_ms).toFixed(0)} ms` : ''}
                </span>
                {t.text}
                {t.citations?.length ? (
                  <div className="small muted mt">
                    sources: {t.citations.map((c) => `${c.title}${c.verified === false ? ' (unverified)' : ''}`).join('; ')}
                  </div>
                ) : null}
                {t.timings ? (
                  <div className="small muted mono">
                    asr {t.timings.asr_ms ?? 0} · retrieval {t.timings.retrieval_ms ?? 0} · llm{' '}
                    {t.timings.llm_ms ?? 0} · tts {t.timings.tts_ms ?? 0}
                  </div>
                ) : null}
              </div>
            ))}

            <h3 className="mt">QA review</h3>
            <div className="row">
              <select value={satisfaction} onChange={(e) => setSatisfaction(e.target.value)} style={{ width: 220 }}>
                {SATISFACTION.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
              <input placeholder="Reviewer note" value={note} onChange={(e) => setNote(e.target.value)} />
              <button className="btn primary" onClick={() => void submitReview()}>
                Save review
              </button>
            </div>
          </div>
        </>
      ) : null}
    </div>
  )
}
