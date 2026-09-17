import { useCallback, useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api'

/**
 * Operations dashboard: call volume, language distribution, top queries,
 * resolved vs escalated, latency by pipeline stage and the unanswered-question
 * backlog that tells staff what to add to the knowledge base next.
 */

const CHART_COLORS = ['#5b8cff', '#37d39b', '#f5b544', '#ff6b7a', '#a78bfa', '#38bdf8', '#f472b6', '#facc15']

function Stat({ label, value, delta, tone }: { label: string; value: string | number; delta?: string; tone?: string }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value" style={tone === 'bad' ? { color: 'var(--bad)' } : tone === 'ok' ? { color: 'var(--ok)' } : undefined}>
        {value}
      </div>
      {delta ? <div className="delta">{delta}</div> : null}
    </div>
  )
}

function pct(n: unknown): string {
  return `${Math.round(Number(n || 0) * 100)}%`
}

export default function Dashboard() {
  const [days, setDays] = useState(30)
  const [overview, setOverview] = useState<Record<string, any> | null>(null)
  const [langs, setLangs] = useState<Record<string, any> | null>(null)
  const [intents, setIntents] = useState<Record<string, any> | null>(null)
  const [top, setTop] = useState<Record<string, any> | null>(null)
  const [latency, setLatency] = useState<Record<string, any> | null>(null)
  const [escalations, setEscalations] = useState<Record<string, any> | null>(null)
  const [quality, setQuality] = useState<Record<string, any> | null>(null)
  const [unanswered, setUnanswered] = useState<Record<string, any> | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [resolving, setResolving] = useState<string>('')
  const [resolution, setResolution] = useState('answered_later')
  const [note, setNote] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const [o, l, i, t, la, e, q, u] = await Promise.all([
        api.analytics.overview(days),
        api.analytics.languages(days),
        api.analytics.intents(days),
        api.analytics.topQueries(days, 12),
        api.analytics.latency(days),
        api.analytics.escalations(days),
        api.analytics.quality(days),
        api.analytics.unanswered(40),
      ])
      setOverview(o)
      setLangs(l)
      setIntents(i)
      setTop(t)
      setLatency(la)
      setEscalations(e)
      setQuality(q)
      setUnanswered(u)
    } catch (err) {
      setError(String((err as Error).message || err))
    } finally {
      setLoading(false)
    }
  }, [days])

  useEffect(() => {
    void load()
    const id = window.setInterval(() => void load(), 20000)
    return () => window.clearInterval(id)
  }, [load])

  const totals = overview?.totals || {}
  const stageRows = Object.entries((latency?.stages || {}) as Record<string, any>).map(([stage, v]) => ({
    stage,
    avg: Number(v?.avg || 0),
    p95: Number(v?.p95 || 0),
  }))
  const langRows = ((langs?.languages || langs?.items || []) as any[]).map((r) => ({
    name: r.language || r.code || r.name || '?',
    calls: Number(r.calls ?? r.count ?? 0),
    pct: Number(r.share ?? r.percent ?? 0),
  }))
  const intentRows = ((intents?.turn_intents || []) as any[]).map((r) => ({
    name: r.intent,
    count: Number(r.count || 0),
  }))
  const topRows = (top?.items || []) as any[]
  const outcomeRows = [
    { name: 'Resolved', value: Number(totals.resolved || 0) },
    { name: 'Escalated', value: Number(totals.escalated || 0) },
    { name: 'Abandoned', value: Number(totals.abandoned || 0) },
    { name: 'Failed', value: Number(totals.failed || 0) },
  ].filter((r) => r.value > 0)
  const unresolved = ((unanswered?.items || []) as any[]).filter((r) => !r.resolution)

  return (
    <div className="grid">
      {error ? <div className="error-banner">{error}</div> : null}

      <div className="row">
        <h2 style={{ margin: 0 }}>Analytics</h2>
        <div className="spacer" />
        <select value={days} onChange={(e) => setDays(Number(e.target.value))} style={{ width: 170 }}>
          <option value={7}>Last 7 days</option>
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
        </select>
        <button onClick={() => void load()} disabled={loading}>
          {loading ? 'Loading…' : 'Refresh'}
        </button>
      </div>

      <div className="grid cols-4">
        <Stat label="Calls" value={Number(totals.calls || 0)} delta={`${Number(totals.avg_turns || 0).toFixed(1)} turns avg`} />
        <Stat
          label="Resolved by AI"
          value={pct(totals.resolution_rate)}
          delta={`${Number(totals.resolved || 0)} calls`}
          tone="ok"
        />
        <Stat
          label="Escalated to human"
          value={pct(totals.escalation_rate)}
          delta={`${Number(totals.escalated || 0)} calls`}
          tone={Number(totals.escalation_rate || 0) > 0.35 ? 'bad' : undefined}
        />
        <Stat
          label="Avg response"
          value={`${Number(overview?.latency_ms?.avg_response || 0).toFixed(0)} ms`}
          delta={`p95 ${Number(overview?.latency_ms?.p95_response || 0).toFixed(0)} ms`}
        />
        <Stat label="Avg call length" value={`${Number(totals.avg_duration_seconds || 0).toFixed(0)}s`} />
        <Stat label="Barge-ins" value={Number(totals.barge_ins || 0)} delta="caller interruptions" />
        <Stat
          label="Grounded answers"
          value={pct(quality?.grounded_rate)}
          delta={`${Number(quality?.grounded || 0)} of ${Number(quality?.assistant_turns || 0)} turns`}
          tone="ok"
        />
        <Stat
          label="Unanswered backlog"
          value={Number(totals.unanswered_questions || 0)}
          delta="knowledge gaps to fill"
          tone={Number(totals.unanswered_questions || 0) > 0 ? 'bad' : 'ok'}
        />
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h3>Language distribution</h3>
          {langRows.length === 0 ? (
            <div className="empty">No calls recorded yet. Place a call in the simulator.</div>
          ) : (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={langRows} dataKey="calls" nameKey="name" innerRadius={55} outerRadius={92} paddingAngle={2}>
                  {langRows.map((_, i) => (
                    <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="card">
          <h3>Call outcomes</h3>
          {outcomeRows.length === 0 ? (
            <div className="empty">No calls yet.</div>
          ) : (
            <>
              <ResponsiveContainer width="100%" height={190}>
                <PieChart>
                  <Pie data={outcomeRows} dataKey="value" nameKey="name" outerRadius={78} label>
                    {outcomeRows.map((_, i) => (
                      <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="row small muted">
                <span>Template fallback rate {pct(quality?.template_fallback_rate)}</span>
                <span>·</span>
                <span>Avg confidence {Number(quality?.avg_confidence || 0).toFixed(2)}</span>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h3>Latency by pipeline stage (ms)</h3>
          {stageRows.length === 0 ? (
            <div className="empty">No turns recorded yet.</div>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={stageRows}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="stage" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="avg" name="average" fill="#5b8cff" radius={[5, 5, 0, 0]} />
                <Bar dataKey="p95" name="p95" fill="#f5b544" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
          <p className="small muted">
            Target is under 1–2s end to end. ASR and TTS run on the provider's clock; retrieval and
            the first audio chunk are what this system controls.
          </p>
        </div>

        <div className="card">
          <h3>What callers ask about</h3>
          {intentRows.length === 0 ? (
            <div className="empty">No intents recorded yet.</div>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={intentRows.slice(0, 10)} layout="vertical" margin={{ left: 34 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={120} />
                <Tooltip />
                <Bar dataKey="count" fill="#37d39b" radius={[0, 5, 5, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h3>Top queries</h3>
          {topRows.length === 0 ? (
            <div className="empty">No queries yet.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Question</th>
                  <th>Intent</th>
                  <th style={{ textAlign: 'right' }}>Asks</th>
                </tr>
              </thead>
              <tbody>
                {topRows.map((r, i) => (
                  <tr key={i}>
                    <td>
                      {r.canonical}
                      <div className="small muted">{(r.examples || [])[0]}</div>
                    </td>
                    <td>
                      <span className="pill brand">{r.intent}</span>
                    </td>
                    <td style={{ textAlign: 'right' }}>{r.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="card">
          <h3>Escalations</h3>
          <div className="row small">
            <span className="pill warn">total {Number(escalations?.total || 0)}</span>
            <span className="pill">avg wait {Number(escalations?.avg_wait_seconds || 0).toFixed(0)}s</span>
            <span className="pill">p95 wait {Number(escalations?.p95_wait_seconds || 0).toFixed(0)}s</span>
          </div>
          {((escalations?.by_reason || []) as any[]).length === 0 ? (
            <div className="empty">No escalations recorded.</div>
          ) : (
            <table className="mt">
              <thead>
                <tr>
                  <th>Reason</th>
                  <th style={{ textAlign: 'right' }}>Count</th>
                </tr>
              </thead>
              <tbody>
                {((escalations?.by_reason || []) as any[]).map((r, i) => (
                  <tr key={i}>
                    <td className="mono">{r.reason}</td>
                    <td style={{ textAlign: 'right' }}>{r.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Unanswered questions — the knowledge-base to-do list</h3>
        <p className="sub">
          Every question the assistant could not ground is logged here. Resolving one should mean
          adding or verifying a knowledge-base record, so the next caller gets an answer.
        </p>
        {unresolved.length === 0 ? (
          <div className="empty">Nothing outstanding. 🎉</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Question</th>
                <th>Lang</th>
                <th>Intent</th>
                <th style={{ textAlign: 'right' }}>Times asked</th>
                <th>Mark resolved</th>
              </tr>
            </thead>
            <tbody>
              {unresolved.slice(0, 15).map((r: any) => (
                <tr key={r.id}>
                  <td>
                    {r.question}
                    <div className="small muted">
                      best score {Number(r.best_score || 0).toFixed(2)} ·{' '}
                      {r.last_seen_at ? new Date(r.last_seen_at).toLocaleString() : ''}
                    </div>
                  </td>
                  <td>{r.language}</td>
                  <td>
                    <span className="pill brand">{r.intent}</span>
                  </td>
                  <td style={{ textAlign: 'right' }}>{r.occurrences}</td>
                  <td>
                    {resolving === r.id ? (
                      <div className="row">
                        <select value={resolution} onChange={(e) => setResolution(e.target.value)} style={{ width: 190 }}>
                          <option value="answered_later">Answered offline</option>
                          <option value="kb_added">Added to knowledge base</option>
                          <option value="out_of_scope">Out of scope</option>
                          <option value="wont_fix">Won't fix</option>
                        </select>
                        <input
                          placeholder="Note (optional)"
                          value={note}
                          onChange={(e) => setNote(e.target.value)}
                          style={{ width: 190 }}
                        />
                        <button
                          className="btn sm primary"
                          onClick={async () => {
                            try {
                              await api.analytics.resolveUnanswered(r.id, resolution, note)
                              setResolving('')
                              setNote('')
                              await load()
                            } catch (err) {
                              setError(String((err as Error).message || err))
                            }
                          }}
                        >
                          Save
                        </button>
                        <button className="btn sm" onClick={() => setResolving('')}>
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button className="btn sm" onClick={() => setResolving(r.id)}>
                        Resolve
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
