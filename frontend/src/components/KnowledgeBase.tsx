import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type { KBRecord, KBRecordDetail, KBStats } from '../api'

/**
 * Knowledge-base console for non-technical staff.
 *
 * The knowledge base is the single source of truth: the assistant only says what
 * is in here. Everything is editable without a redeploy — records, verification
 * status, bulk CSV/YAML import, Google-Sheet sync — and each record can be
 * previewed as the chunks the assistant will actually retrieve.
 */

const BLANK = {
  slug: '',
  category: 'course',
  subcategory: '',
  title: '',
  body: '',
  structured: '{}',
  language: 'en-IN',
  tags: '',
  aliases: '',
  academic_year: '2025-26',
  source: '',
  source_uri: '',
  verified: false,
}

type FormState = typeof BLANK

export default function KnowledgeBase() {
  const [stats, setStats] = useState<KBStats | null>(null)
  const [records, setRecords] = useState<KBRecord[]>([])
  const [total, setTotal] = useState(0)
  const [categories, setCategories] = useState<string[]>([])
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState('')
  const [verified, setVerified] = useState('')
  const [language, setLanguage] = useState('')
  const [limit, setLimit] = useState(25)
  const [offset, setOffset] = useState(0)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [loading, setLoading] = useState(false)

  const [editing, setEditing] = useState<FormState | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [detail, setDetail] = useState<KBRecordDetail | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())

  const [importOpen, setImportOpen] = useState(false)
  const [importText, setImportText] = useState('')
  const [importFile, setImportFile] = useState<File | null>(null)
  const [dryRun, setDryRun] = useState(true)
  const [sheetUrl, setSheetUrl] = useState('')

  const flash = useCallback((msg: string) => {
    setNotice(msg)
    setError('')
    window.setTimeout(() => setNotice(''), 6000)
  }, [])

  const loadStats = useCallback(async () => {
    try {
      setStats(await api.kbStats())
    } catch (e) {
      setError(String((e as Error).message || e))
    }
  }, [])

  const loadRecords = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.kbRecords({ search, category, verified, language, limit, offset })
      setRecords(res.items)
      setTotal(res.total)
    } catch (e) {
      setError(String((e as Error).message || e))
    } finally {
      setLoading(false)
    }
  }, [search, category, verified, language, limit, offset])

  useEffect(() => {
    void loadStats()
    api
      .kbCategories()
      .then((c) => {
        const list = (c.categories || c.items || Object.keys(c.by_category || {})) as string[]
        setCategories(Array.isArray(list) ? list.map(String) : [])
      })
      .catch(() => setCategories([]))
  }, [loadStats])

  useEffect(() => {
    void loadRecords()
  }, [loadRecords])

  const openEditor = useCallback(
    async (record?: KBRecord) => {
      if (!record) {
        setEditingId(null)
        setDetail(null)
        setEditing({ ...BLANK })
        return
      }
      setEditingId(record.id)
      setEditing({
        slug: record.slug,
        category: record.category,
        subcategory: record.subcategory || '',
        title: record.title,
        body: record.body || '',
        structured: JSON.stringify(record.structured || {}, null, 2),
        language: record.language,
        tags: (record.tags || []).join(', '),
        aliases: (record.aliases || []).join(', '),
        academic_year: record.academic_year || '',
        source: record.source || '',
        source_uri: record.source_uri || '',
        verified: record.verified,
      })
      try {
        setDetail(await api.kbRecord(record.id))
      } catch {
        setDetail(null)
      }
    },
    [],
  )

  const save = useCallback(async () => {
    if (!editing) return
    let structured: Record<string, unknown> = {}
    if (editing.structured.trim()) {
      try {
        structured = JSON.parse(editing.structured)
      } catch {
        setError('The “structured facts” field is not valid JSON.')
        return
      }
    }
    const split = (s: string) =>
      s
        .split(',')
        .map((x) => x.trim())
        .filter(Boolean)
    const payload = {
      slug: editing.slug.trim(),
      category: editing.category,
      subcategory: editing.subcategory.trim() || null,
      title: editing.title.trim(),
      body: editing.body,
      structured,
      language: editing.language,
      tags: split(editing.tags),
      aliases: split(editing.aliases),
      academic_year: editing.academic_year.trim() || null,
      source: editing.source.trim(),
      source_uri: editing.source_uri.trim() || null,
      verified: editing.verified,
      change_note: editingId ? 'dashboard update' : 'dashboard create',
    }
    if (!payload.slug || !payload.title) {
      setError('A slug and a title are required.')
      return
    }
    try {
      if (editingId) await api.kbUpdate(editingId, payload)
      else await api.kbCreate(payload)
      setEditing(null)
      setEditingId(null)
      setDetail(null)
      flash(editingId ? 'Record updated and re-indexed.' : 'Record created and indexed.')
      await Promise.all([loadRecords(), loadStats()])
    } catch (e) {
      setError(String((e as Error).message || e))
    }
  }, [editing, editingId, flash, loadRecords, loadStats])

  const toggleVerify = useCallback(
    async (record: KBRecord) => {
      try {
        if (record.verified) {
          await api.kbUpdate(record.id, { ...record, verified: false, change_note: 'unverified from dashboard' })
        } else {
          await api.kbVerify(record.id)
        }
        flash(record.verified ? `Unverified “${record.title}”.` : `Verified “${record.title}”.`)
        await Promise.all([loadRecords(), loadStats()])
      } catch (e) {
        setError(String((e as Error).message || e))
      }
    },
    [flash, loadRecords, loadStats],
  )

  const remove = useCallback(
    async (record: KBRecord) => {
      if (!window.confirm(`Delete “${record.title}”? The assistant will stop using it immediately.`)) return
      try {
        await api.kbDelete(record.id)
        flash('Record deleted.')
        await Promise.all([loadRecords(), loadStats()])
      } catch (e) {
        setError(String((e as Error).message || e))
      }
    },
    [flash, loadRecords, loadStats],
  )

  const bulkVerify = useCallback(async () => {
    if (selected.size === 0) return
    try {
      const res = await api.kbBulkVerify([...selected])
      flash(`Verified ${res.updated ?? selected.size} records.`)
      setSelected(new Set())
      await Promise.all([loadRecords(), loadStats()])
    } catch (e) {
      setError(String((e as Error).message || e))
    }
  }, [selected, flash, loadRecords, loadStats])

  const runImport = useCallback(async () => {
    try {
      const res = await api.kbImport(importFile, importText, dryRun)
      const r = res.report || res
      flash(
        `${dryRun ? 'Dry run' : 'Import'}: ${r.rows ?? 0} rows · created ${r.created ?? 0} · updated ${
          r.updated ?? 0
        } · failed ${r.failed ?? 0}`,
      )
      if (!dryRun) {
        setImportOpen(false)
        setImportText('')
        setImportFile(null)
        await Promise.all([loadRecords(), loadStats()])
      }
    } catch (e) {
      setError(String((e as Error).message || e))
    }
  }, [importFile, importText, dryRun, flash, loadRecords, loadStats])

  const preview = useMemo(() => detail?.chunks || [], [detail])

  return (
    <div className="grid">
      {error ? <div className="error-banner">{error}</div> : null}
      {notice ? <div className="notice">{notice}</div> : null}

      <div className="grid cols-4">
        <div className="card stat">
          <div className="label">Records</div>
          <div className="value">{stats?.records.records ?? '—'}</div>
          <div className="delta">{stats?.records.chunks ?? 0} retrievable chunks</div>
        </div>
        <div className="card stat">
          <div className="label">Verified</div>
          <div className="value" style={{ color: 'var(--ok)' }}>
            {stats?.records.verified ?? '—'}
          </div>
          <div className="delta">{stats?.records.unverified ?? 0} still unverified</div>
        </div>
        <div className="card stat">
          <div className="label">Stale</div>
          <div className="value" style={{ color: (stats?.records.stale ?? 0) > 0 ? 'var(--warn)' : undefined }}>
            {stats?.records.stale ?? '—'}
          </div>
          <div className="delta">older than {stats?.staleness_days ?? 180} days</div>
        </div>
        <div className="card stat">
          <div className="label">Academic year</div>
          <div className="value" style={{ fontSize: 21 }}>
            {stats?.academic_year ?? '—'}
          </div>
          <div className="delta">
            {stats?.google_sheet_configured ? 'Google Sheet sync on' : 'Sheet sync not configured'}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="row">
          <h2 style={{ margin: 0 }}>Knowledge base</h2>
          <div className="spacer" />
          <a className="btn sm" href={api.csvTemplateUrl} download>
            CSV template
          </a>
          <a className="btn sm" href={api.csvExportUrl} download>
            Export CSV
          </a>
          <button className="btn sm" onClick={() => setImportOpen((v) => !v)}>
            Import
          </button>
          <button
            className="btn sm"
            onClick={async () => {
              try {
                const r = await api.kbSyncSheet(sheetUrl)
                flash(`Sheet sync: updated ${r.updated ?? 0}, created ${r.created ?? 0}`)
                await Promise.all([loadRecords(), loadStats()])
              } catch (e) {
                setError(String((e as Error).message || e))
              }
            }}
          >
            Sync Google Sheet
          </button>
          <button
            className="btn sm"
            onClick={async () => {
              try {
                await api.kbReindex()
                flash('Re-indexed every record.')
                await loadStats()
              } catch (e) {
                setError(String((e as Error).message || e))
              }
            }}
          >
            Re-index
          </button>
          <button className="btn sm primary" onClick={() => void openEditor()}>
            + New record
          </button>
        </div>

        {importOpen ? (
          <div className="mt" style={{ borderTop: '1px solid var(--line)', paddingTop: 14 }}>
            <h3>Bulk import</h3>
            <p className="sub">
              CSV, YAML or JSON. CSV is the easiest route for a fee table: download the template,
              fill it in a spreadsheet, upload it. Always dry-run first — it reports what would
              change without touching the live knowledge base.
            </p>
            <div className="row">
              <input
                type="file"
                accept=".csv,.yaml,.yml,.json,.txt"
                onChange={(e) => setImportFile(e.target.files?.[0] || null)}
                style={{ maxWidth: 340 }}
              />
              <label className="row small" style={{ gap: 6 }}>
                <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} style={{ width: 16 }} />
                Dry run only
              </label>
              <button className="btn sm primary" onClick={() => void runImport()}>
                {dryRun ? 'Preview import' : 'Import now'}
              </button>
            </div>
            <textarea
              className="mt"
              placeholder="…or paste CSV / YAML rows here"
              value={importText}
              onChange={(e) => setImportText(e.target.value)}
            />
            <label className="field mt">
              <span>Google Sheet “publish as CSV” URL (optional)</span>
              <input
                placeholder="https://docs.google.com/spreadsheets/d/e/…/pub?output=csv"
                value={sheetUrl}
                onChange={(e) => setSheetUrl(e.target.value)}
              />
            </label>
          </div>
        ) : null}

        <div className="searchbar mt">
          <label className="field" style={{ margin: 0 }}>
            <span>Search</span>
            <input
              placeholder="title, body, alias, course code…"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value)
                setOffset(0)
              }}
            />
          </label>
          <label className="field" style={{ margin: 0 }}>
            <span>Category</span>
            <select
              value={category}
              onChange={(e) => {
                setCategory(e.target.value)
                setOffset(0)
              }}
            >
              <option value="">All</option>
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="field" style={{ margin: 0 }}>
            <span>Verified</span>
            <select
              value={verified}
              onChange={(e) => {
                setVerified(e.target.value)
                setOffset(0)
              }}
            >
              <option value="">Any</option>
              <option value="true">Verified</option>
              <option value="false">Unverified</option>
            </select>
          </label>
          <label className="field" style={{ margin: 0 }}>
            <span>Language</span>
            <select
              value={language}
              onChange={(e) => {
                setLanguage(e.target.value)
                setOffset(0)
              }}
            >
              <option value="">Any</option>
              {['en-IN', 'hi-IN', 'mr-IN', 'gu-IN'].map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="field" style={{ margin: 0 }}>
            <span>Per page</span>
            <select
              value={limit}
              onChange={(e) => {
                setLimit(Number(e.target.value))
                setOffset(0)
              }}
            >
              {[25, 50, 100].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <button onClick={() => void loadRecords()} disabled={loading}>
            {loading ? '…' : 'Apply'}
          </button>
        </div>

        {selected.size > 0 ? (
          <div className="row mt">
            <span className="pill brand">{selected.size} selected</span>
            <button className="btn sm primary" onClick={() => void bulkVerify()}>
              Verify selected
            </button>
            <button className="btn sm" onClick={() => setSelected(new Set())}>
              Clear
            </button>
          </div>
        ) : null}

        <table className="mt">
          <thead>
            <tr>
              <th style={{ width: 30 }} />
              <th>Title</th>
              <th>Category</th>
              <th>Lang</th>
              <th>Status</th>
              <th style={{ textAlign: 'right' }}>Chunks</th>
              <th style={{ textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {records.length === 0 ? (
              <tr>
                <td colSpan={7}>
                  <div className="empty">No records match those filters.</div>
                </td>
              </tr>
            ) : null}
            {records.map((r) => (
              <tr key={r.id}>
                <td>
                  <input
                    type="checkbox"
                    style={{ width: 16 }}
                    checked={selected.has(r.id)}
                    onChange={(e) => {
                      const next = new Set(selected)
                      if (e.target.checked) next.add(r.id)
                      else next.delete(r.id)
                      setSelected(next)
                    }}
                  />
                </td>
                <td>
                  <a
                    href="#"
                    onClick={(e) => {
                      e.preventDefault()
                      void openEditor(r)
                    }}
                  >
                    {r.title}
                  </a>
                  <div className="small muted mono">{r.slug}</div>
                </td>
                <td>
                  <span className="pill">{r.category}</span>
                  {r.subcategory ? <div className="small muted">{r.subcategory}</div> : null}
                </td>
                <td className="mono">{r.language}</td>
                <td>
                  {r.verified ? <span className="pill ok">verified</span> : <span className="pill warn">unverified</span>}
                  {r.stale ? <span className="pill bad" style={{ marginLeft: 5 }}>stale</span> : null}
                  <div className="small muted">rev {r.revision}</div>
                </td>
                <td style={{ textAlign: 'right' }}>{r.chunk_count ?? '—'}</td>
                <td style={{ textAlign: 'right' }}>
                  <div className="row end">
                    <button className="btn sm" onClick={() => void toggleVerify(r)}>
                      {r.verified ? 'Unverify' : 'Verify'}
                    </button>
                    <button className="btn sm" onClick={() => void openEditor(r)}>
                      Edit
                    </button>
                    <button className="btn sm danger" onClick={() => void remove(r)}>
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="row mt">
          <span className="small muted">
            {total} records · showing {offset + 1}–{Math.min(offset + limit, total)}
          </span>
          <div className="spacer" />
          <button className="btn sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
            ← Prev
          </button>
          <button
            className="btn sm"
            disabled={offset + limit >= total}
            onClick={() => setOffset(offset + limit)}
          >
            Next →
          </button>
        </div>
      </div>

      {editing ? (
        <>
          <div className="scrim" onClick={() => setEditing(null)} />
          <div className="drawer">
            <div className="row">
              <h2 style={{ margin: 0 }}>{editingId ? 'Edit record' : 'New record'}</h2>
              <div className="spacer" />
              <button className="btn sm" onClick={() => setEditing(null)}>
                Close
              </button>
            </div>
            <p className="sub">
              The assistant only ever says what is written here. Keep the spoken answer short — one
              or two sentences — and put long lists in <code>structured</code> so they can be sent by
              message instead of read aloud.
            </p>

            <label className="field">
              <span>Slug (unique id)</span>
              <input value={editing.slug} onChange={(e) => setEditing({ ...editing, slug: e.target.value })} />
            </label>
            <div className="row">
              <label className="field" style={{ flex: 1 }}>
                <span>Category</span>
                <select value={editing.category} onChange={(e) => setEditing({ ...editing, category: e.target.value })}>
                  {(categories.length ? categories : [editing.category]).map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field" style={{ flex: 1 }}>
                <span>Sub-category</span>
                <input
                  value={editing.subcategory}
                  onChange={(e) => setEditing({ ...editing, subcategory: e.target.value })}
                />
              </label>
              <label className="field" style={{ flex: 1 }}>
                <span>Language</span>
                <select value={editing.language} onChange={(e) => setEditing({ ...editing, language: e.target.value })}>
                  {['en-IN', 'hi-IN', 'mr-IN', 'gu-IN', 'ta-IN', 'bn-IN', 'te-IN', 'kn-IN', 'ml-IN', 'pa-IN', 'ur-IN', 'raj-IN'].map(
                    (c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ),
                  )}
                </select>
              </label>
            </div>
            <label className="field">
              <span>Title</span>
              <input value={editing.title} onChange={(e) => setEditing({ ...editing, title: e.target.value })} />
            </label>
            <label className="field">
              <span>Aliases (comma separated — what callers actually say)</span>
              <input value={editing.aliases} onChange={(e) => setEditing({ ...editing, aliases: e.target.value })} />
            </label>
            <label className="field">
              <span>Body</span>
              <textarea
                style={{ minHeight: 150 }}
                value={editing.body}
                onChange={(e) => setEditing({ ...editing, body: e.target.value })}
              />
            </label>
            <label className="field">
              <span>Structured facts (JSON)</span>
              <textarea
                className="mono"
                style={{ minHeight: 170 }}
                value={editing.structured}
                onChange={(e) => setEditing({ ...editing, structured: e.target.value })}
              />
            </label>
            <div className="row">
              <label className="field" style={{ flex: 1 }}>
                <span>Academic year</span>
                <input
                  value={editing.academic_year}
                  onChange={(e) => setEditing({ ...editing, academic_year: e.target.value })}
                />
              </label>
              <label className="field" style={{ flex: 1 }}>
                <span>Tags</span>
                <input value={editing.tags} onChange={(e) => setEditing({ ...editing, tags: e.target.value })} />
              </label>
            </div>
            <label className="field">
              <span>Source</span>
              <input value={editing.source} onChange={(e) => setEditing({ ...editing, source: e.target.value })} />
            </label>
            <label className="row" style={{ gap: 8, marginBottom: 14 }}>
              <input
                type="checkbox"
                style={{ width: 17 }}
                checked={editing.verified}
                onChange={(e) => setEditing({ ...editing, verified: e.target.checked })}
              />
              <span className="small">
                Verified — a human has confirmed this against an official document. Unverified
                answers are spoken with a “please confirm with admissions” qualification.
              </span>
            </label>

            <div className="row">
              <button className="btn primary" onClick={() => void save()}>
                Save &amp; re-index
              </button>
              <div className="spacer" />
              <span className="small muted">{editing.structured.trim().length} chars of facts</span>
            </div>

            {preview.length > 0 ? (
              <div className="mt">
                <h3>What the assistant retrieves ({preview.length} chunks)</h3>
                {preview.map((c) => (
                  <div className="chunk-box" key={c.id}>
                    <b className="small muted">
                      #{c.position} · {c.text.length} chars · ~{c.tokens} tokens
                    </b>
                    {'\n'}
                    {c.text}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </>
      ) : null}
    </div>
  )
}
