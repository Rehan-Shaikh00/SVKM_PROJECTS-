import { useEffect, useState } from 'react'
import Calls from './components/Calls'
import Dashboard from './components/Dashboard'
import KnowledgeBase from './components/KnowledgeBase'
import Simulator from './components/Simulator'

type Tab = 'phone' | 'analytics' | 'kb' | 'calls'

const TABS: { id: Tab; label: string }[] = [
  { id: 'phone', label: 'Phone simulator' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'kb', label: 'Knowledge base' },
  { id: 'calls', label: 'Calls & QA' },
]

export default function App() {
  const [tab, setTab] = useState<Tab>('phone')
  const [config, setConfig] = useState<Record<string, any> | null>(null)
  const [health, setHealth] = useState<Record<string, any> | null>(null)

  useEffect(() => {
    const load = async () => {
      try {
        setConfig(await api_config())
      } catch {
        setConfig(null)
      }
      try {
        setHealth(await api_health())
      } catch {
        setHealth(null)
      }
    }
    void load()
    const id = window.setInterval(() => void load(), 15000)
    return () => window.clearInterval(id)
  }, [])

  const ready = health?.status === 'ready'

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">N</div>
          <div>
            <h1>NMIMS Global University, Dhule · AI Voice Assistant</h1>
            <p>
              {config?.assistant_name || 'Saarthi'} · admissions helpline {config?.helpline || ''} ·{' '}
              {config?.academic_year || ''}
            </p>
          </div>
        </div>

        <div className="row" style={{ gap: 6 }}>
          <span className={`pill ${ready ? 'ok' : health ? 'warn' : ''}`}>
            {ready ? 'knowledge base ready' : health ? 'degraded' : 'connecting…'}
          </span>
          {health?.records ? <span className="pill">{health.records} records</span> : null}
          {health?.chunks ? <span className="pill">{health.chunks} chunks</span> : null}
          <span className="pill brand">{(config?.languages || []).length} languages</span>
        </div>

        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} className={`tab ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="content">
        {tab === 'phone' ? <Simulator /> : null}
        {tab === 'analytics' ? <Dashboard /> : null}
        {tab === 'kb' ? <KnowledgeBase /> : null}
        {tab === 'calls' ? <Calls /> : null}
      </main>
    </div>
  )
}

// Small local wrappers so the header can poll without re-importing the client.
async function api_config() {
  const res = await fetch('/api/config')
  if (!res.ok) throw new Error(String(res.status))
  return res.json()
}

async function api_health() {
  const res = await fetch('/health/ready')
  if (!res.ok) throw new Error(String(res.status))
  return res.json()
}
