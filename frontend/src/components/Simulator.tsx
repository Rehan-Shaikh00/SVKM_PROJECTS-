import { useCallback, useEffect, useRef, useState } from 'react'
import { api, simulatorSocket } from '../api'

/**
 * Browser phone. Drives the same WebSocket session as a real Twilio call:
 * greeting → spoken language identification → grounded RAG answers → barge-in →
 * follow-up SMS offer → human handoff.
 *
 * Two input modes:
 *   voice — Web Speech API does ASR in the browser, SpeechSynthesis (or decoded
 *           server audio) does TTS. This is what ASR_PROVIDER=client means.
 *   text  — type instead of speaking. Nothing is played aloud, so the server
 *           skips its client-speech wait entirely.
 */

interface Line {
  id: number
  who: 'bot' | 'me' | 'sys'
  text: string
  tag?: string
}

interface Ev {
  t: number
  kind: string
  detail: string
}

// The Web Speech API is not in TypeScript's DOM lib; declare the parts we use.
type AnyRec = any

const QUICK_QUESTIONS = [
  'What courses do you offer?',
  'How do I apply for admission?',
  'What is the fee for BBA?',
  'Do you offer MBBS?',
  'Is hostel available on campus?',
  'What documents do I need?',
  'Scholarships available',
  'What are the placements like?',
  'बीटेक की फीस कितनी है',
  'एडमिशन के लिए कौन सी परीक्षा देनी होगी',
  'बीटेक संगणक अभियांत्रिकीची माहिती द्या',
  'प्रवेशासाठी किती गुण आवश्यक आहेत',
  'Talk to a human',
]

/** G.711 µ-law → linear PCM16 (matches backend/app/voice/audio.py). */
function mulawToPcm16(bytes: Uint8Array): Int16Array {
  const out = new Int16Array(bytes.length)
  for (let i = 0; i < bytes.length; i++) {
    let b = ~bytes[i] & 0xff
    const sign = b & 0x80
    if (sign) b ^= 0x80
    let sample = ((b & 0x0f) << 3) + 0x84
    sample = (b & 0x70) === 0 ? sample - 0x84 : sample << (((b & 0x70) >> 4) - 1)
    out[i] = sign ? -sample : sample
  }
  return out
}

export default function Simulator() {
  const [config, setConfig] = useState<Record<string, any> | null>(null)
  const [mode, setMode] = useState<'voice' | 'text'>('text')
  const [connected, setConnected] = useState(false)
  const [state, setState] = useState('idle')
  const [language, setLanguage] = useState<string>('')
  const [lines, setLines] = useState<Line[]>([])
  const [events, setEvents] = useState<Ev[]>([])
  const [draft, setDraft] = useState('')
  const [listening, setListening] = useState(false)
  const [level, setLevel] = useState(0)
  const [error, setError] = useState('')
  const [callId, setCallId] = useState('')
  const [showPad, setShowPad] = useState(false)
  const [partial, setPartial] = useState('')

  const wsRef = useRef<WebSocket | null>(null)
  const recRef = useRef<AnyRec>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const nextPlayRef = useRef(0)
  const idRef = useRef(1)
  const transcriptRef = useRef<HTMLDivElement | null>(null)
  const speakingRef = useRef(false)

  const push = useCallback((who: Line['who'], text: string, tag?: string) => {
    if (!text) return
    setLines((prev) => [...prev.slice(-120), { id: idRef.current++, who, text, tag }])
  }, [])

  const log = useCallback((kind: string, detail: string) => {
    setEvents((prev) => [...prev.slice(-240), { t: Date.now() / 1000, kind, detail }])
  }, [])

  useEffect(() => {
    api.simulatorConfig().then(setConfig).catch((e) => setError(String(e.message || e)))
  }, [])

  useEffect(() => {
    const el = transcriptRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines, partial])

  /** Tell the server the browser finished speaking so it can take the next turn. */
  const sendSpeechDone = useCallback(() => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'control', name: 'speech_done' }))
    }
  }, [])

  const speak = useCallback(
    (text: string, lang: string) => {
      if (mode !== 'voice') return
      const synth = window.speechSynthesis
      if (!synth) {
        sendSpeechDone()
        return
      }
      synth.cancel()
      const utter = new SpeechSynthesisUtterance(text)
      utter.lang = lang || 'en-IN'
      utter.rate = 1.02
      const voices = synth.getVoices()
      const match =
        voices.find((v) => v.lang === lang) ||
        voices.find((v) => v.lang?.startsWith(String(lang).split('-')[0])) ||
        voices.find((v) => v.lang?.startsWith('en-IN'))
      if (match) utter.voice = match
      speakingRef.current = true
      utter.onend = () => {
        speakingRef.current = false
        sendSpeechDone()
      }
      utter.onerror = () => {
        speakingRef.current = false
        sendSpeechDone()
      }
      synth.speak(utter)
    },
    [mode, sendSpeechDone],
  )

  /** Play server-synthesised µ-law audio (used when a cloud TTS provider is on). */
  const playMulaw = useCallback(
    (b64: string, sampleRate: number) => {
      if (mode !== 'voice') return
      try {
        const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
        const pcm = mulawToPcm16(raw)
        const Ctx = window.AudioContext || (window as any).webkitAudioContext
        if (!Ctx) return
        if (!audioCtxRef.current) audioCtxRef.current = new Ctx()
        const ctx = audioCtxRef.current
        if (ctx.state === 'suspended') void ctx.resume()
        const buf = ctx.createBuffer(1, pcm.length, sampleRate || 8000)
        const data = buf.getChannelData(0)
        for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768
        const src = ctx.createBufferSource()
        src.buffer = buf
        src.connect(ctx.destination)
        const start = Math.max(ctx.currentTime, nextPlayRef.current)
        src.start(start)
        nextPlayRef.current = start + buf.duration
        speakingRef.current = true
        src.onended = () => {
          speakingRef.current = false
          sendSpeechDone()
        }
      } catch (e) {
        log('audio_error', String(e))
        sendSpeechDone()
      }
    },
    [mode, sendSpeechDone, log],
  )

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis?.cancel()
    speakingRef.current = false
  }, [])

  /** Interrupt the assistant, exactly like a caller talking over it. */
  const bargeIn = useCallback(() => {
    const ws = wsRef.current
    stopSpeaking()
    if (audioCtxRef.current) {
      try {
        void audioCtxRef.current.close()
      } catch {
        /* already closed */
      }
      audioCtxRef.current = null
      nextPlayRef.current = 0
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'control', name: 'barge_in' }))
      log('barge_in', 'client interrupted playback')
    }
  }, [stopSpeaking, log])

  const startRecognition = useCallback(
    (lang: string) => {
      if (mode !== 'voice') return
      const Ctor = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
      if (!Ctor) {
        setError('This browser has no Web Speech API. Use text mode, or Chrome/Edge for voice.')
        return
      }
      try {
        recRef.current?.abort?.()
      } catch {
        /* ignore */
      }
      const rec: AnyRec = new Ctor()
      rec.lang = lang || 'en-IN'
      rec.continuous = true
      rec.interimResults = true
      rec.maxAlternatives = 1
      let stopped = false
      rec.onstart = () => setListening(true)
      rec.onerror = (e: any) => {
        log('asr_error', String(e?.error || e))
        if (e?.error === 'not-allowed') setError('Microphone permission denied.')
      }
      rec.onend = () => {
        setListening(false)
        // Keep listening for the whole call unless we hung up on purpose.
        if (!stopped && wsRef.current?.readyState === WebSocket.OPEN) {
          try {
            rec.start()
          } catch {
            /* restart raced with a stop */
          }
        }
      }
      rec.onresult = (e: any) => {
        let interim = ''
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const res = e.results[i]
          const text = String(res[0]?.transcript || '').trim()
          if (!text) continue
          if (res.isFinal) {
            if (speakingRef.current) bargeIn()
            const ws = wsRef.current
            if (ws?.readyState === WebSocket.OPEN) {
              ws.send(
                JSON.stringify({
                  type: 'transcript',
                  text,
                  is_final: true,
                  language: rec.lang,
                  confidence: Number(res[0]?.confidence || 0.8),
                }),
              )
            }
            push('me', text)
            setPartial('')
          } else {
            interim += text
          }
        }
        if (interim) {
          setPartial(interim)
          if (speakingRef.current) bargeIn()
        }
        setLevel(Math.min(100, 25 + Math.random() * 60))
      }
      recRef.current = rec
      try {
        rec.start()
      } catch {
        /* already started */
      }
      return () => {
        stopped = true
        try {
          rec.abort()
        } catch {
          /* ignore */
        }
      }
    },
    [mode, push, log, bargeIn],
  )

  const startCall = useCallback(() => {
    setError('')
    setLines([])
    setEvents([])
    setPartial('')
    const ws = simulatorSocket()
    wsRef.current = ws
    ws.onopen = () => {
      setConnected(true)
      ws.send(JSON.stringify({ type: 'start', mode, from: '+919000000000', to: '+911800120102' }))
      log('ws_open', `mode=${mode}`)
      if (mode === 'voice') {
        push('sys', 'Voice mode: allow microphone access when the browser asks.')
        startRecognition('en-IN')
      }
    }
    ws.onclose = () => {
      setConnected(false)
      setListening(false)
      setState('ended')
      log('ws_close', '')
    }
    ws.onerror = () => {
      setError('WebSocket error — is the backend running on this origin?')
      setConnected(false)
    }
    ws.onmessage = (msg) => {
      let m: any
      try {
        m = JSON.parse(String(msg.data))
      } catch {
        return
      }
      const kind = String(m.type || 'event')
      log(kind, JSON.stringify({ ...m, type: undefined }).slice(0, 160))
      switch (kind) {
        case 'call_started':
          setState(m.state || 'started')
          setCallId(String(m.call_id || ''))
          push('sys', `Call connected · ${m.call_id || ''}`)
          break
        case 'state':
          setState(String(m.to || ''))
          break
        case 'assistant_text':
          push('bot', String(m.text || ''), m.language ? String(m.language) : undefined)
          break
        case 'assistant_speech':
          // The browser voices it, then reports back so the turn can continue.
          if (m.mode === 'client_tts') speak(String(m.text || ''), String(m.language || 'en-IN'))
          break
        case 'audio':
          playMulaw(String(m.payload || ''), Number(m.sample_rate || 8000))
          break
        case 'caller_transcript':
          break
        case 'partial_transcript':
          setPartial(String(m.text || ''))
          break
        case 'asr_config': {
          const lang = String(m.language || (m.candidates && m.candidates[0]) || '')
          if (lang) startRecognition(lang)
          break
        }
        case 'language_detected':
        case 'language_switched':
          push(
            'sys',
            `Language: ${m.name || m.language || ''} (${m.method || 'detected'}${
              m.confidence ? ` · ${Math.round(Number(m.confidence) * 100)}%` : ''
            })`,
          )
          setLanguage(String(m.language || ''))
          if (m.language) startRecognition(String(m.language))
          break
        case 'followup_offer':
          push(
            'sys',
            `Offered by ${m.channel || 'sms'}: ${m.title || ''} (${(m.items || []).length} items)`,
          )
          break
        case 'barge_in':
          push('sys', 'Barge-in: assistant stopped speaking')
          break
        case 'silence':
          push('sys', 'Silence detected')
          break
        case 'unsupported_language':
          push('sys', `Unsupported language requested: ${m.name || m.language}`)
          break
        case 'dtmf_ignored':
        case 'dtmf_unmapped':
          push('sys', `Keypad input not accepted (${m.digit || ''})`)
          break
        case 'tts_error':
          push('sys', `TTS error: ${m.message || ''}`)
          break
        case 'error':
          setError(String(m.message || 'server error'))
          break
        case 'call_ended':
          setState('ended')
          setConnected(false)
          stopSpeaking()
          push(
            'sys',
            `Call ended · ${m.status || ''}${m.end_reason ? ` · ${m.end_reason}` : ''} · ${Math.round(
              Number(m.duration_seconds || 0),
            )}s${m.escalated ? ' · escalated' : ''}`,
          )
          break
        default:
          break
      }
    }
  }, [mode, push, log, speak, playMulaw, startRecognition, stopSpeaking])

  const endCall = useCallback(() => {
    const ws = wsRef.current
    stopSpeaking()
    try {
      recRef.current?.abort?.()
    } catch {
      /* ignore */
    }
    setListening(false)
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'hangup' }))
      setTimeout(() => ws.close(), 400)
    }
  }, [stopSpeaking])

  useEffect(
    () => () => {
      try {
        recRef.current?.abort?.()
      } catch {
        /* ignore */
      }
      window.speechSynthesis?.cancel()
      wsRef.current?.close()
    },
    [],
  )

  const sendText = useCallback(
    (text: string) => {
      const ws = wsRef.current
      const value = text.trim()
      if (!value || !ws || ws.readyState !== WebSocket.OPEN) return
      if (speakingRef.current) bargeIn()
      ws.send(JSON.stringify({ type: 'text', text: value }))
      push('me', value)
      setDraft('')
    },
    [push, bargeIn],
  )

  const sendDtmf = useCallback(
    (digit: string) => {
      const ws = wsRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      ws.send(JSON.stringify({ type: 'dtmf', digit }))
      push('me', `⌨ ${digit}`, 'dtmf')
    },
    [push],
  )

  const overrideLanguage = useCallback(
    (code: string) => {
      const ws = wsRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN || !code) return
      ws.send(JSON.stringify({ type: 'language_override', language: code }))
      setLanguage(code)
      startRecognition(code)
      push('sys', `Language forced to ${code}`)
    },
    [push, startRecognition],
  )

  const inCall = connected && state !== 'ended'
  const languages: string[] = config?.supported_languages || []

  return (
    <div className="sim-layout">
      <div className="phone">
        <div className="phone-head">
          <div className="phone-avatar">S</div>
          <div>
            <div className="who">Saarthi · NMIMS Global University, Dhule</div>
            <div className="meta">
              {config?.helpline || '+91 1800 102 5138'} · state <b>{state}</b>
              {language ? ` · ${language}` : ''}
            </div>
          </div>
          <div className="spacer" />
          <span className={`pill ${inCall ? 'ok' : state === 'ended' ? 'warn' : ''}`}>
            {inCall ? 'live' : state === 'ended' ? 'ended' : 'idle'}
          </span>
        </div>

        <div className="transcript" ref={transcriptRef}>
          {lines.length === 0 && (
            <div className="empty">
              Place a call to hear the greeting, pick a language by speaking, and ask about courses,
              fees, admissions or campus life.
            </div>
          )}
          {lines.map((l) => (
            <div key={l.id} className={`bubble ${l.who}`}>
              {l.tag ? <span className="tag">{l.tag}</span> : null}
              {l.text}
            </div>
          ))}
          {partial ? (
            <div className="bubble me" style={{ opacity: 0.6 }}>
              <span className="tag">listening…</span>
              {partial}
            </div>
          ) : null}
        </div>

        {mode === 'voice' && (
          <div className="meter">
            <i style={{ width: `${listening ? level : 0}%` }} />
          </div>
        )}

        {showPad && (
          <div className="dialpad">
            {['1', '2', '3', '4', '5', '6', '7', '8', '9', '*', '0', '#'].map((d) => (
              <button key={d} onClick={() => sendDtmf(d)}>
                {d}
              </button>
            ))}
          </div>
        )}

        <div className="row mt">
          <input
            value={draft}
            placeholder={inCall ? (mode === 'voice' ? 'Speak, or type to send…' : 'Type what the caller says…') : 'Start a call first'}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') sendText(draft)
            }}
            disabled={!inCall}
          />
          <button className="btn primary" disabled={!inCall || !draft.trim()} onClick={() => sendText(draft)}>
            Send
          </button>
        </div>

        <div className="call-controls">
          {!inCall ? (
            <button className="btn call" onClick={startCall}>
              ▶ Place call
            </button>
          ) : (
            <button className="btn hangup" onClick={endCall}>
              ■ Hang up
            </button>
          )}
          <button onClick={bargeIn} disabled={!inCall} title="Interrupt the assistant">
            ✋ Barge in
          </button>
          <button onClick={() => setShowPad((v) => !v)} disabled={!inCall} title="Keypad (last resort only)">
            ⌨
          </button>
        </div>

        <div className="row mt">
          <select value={mode} onChange={(e) => setMode(e.target.value as 'voice' | 'text')} disabled={inCall}>
            <option value="text">Text mode (no mic)</option>
            <option value="voice">Voice mode (mic + speech)</option>
          </select>
          <select value={language} onChange={(e) => overrideLanguage(e.target.value)} disabled={!inCall}>
            <option value="">Force language…</option>
            {languages.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        {callId ? <div className="small muted mt mono">call_id {callId}</div> : null}
      </div>

      <div className="grid" style={{ gap: 16 }}>
        {error ? <div className="error-banner">{error}</div> : null}

        <div className="card">
          <h3>Try a question</h3>
          <p className="sub">
            Sent as the caller's next turn. Ask something not in the knowledge base to see the
            “I don't have that info, connecting you to a human” path.
          </p>
          <div className="row">
            {QUICK_QUESTIONS.map((q) => (
              <button key={q} className="btn sm" disabled={!inCall} onClick={() => sendText(q)}>
                {q}
              </button>
            ))}
          </div>
        </div>

        <div className="card">
          <h3>Providers in use</h3>
          <div className="grid cols-4">
            {[
              ['ASR', config?.server_asr],
              ['TTS', config?.server_tts],
              ['LLM', config?.llm],
              ['LID', config?.lid],
            ].map(([label, value]) => (
              <div key={String(label)}>
                <div className="label muted small">{label}</div>
                <div style={{ fontWeight: 700 }}>{String(value || '—')}</div>
              </div>
            ))}
          </div>
          <p className="small muted mt">
            {config?.tts_text_only
              ? 'No server TTS key configured — the browser speaks the text (client TTS).'
              : 'Server-side TTS is configured; audio is streamed to the browser.'}{' '}
            DTMF fallback: {config?.dtmf_fallback ? 'enabled' : 'disabled'} · academic year{' '}
            {config?.academic_year || '—'}
          </p>
        </div>

        <div className="card">
          <h3>Session events</h3>
          <div className="event-log">
            {events.length === 0 ? <div className="empty">No events yet.</div> : null}
            {events
              .slice()
              .reverse()
              .map((e, i) => (
                <div key={i}>
                  <span className="t">{new Date(e.t * 1000).toLocaleTimeString()}</span>
                  <span className="k">{e.kind}</span>
                  <span className="muted">{e.detail}</span>
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  )
}
