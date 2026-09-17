# Compliance — TRAI and DPDP Act 2023

This document maps Indian regulatory obligations for an AI voice helpline onto
**controls that are actually implemented in this codebase**, with the setting or
module that enforces each one.

> **Not legal advice.** This is engineering guidance for the team operating the
> assistant. Have the university's legal/compliance officer review the wording of
> the consent announcement, the retention schedule and the grievance process
> before go-live, and re-review when the DPDP Rules are amended.

---

## 1. Recording consent and disclosure

**Obligation.** A caller must be told the call may be recorded, be told they are
speaking to an automated system, and be given a way to object — before any of
their speech is captured or stored.

**Implementation.**

- The greeting sequence speaks, in this order, before the caller says anything
  substantive (`app/orchestrator/session.py::_greeting_sequence`):
  1. **Identity + AI disclosure** — "Thank you for calling NMIMS Global
     University, Dhule. You are speaking with Saarthi, an AI assistant."
  2. **Recording notice + opt-out** — "For quality and training, this call may be
     recorded. If you prefer, you can ask us not to record."
- The notice is announced once per call and the outcome is stored on the call
  record (`calls.recording_consent`), so consent is *auditable per call*, not
  merely asserted in a policy document.
- `RECORDING_CONSENT_ANNOUNCE=true` (default) controls the announcement.
- **Recording is off by default**: `CALL_RECORDING_ENABLED=false` produces no
  audio; `ALLOW_CALL_RECORDING_STORAGE=false` makes the telephony status webhook
  **discard** any `RecordingUrl` a provider sends, so audio cannot accumulate by
  accident.
- The TwiML builders expose `recording_consent_twiml()` for providers that need
  the disclosure inside the call flow itself.
- If a caller asks not to be recorded, that is handled as a caller instruction and
  recorded as an objection; the operational procedure is to stop recording and
  note it on the call — and the assistant can escalate to a human who can continue
  without recording.

**Operational checklist.** Confirm your telephony provider's recording behaviour
matches these flags, confirm the hold-music/queue path also carries the disclosure
when a caller waits before transfer, and keep the consent text under version
control (it lives in `app/i18n/languages.py`, not in a spreadsheet).

---

## 2. AI disclosure (no deception)

**Obligation.** Callers must not be misled into believing they are speaking to a
human, and must be able to reach a human.

**Implementation.**

- The greeting names the assistant and states it is an AI (`ASSISTANT_NAME =
  "Saarthi"`, `app/ai/prompts.py`).
- A request for a person is treated as a first-class intent: `human_request`
  (English, Hindi, Marathi and Rajasthani phrasings, confidence threshold 0.45)
  triggers
  **immediate escalation** without the caller having to navigate anything.
- A harm, grievance, legal or distress topic is escalated by guardrail
  (`sensitive_or_legal`) even if the caller does not ask for a human — an AI must
  not handle a harassment complaint, a legal threat or a mental-health disclosure.
- When the KB has no verified answer, the assistant says it does not have that
  information and offers a human rather than improvising.
- Escalation never dead-ends: if no agent is available the caller is given the
  admissions helpline number and the reason is logged (`no_agent:*`).

---

## 3. DPDP Act 2023 — data protection

The university is the **Data Fiduciary**; callers are **Data Principals**.

### 3.1 Notice and consent
- Purpose is stated at the start of the call (admissions help + quality/training
  recording), in plain language, before personal data is processed.
- Consent for the *follow-up* channel is asked explicitly and separately: the
  assistant offers SMS/WhatsApp/email and only captures a destination after the
  caller agrees (`followup_offer` → `followup_destination` states). Nothing is
  sent to a number the caller did not agree to in that call.
- Consent is per-call and recorded (`calls.recording_consent`, `follow_ups`).

### 3.2 Data minimisation
- Only what the call needs is stored: transcript text, language, timings,
  citations, escalation reason, outcome.
- **Caller numbers are hashed** (`redacted_caller`) before they reach the call
  record, so the log cannot be used as a call-back list.
- **PII is redacted and hashed before it is written to the transcript**
  (`REDACT_PII=true`, `app/ai/guardrails.py::redact_pii`): email, URL, PAN, phone,
  landline, Aadhaar and card numbers are replaced with typed placeholders
  (`[phone:*******210]`) and only a salted hash is retained for correlation. A
  caller who recites an Aadhaar number does not have it stored in the clear.
- No payment data is collected by the assistant; fee *disputes* and refunds are
  routed to the accounts office by policy, not processed.

### 3.3 Purpose limitation
- Transcripts are used for QA, analytics and escalation hand-off. The analytics
  endpoints aggregate; the QA views are behind admin auth
  (`ADMIN_AUTH_ENABLED=true` in production).
- Call audio is not stored unless explicitly enabled (§1).

### 3.4 Accuracy
- KB rows carry `verified`, `verified_by`, `source`, `source_uri` and
  `academic_year`; unverified and stale content is ranked down and hedged, so the
  system does not assert outdated personal-impact facts (fees, deadlines).
- Every KB edit is versioned in `kb_revisions` — an audit trail for content that
  callers relied on.

### 3.5 Storage limitation / erasure
- `MAX_CALL_MINUTES` caps call length; `SILENCE_MAX_REPROMPTS` ends dead calls.
- Call records, turns and escalations cascade-delete with the call
  (`ondelete="CASCADE"`), so a documented erasure request can be executed with a
  single delete.
- **Set a retention schedule** (recommended: transcripts 90 days, aggregates
  longer, audio never unless enabled) and enforce it with a scheduled purge — the
  schema supports it; the policy decision is yours. `POST /api/kb/purge` covers KB
  content; add a cron job for `calls` if you want automated expiry.

### 3.6 Security
- Secrets only via environment (`APP_SECRET`, provider keys); `.env` is
  git-ignored, `.env.example` ships with no real values.
- Admin/dashboard endpoints sit behind `require_admin`; disable auth only for
  local development.
- Telephony webhooks should be signature-verified and reachable only over HTTPS;
  `PUBLIC_BASE_URL` must be `wss://`/`https://` in production.
- Structured JSON logging with request correlation; no credentials in logs.

### 3.7 Grievance redressal and data-principal rights
- The guardrail treats a **complaint** (`complaint`, `शिकायत`, legal/court
  vocabulary) as a sensitive topic and escalates it to a human — a grievance is
  never answered by the AI.
- Requests for access, correction, erasure or nomination must be handled by the
  university's Grievance Redressal Officer; the call log and transcript store
  give you the records needed to respond within the statutory timeline.
- **Children's data:** admissions enquiries frequently concern minors. Do not
  enable profiling, targeting or marketing use of call data; the system stores no
  behavioural profile and no ad-tech identifiers, and follow-up messages are
  service communications requested in-call. Verifiable parental consent applies to
  any processing beyond the enquiry itself.

---

## 4. TRAI — telecom obligations

- **Inbound-only by design.** The assistant answers calls to the university
  helpline; it does not place outbound marketing calls, so DND/`do-not-call`
  scrubbing for telemarketing does not apply.
- **Follow-up SMS/WhatsApp/email are service communications triggered by the
  caller in that call**, after an explicit spoken offer and agreement. They are
  not unsolicited commercial communication. Keep them transactional in content
  (the details the caller asked for), send them from a registered
  sender-id/template as your aggregator requires, and honour an in-call refusal.
  Controlled by `FOLLOWUP_SMS_ENABLED`, `FOLLOWUP_WHATSAPP_ENABLED`,
  `FOLLOWUP_EMAIL_ENABLED`.
- **No keypad coercion.** Language selection is by speech; DTMF exists only as a
  documented fallback when speech recognition repeatedly fails
  (`ALLOW_DTMF_FALLBACK`, off by default), which avoids the TRAI-unfriendly
  pattern of forcing callers through long IVR trees.
- **Number display and identity.** Ensure the published helpline number and the
  caller-ID presented on transfer are the university's own registered numbers.
- **Recording.** TRAI/DoT expectations on call recording align with §1: disclose,
  allow objection, minimise retention.

---

## 5. Configuration baseline for production

```ini
ENVIRONMENT=production
ADMIN_AUTH_ENABLED=true
ADMIN_USERNAME=<unique>
ADMIN_PASSWORD=<strong, rotated>
APP_SECRET=<openssl rand -hex 32>
PUBLIC_BASE_URL=https://voice.dhule.nmims.edu

RECORDING_CONSENT_ANNOUNCE=true
CALL_RECORDING_ENABLED=false
ALLOW_CALL_RECORDING_STORAGE=false
REDACT_PII=true
MAX_CALL_MINUTES=15

ESCALATION_AGENTS=+911412345678,+911418765432
ESCALATION_MAX_WAIT_SECONDS=120
ESCALATION_WHISPER_CONTEXT=true

DATABASE_URL=postgresql+psycopg://nims:<password>@db:5432/nims_voice
```

Verify before go-live:

- [ ] Consent announcement text reviewed and signed off by legal.
- [ ] A real call announces AI identity + recording + opt-out before the caller speaks.
- [ ] `recording_consent` is populated on call records.
- [ ] No audio is stored (`CALL_RECORDING_ENABLED=false`) or storage is justified,
      access-controlled and time-limited.
- [ ] PII redaction confirmed on a test call where the caller reads out a phone
      number and an email address.
- [ ] "Talk to a person" escalates within one turn, in the caller's language.
- [ ] A harassment/legal/complaint utterance escalates without answering.
- [ ] Unverified fee rows are corrected, sourced and marked verified.
- [ ] Retention schedule documented and automated.
- [ ] Grievance Redressal Officer named, and complaints from calls reach them.
- [ ] Admin dashboard is not publicly reachable without credentials.
- [ ] Webhook endpoints are HTTPS and signature-verified.
