# Knowledge Base — guide for admissions staff

The knowledge base (KB) is the **only** source the assistant answers from. If a
fact is not in the KB, the assistant says so and connects the caller to a human —
it never guesses. That is why keeping the KB current is the single most important
operational task, and why it requires **no developer and no redeploy**.

You can work entirely from the dashboard: **Knowledge base** tab.

---

## The 5-minute workflow

1. **Find** — search by title, course or keyword; filter by category, language or
   verification status.
2. **Edit** — change the answer text, fee figures, dates, eligibility, seats.
3. **Verify** — tick *verified* when the content matches an official document
   (prospectus, fee handbook, admission notification). Add your name in
   *verified by*.
4. **Save** — the record is re-chunked and re-embedded immediately. Live calls
   pick it up on their next question; nothing restarts.
5. **Check** — ask the question in the **Phone** tab and read the answer plus its
   citations. If the citation is the record you just edited, you are done.

---

## Anatomy of a record

| Field | What to put | Notes |
|---|---|---|
| `slug` | Stable id, e.g. `course-btech-cse` | Changing it creates a new record |
| `category` | `course`, `fees`, `eligibility`, `admission_process`, `important_dates`, `documents`, `entrance_exam`, `scholarships`, `hostel`, `facilities`, `placements`, `transport`, `contact`, `loan_payment`, `university`, `faq`, `policy`, `specialisation` | Drives retrieval; pick the closest |
| `title` | What a human would call it | Spoken in answers and shown in citations |
| `aliases` | Other ways callers say it: `"बीटेक"`, `"BTech CSE"`, `"computer science"` | Big retrieval win for code-mixed callers |
| `academic_year` | `2026-27` | Stale years are ranked down and flagged |
| `language` | `en-IN`, `hi-IN`, `mr-IN`, … | Answers are matched to the caller's language |
| `verified` / `verified_by` | `true` + your name | **Unverified content is ranked lower and the assistant hedges or escalates** |
| `source` / `source_uri` | The official document and a link | Required for audit; shown to QA staff |
| `structured` | Typed facts: `annual_fee`, `total_fee`, `eligibility`, `minimum_marks`, `entrance_exam`, `duration_years`, `seats`, `documents_required`, `important_dates`, … | These feed the deterministic answer templates — exact numbers must live here |
| `body` | Free prose | Chunked for retrieval; good for policies and processes |
| `tags` | Free-form labels | Filtering and reporting |

**Rule of thumb:** exact numbers (fees, dates, seat counts, cut-offs) belong in
`structured` fields; explanations belong in `body`. The assistant quotes
`structured` values verbatim, which is what the numeric grounding check verifies.

### Example

```yaml
- slug: course-btech-cse
  category: course
  title: "B.Tech Computer Science and Engineering"
  aliases: ["BTech CSE", "computer science", "सीएसई", "बीटेक सीएसई"]
  academic_year: "2026-27"
  language: en-IN
  verified: true
  verified_by: "Admissions Office"
  source: "Fee and Admission Handbook 2026-27"
  structured:
    degree: "B.Tech"
    level: UG
    duration_years: 4
    seats: 180
    annual_fee: 150000
    total_fee: 600000
    eligibility: "10+2 with Physics, Chemistry and Mathematics, 50% aggregate"
    entrance_exam: "NMIMS-NPAT or JEE Main"
  body: |
    Four-year undergraduate programme covering algorithms, systems, AI and
    electives, with internship in the seventh semester.
```

---

## Bulk editing

### CSV import (recommended for fee tables and course lists)
1. Download the template: **Knowledge base → CSV template** (or
   `GET /api/kb/template.csv`). It has every column with one worked example row.
2. Fill it in Excel/Google Sheets. Keep `slug` stable to *update*; leave it blank
   to *create*.
3. **Preview (dry run)** first — the dashboard shows exactly which records would
   be created, updated or skipped, and the chunking preview shows how each record
   will be split for retrieval.
4. Import. The KB is re-indexed automatically.

Multi-value cells use `;` (e.g. `aliases` = `BTech CSE;Computer Science;सीएसई`).
Fees are plain numbers in rupees — no `₹`, no commas, no "per year" text; the
assistant renders them into spoken form itself.

### Google Sheets sync
Set `KB_GOOGLE_SHEET_CSV_URL` to a sheet published as CSV, then use **Sync sheet**
in the dashboard (or `POST /api/kb/sync-sheet`). With `KB_SYNC_INTERVAL_MINUTES`
set, the sync also runs on a schedule — useful when the admissions team works in
Sheets and never wants to touch the dashboard.

### YAML / JSON
`data/kb/*.yaml` is the seed set, re-ingested on every boot. Ingestion is
**idempotent by content hash**: unchanged records are skipped, so a restart never
creates duplicates or spurious revision history.

### Export
`GET /api/kb/export.csv` (or the dashboard button) exports the whole KB for
backup, review or handover.

---

## Verification workflow (important)

The seeded data ships with **almost every record marked `verified: false`** and a
`source` naming where the fact came from, because the only public sources for
this campus are aggregator listings and the university's own web pages, and they
disagree. Where no figure could be sourced at all — engineering, pharmacy,
BCA/MCA, MBA, M.Com and Ph.D. fees — the number is **left out on purpose**, so
the assistant escalates instead of inventing one. Only the escalation-boundary
policy record ships verified, because it describes the assistant's own behaviour
rather than a fact about the university.

Everything else needs a human to check it against the official handbook and tick
`verified` in the dashboard: contact details, the programme lists, eligibility
percentages, the indicative fee totals, and the admission timeline.

Unverified content is treated as provisional:
- ranked below verified content in retrieval;
- surfaced with a hedge, or escalated to a human, rather than asserted;
- listed in the dashboard's *unverified* filter so you can work through it.

**Before go-live:** filter `verified = false`, check each record against the
official handbook, correct it, tick *verified*, add your name. Use **Bulk verify**
once a whole category has been checked against a single source document.

Every edit is stored in `kb_revisions` (who/what/when), so a mistake can be
traced and reverted.

---

## Admission-cycle updates

A new cycle is normally: new academic year, new fees, new dates, new seat counts,
occasionally new programmes.

1. Export the current KB (backup).
2. Update the fee/date/seat fields — CSV import is fastest for a full fee table.
3. Set `academic_year` to the new cycle on every affected record.
4. Verify the changed records.
5. Reindex (automatic on import; **Reindex** button if you edited YAML on disk).
6. Smoke-test the top questions in the **Phone** tab: fee for the biggest
   programmes, last date to apply, eligibility, documents, hostel, scholarships.
7. Check **Analytics → Unanswered** afterwards: questions callers asked that the
   KB could not answer are your backlog of content to add.

`KB_STALENESS_DAYS` controls when a record is considered stale; stale records are
penalised in ranking so last cycle's dates stop outranking this cycle's.

---

## Writing answers that work well on a phone

The assistant shortens and speaks your content, so write for the ear:

- **One fact per sentence.** "The fee is ₹1,50,000 per year. The total programme
  cost is about ₹6,00,000."
- **Numbers as digits** in `structured` fields — the system converts them to
  spoken Indian form ("one lakh fifty thousand rupees").
- **No markdown, tables, bullets or URLs** in spoken text; they are stripped.
  Put links in `source_uri` and let the follow-up SMS/email carry them.
- **Short lists.** More than ~4 items is better sent by SMS/WhatsApp/email — the
  assistant offers that automatically when an answer runs long.
- **Spell out ambiguity.** "the NMIMS-NPAT entrance test" beats "NPAT", and
  "MHT-CET" should be written with its hyphen so TTS reads it as letters.
- **Hindi/Marathi content:** add Indic aliases in *both* scripts even on English
  records — a caller says "बीटेक" or "बीटेक"/"अभियांत्रिकी" while the record is
  titled "B.Tech". Marathi aliases matter most here: `"प्रवेश"`, `"शुल्क"`,
  `"वसतिगृह"`, `"अभ्यासक्रम"` retrieve nothing if only English and Hindi are
  listed.

---

## What happens when something is missing

The assistant will not improvise. It responds that it does not have verified
information and offers a human, and the question is written to the
**unanswered backlog** (`Analytics → Unanswered`). Work through that list each
week: it is a direct, prioritised signal of what callers actually want and the KB
does not yet contain. Resolve an entry from the dashboard once you have added the
content.

### Recording a verified absence

Some questions have a real answer, and that answer is *"the university publishes
nothing about this"*. Fees, hostel, refunds, education loans and payment modes are
all like that at Dhule today. Recording the absence is what stops the assistant
from either inventing a number or falling silent, so write it down explicitly:

```yaml
structured:
  published_fee_table: false      # or published: false / published_by_university: false
  where_the_fee_is_confirmed: "At registration on the SVKM admission portal"
  what_the_university_can_issue: "A bonafide certificate and the fee structure"
```

Any of those `false` flags does two things automatically:

1. The call **escalates to a human** with reason `not_published`, even when the
   sentence spoken was useful — a refund or a loan cannot be settled from an
   absence of data.
2. The caller hears a **native sentence in their own language** (English, Hindi,
   Marathi or Rajasthani) rather than an English paragraph, because these cases
   have their own frames.

Remove the flag the day the university publishes the figure, and put the number in
the matching field — the frames for fees, duration, seats and eligibility pick it
up with no code change.

### Write `body` text for the caller, not for staff

`body` is spoken aloud. Write it in the second person:

- **Yes:** "If you want to withdraw, the accounts office handles it, and it helps
  to have the payment details to hand."
- **No:** "A caller who wants to withdraw needs the accounts office."

Guidance for the assistant itself belongs in `assistant_instruction`,
`assistant_must_not` and `assistant_behaviour`, which are read by the model but
never rendered into speech. A test fails the build if a `body` refers to "a
caller" or "the caller".
