# 📒 Khata Copilot

An agentic digitizer for a small Indian shopkeeper's handwritten **khata**
(udhaar/bahi ledger). Photograph a page, and a 5-agent pipeline reads it,
audits its own reading, stores it, answers questions about it, and drafts
payment reminders — all with honest confidence scores and zero invented
numbers.

Built for the **NIAT TakeOver'26** hackathon (~7-day build).

## What it does

A shopkeeper photographs a ledger page. Khata Copilot:

1. **Reads** the handwriting into structured entries (name, amount, date, status).
2. **Audits** its own reading — checks the math, flags implausible amounts,
   flags anything it isn't confident about. It never silently "fixes" a bad read.
3. **Stores** verified entries in a searchable ledger, deduplicating re-scans.
4. **Answers** plain-language questions ("Who owes the most?", "Kitna baki hai?")
   grounded in the actual data, with citations.
5. **Drafts** short bilingual (Hindi/English) payment reminders — copy-paste
   ready, never auto-sent.

## The 5-agent pipeline

```
Photo ──▶ Vision ──▶ Verification ──▶ Ledger ──▶ Insights ──▶ Reminder
          (read)      (audit, flag)    (store)    (Q&A, RAG)   (draft)
```

| Agent | File | Job |
|---|---|---|
| **Vision** | `agents/vision_agent.py` | Reads a photo into a `PageExtraction` (Gemini 2.0 Flash → Groq vision fallback → mock, see below). |
| **Verification** | `agents/verification_agent.py` | Pure-Python audit: math mismatch, absurd amounts, missing fields, low confidence. **Flags, never fixes.** Can ask Vision for one bounded re-read if it suspects a whole-page misread. |
| **Ledger** (`core/db.py`) | — | SQLite storage. Dedup keyed on (customer, amount, date, source image) so re-scanning a page updates rather than duplicates. |
| **Insights** | `agents/insights_agent.py` | Hybrid FAISS + BM25 search over ledger entries. Deterministic totals are computed in Python and injected into the LLM's context — the LLM phrases the answer, never recalculates it. A guardrail refuses out-of-scope questions. |
| **Reminder** | `agents/reminder_agent.py` | Drafts a bilingual reminder per customer with an outstanding balance. The amount is always the exact deterministic figure; the LLM only writes the wording. |

`core/orchestrator.py` runs Vision → Verification → Ledger sequentially for
one scanned page, and never raises — every stage degrades gracefully into a
flagged, explainable result instead of crashing the demo.

## Anti-hallucination design

This is the core idea of the project, not an afterthought:

- **Deterministic numbers.** Every total, balance, and reminder amount is
  computed in Python straight from the database. An LLM never does arithmetic
  that ends up on screen — it only phrases sentences around numbers it's handed.
- **Source tags.** Every Insights answer is tagged with where it came from:
  `deterministic` (pure SQL), `llm` (grounded RAG), `extractive_fallback`
  (no LLM available), `guardrail_refusal`, or `template` (Reminder agent
  without a live Groq call). Judges can see exactly how confident to be.
- **Self-flagging, not self-fixing.** The Verification Agent's whole job is to
  find problems and flag them for human review — it never silently edits a
  number to make totals balance. A flagged low-confidence entry is the
  feature, not a bug.
- **Graceful degradation everywhere.** Every agent is designed to return a
  usable, honestly-labeled result even when Gemini/Groq are down, rate-limited,
  or absent — see the mock-mode section below.

## Tech stack (all free tier)

- **Python 3.10**, **Pydantic v2** schemas as the strict typing gate between agents
- **Gemini 2.0 Flash** (free tier) — primary handwriting reader
- **Groq `llama-3.3-70b-versatile`** — reasoning (Insights phrasing, Reminder drafting, guardrail)
- **Groq vision** (`llama-4-scout-17b-16e-instruct`, configurable) — vision **fallback only** when Gemini is unavailable/quota-exhausted; honestly weaker on messy handwriting than Gemini, and says so in its notes
- **SBERT `all-MiniLM-L6-v2` + FAISS** — local dense embeddings for hybrid search
- **`rank-bm25`** — sparse keyword half of the hybrid search fusion
- **SQLite** (stdlib) — structured ledger storage
- **Streamlit** — 4-tab UI (Scan / Ledger / Insights / Reminders)
- Hand-rolled sequential orchestrator (deliberately not a framework like CrewAI)

## How to run it

```powershell
# From the project root, using the existing .venv (note: quote the path — it has a space)
"C:\Khata Copilot\.venv\Scripts\python.exe" -m streamlit run app/streamlit_app.py
```

Copy `.env.example` to `.env` and fill in `GEMINI_API_KEYS` / `GROQ_API_KEYS`
for live mode (multiple comma-separated keys are round-robin rotated with
per-key cooldown on rate limits).

**No keys? No problem.** Set `KHATA_MOCK=1` (or just leave the keys blank) and
the whole app — Scan, Ledger, Insights, Reminders — runs fully offline on a
canned, clearly-labeled synthetic page. Use the **"🎬 Load Demo Data"** button
in the sidebar to instantly seed a clean, known set of ledger entries so you
can demo Insights and Reminders without scanning anything.

### Vision reader order

Gemini (primary) → Groq vision (fallback, only tried if Gemini is unavailable
or quota-errors) → mock (last resort, only if both real readers fail). This
means a Gemini free-tier quota error no longer forces the whole page into mock
mode — the demo keeps reading real photos through Groq instead.

## Tests

29 tests from Phases 1–5, plus 5 new Phase 7 tests for the Groq vision
fallback ordering — **34 passing**.

```powershell
"C:\Khata Copilot\.venv\Scripts\python.exe" -m pytest -v
```

All tests run fully offline (no real API calls) — Gemini/Groq calls are
either not exercised (pure-Python audit/storage logic) or monkeypatched at
the module boundary.
