"""
Pydantic v2 schemas — the strict typing gate between agents.

Modeling style ported from project-overwatch/main_workflow/schemas.py:
  - Field(..., description=...) so the schema self-documents.
  - @field_validator(mode="before") to normalize messy LLM/OCR input.
  - @model_validator(mode="after") for cross-field defaults.
  - confidence as float in [0, 1].

Design decision (Phase 1, confirmed): amount is a float (rupees) and date is
the RAW string the model read. We do NOT parse dates or do exact-money math
here — the Verification Agent (Phase 2) audits and flags. Vision's job is to
report honestly, including how unsure it is.
"""
from __future__ import annotations

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Status tokens a shopkeeper might write, in Hindi (Latin + Devanagari) and
# English. Everything else falls through to "unknown" so it gets flagged.
_PAID_TOKENS = {
    "paid", "p", "pd", "cleared", "clear", "done", "received", "recd", "settled",
    "jama", "jma", "जमा", "chukta", "chuka", "चुकता", "ok", "✓", "✔",
}
_UNPAID_TOKENS = {
    "unpaid", "u", "due", "pending", "owes", "owe", "baki", "baaki", "बाकी",
    "udhaar", "udhar", "उधार", "udhaari", "credit", "cr", "balance", "left",
}

StatusLiteral = Literal["paid", "unpaid", "unknown"]

# ── Entry type: a SEPARATE axis from `status` (Phase 8, billing) ──────────────
# `status` answers "is it settled?" (paid/unpaid). `entry_type` answers "what
# KIND of line is this?" and never mixes with settlement state:
#   - "udhaar":  credit the shopkeeper extended (the original khata).
#   - "sale":    a bill / cash sale of goods.
#   - "unknown": type couldn't be determined -> flagged for review and EXCLUDED
#                from BOTH the udhaar and the sales rollups (the honesty rule).
# Defaults to "udhaar" everywhere: every row that predates billing IS udhaar by
# definition, so all existing data and queries keep their exact meaning.
EntryTypeLiteral = Literal["udhaar", "sale", "unknown"]


def _to_float_amount(value) -> float:
    """Coerce a messy amount (₹1,200/-, 'Rs 500', 1200.0) into a float.

    Tolerant by design: an unparseable amount becomes 0.0 rather than raising,
    so one bad row never discards a whole page. The Verification Agent flags
    zero/implausible amounts downstream.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    # Grab the first number: optional leading minus, digits with optional
    # thousands/lakh commas, optional decimals. This correctly ignores ₹, "Rs",
    # and a trailing "/-" (Indian rupee notation) that a naive strip would keep.
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(value))
    if not match:
        return 0.0
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return 0.0


class LedgerEntry(BaseModel):
    """One customer line from a khata page."""

    name: str = Field(..., description="Customer name exactly as written.")
    amount: float = Field(..., description="Transaction amount in rupees.")
    date: Optional[str] = Field(
        None, description="Date exactly as written on the page (raw string), or null if absent."
    )
    status: StatusLiteral = Field(
        "unknown", description="'paid', 'unpaid', or 'unknown' if not legible."
    )
    entry_type: EntryTypeLiteral = Field(
        "udhaar",
        description="'udhaar' (credit given), 'sale' (bill/cash sale), or 'unknown'. "
        "A SEPARATE axis from status; defaults to 'udhaar' (the pre-billing meaning).",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="How sure the reader is about THIS entry, 0-1. Be honest; low is fine.",
    )
    raw_text: str = Field(
        "", description="The original line text as read, for human cross-checking."
    )

    @field_validator("name", "raw_text", mode="before")
    @classmethod
    def _clean_text(cls, value):
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @field_validator("amount", mode="before")
    @classmethod
    def _normalize_amount(cls, value):
        return _to_float_amount(value)

    @field_validator("date", mode="before")
    @classmethod
    def _normalize_date(cls, value):
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text or None

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value):
        token = str(value or "").strip().lower()
        if token in _PAID_TOKENS:
            return "paid"
        if token in _UNPAID_TOKENS:
            return "unpaid"
        if token in ("paid", "unpaid", "unknown"):
            return token
        return "unknown"

    @field_validator("entry_type", mode="before")
    @classmethod
    def _normalize_entry_type(cls, value):
        # Minimal in this phase: accept the three canonical literals, default
        # anything empty/unrecognized to "udhaar". CHUNK 2 teaches the Vision
        # prompt to emit the richer synonyms (baaki/jama -> udhaar, bill/cash ->
        # sale); those canonical values already pass through here unchanged.
        token = str(value or "").strip().lower()
        return token if token in ("udhaar", "sale", "unknown") else "udhaar"

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value):
        try:
            c = float(value)
        except (TypeError, ValueError):
            return 0.0
        if c > 1.0:  # model gave a percentage like 85
            c = c / 100.0
        return max(0.0, min(1.0, c))


class PageExtraction(BaseModel):
    """The Vision Agent's full output for a single photographed page."""

    source_image: str = Field(..., description="Path/filename of the source photo.")
    entries: List[LedgerEntry] = Field(
        default_factory=list, description="All ledger lines read from the page."
    )
    written_total: Optional[float] = Field(
        None, description="A total figure written on the page itself, if any (for math auditing)."
    )
    overall_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Aggregate confidence for the whole page."
    )
    notes: str = Field(
        "", description="Reader notes: illegible regions, ambiguity, assumptions."
    )
    degraded: bool = Field(
        False, description="True when this came from mock mode or a failed/partial extraction."
    )
    error: Optional[str] = Field(
        None, description="Populated when extraction failed; the pipeline stays alive."
    )

    @field_validator("written_total", mode="before")
    @classmethod
    def _normalize_written_total(cls, value):
        if value is None or value == "":
            return None
        return _to_float_amount(value)

    @model_validator(mode="after")
    def _default_overall_confidence(self):
        # If the model didn't supply a page-level score, derive it from entries.
        if not self.overall_confidence and self.entries:
            self.overall_confidence = round(
                sum(e.confidence for e in self.entries) / len(self.entries), 3
            )
        return self

    @property
    def computed_total(self) -> float:
        """Sum of entry amounts — compared against written_total by Verification."""
        return round(sum(e.amount for e in self.entries), 2)

    def flagged_entries(self, threshold: float) -> List[LedgerEntry]:
        """Entries at or below the confidence threshold (need human review)."""
        return [e for e in self.entries if e.confidence <= threshold]


# ── Verification (Phase 2) ───────────────────────────────────────────────────
# The Verification Agent produces a VerificationResult ABOUT a PageExtraction.
# It never mutates the extraction — it points at problems for a human to resolve.

# Machine-readable tags for each kind of problem the auditor can find. Kept as a
# Literal so a typo becomes a validation error, and so the UI can switch on them.
IssueCode = Literal[
    "math_mismatch",       # written_total disagrees with the sum of entries
    "absurd_amount",       # amount too large to be a plausible single khata line
    "nonpositive_amount",  # amount <= 0
    "missing_name",        # name blank or literally "UNREADABLE"
    "missing_date",        # no date recorded
    "unknown_status",      # paid/unpaid could not be determined
    "low_confidence",      # reader was unsure about this entry
    "degraded_extraction", # page came from mock/failed extraction
]
SeverityLiteral = Literal["info", "warning", "error"]
VerdictLiteral = Literal["accept", "needs_review"]


class EntryIssue(BaseModel):
    """One problem the auditor found, tied to an entry (or the page itself)."""

    entry_index: int = Field(
        ..., description="Index into PageExtraction.entries, or -1 for a page-level issue."
    )
    entry_name: str = Field(
        "", description="Customer name for human readability ('' for page-level issues)."
    )
    code: IssueCode = Field(..., description="Machine-readable problem tag.")
    severity: SeverityLiteral = Field(..., description="'info', 'warning', or 'error'.")
    message: str = Field(..., description="Human-readable, judge-explainable description.")


class VerificationResult(BaseModel):
    """The audit report for a single page. Flags problems; never fixes data."""

    source_image: str = Field(..., description="Echoed from the audited PageExtraction.")
    verdict: VerdictLiteral = Field(
        "accept", description="'accept' if clean, else 'needs_review'."
    )
    issues: List[EntryIssue] = Field(
        default_factory=list, description="Every problem found, entry- and page-level."
    )
    overall_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Carried/derived page confidence, 0-1."
    )
    computed_total: float = Field(
        0.0, description="Sum of entry amounts (echoed for the UI)."
    )
    written_total: Optional[float] = Field(
        None, description="Total written on the page, if any (echoed for the UI)."
    )
    total_difference: Optional[float] = Field(
        None, description="written_total - computed_total when both exist, else null."
    )
    corrected: bool = Field(
        False, description="True if a self-correction re-extraction was attempted."
    )
    correction_note: str = Field(
        "", description="What the self-correction loop did, and why."
    )

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def issue_lines(self) -> List[str]:
        """Flat, human-readable issue list for the CLI / Streamlit review panel."""
        lines: List[str] = []
        for issue in self.issues:
            where = issue.entry_name or "page"
            lines.append(f"[{issue.severity.upper()}] {where}: {issue.message}")
        return lines


# ── Ledger (Phase 3) ─────────────────────────────────────────────────────────
# core/db.py returns these instead of raw sqlite rows, so downstream agents
# (Insights, Reminder) get the same typed-schema guarantee as everything else.

class CustomerBalance(BaseModel):
    """A customer's outstanding total, as queried from the ledger."""

    name: str = Field(..., description="Customer display name.")
    unpaid_total: float = Field(..., description="Sum of amounts where status != 'paid'.")
    paid_total: float = Field(..., description="Sum of amounts where status == 'paid'.")
    entry_count: int = Field(..., description="Total number of ledger entries for this customer.")


class LedgerEntryRecord(BaseModel):
    """A single stored ledger entry, as read back from SQLite."""

    id: int = Field(..., description="Row id in the entries table.")
    customer_name: str = Field(..., description="Customer display name.")
    amount: float = Field(..., description="Transaction amount in rupees.")
    raw_date: Optional[str] = Field(None, description="Date exactly as written, or null.")
    status: StatusLiteral = Field(..., description="'paid', 'unpaid', or 'unknown'.")
    entry_type: EntryTypeLiteral = Field(
        "udhaar", description="'udhaar', 'sale', or 'unknown' — a separate axis from status."
    )
    confidence: float = Field(..., description="Reader confidence for this entry, 0-1.")
    raw_text: str = Field("", description="Original line text as read.")
    source_image: str = Field(..., description="Source page image this entry came from.")
    needs_review: bool = Field(..., description="True if this entry was flagged by verification.")
    scanned_at: str = Field(..., description="Timestamp this entry was last stored/updated.")


class MonthlyTotal(BaseModel):
    """Sum of entry amounts for one calendar month (best-effort date parsing)."""

    month: str = Field(..., description="'YYYY-MM', or 'unknown' if the raw date couldn't be parsed.")
    total: float = Field(..., description="Sum of amounts in this bucket.")
    entry_count: int = Field(..., description="Number of entries in this bucket.")


class IngestSummary(BaseModel):
    """What happened when a verified page was ingested into the ledger."""

    source_image: str = Field(..., description="Page that was ingested.")
    inserted: int = Field(0, description="Number of new entries inserted.")
    updated: int = Field(0, description="Number of existing entries updated (re-scan).")
    page_verdict: VerdictLiteral = Field(..., description="Echoed verdict from VerificationResult.")


# ── Billing / Sales (Phase 8) ────────────────────────────────────────────────
# Sales live on the `entry_type='sale'` axis, kept strictly separate from udhaar
# credit. Like every other rollup these figures are summed in Python from raw SQL
# rows (never an LLM, never SQL SUM()); 'unknown'-TYPE entries are excluded, so an
# unclassified line is honestly counted in neither the sales nor the udhaar total.

class SalesSummary(BaseModel):
    """Deterministic summary of sale-type entries (billing), separate from udhaar."""

    total_sales: float = Field(0.0, description="Sum of all sale-type entry amounts.")
    completed_total: float = Field(
        0.0, description="Completed sales (sale + status 'paid') — cash sales / cleared bills."
    )
    pending_total: float = Field(
        0.0, description="Pending invoices (sale + status not 'paid'; unknown status counts here)."
    )
    sale_count: int = Field(0, description="Number of sale-type entries.")
    completed_count: int = Field(0, description="Number of completed sales (status 'paid').")
    pending_count: int = Field(0, description="Number of pending invoices (status not 'paid').")


# ── Search + Insights (Phase 4) ──────────────────────────────────────────────
# core/search.py returns LedgerSearchHits; agents/insights_agent.py returns a
# LedgerStats (deterministic — no LLM ever touches these numbers) and an
# InsightAnswer (the chat reply, tagged with HOW it was produced so nothing is
# silently trusted — a hallucinated answer and a grounded one never look alike).

AnswerSource = Literal[
    "deterministic",        # answered straight from SQL aggregates — no LLM touched the number
    "llm",                  # Groq generated a grounded answer from the retrieved entries
    "extractive_fallback",  # LLM unavailable — sentences lifted verbatim from the entries
    "guardrail_refusal",    # question was out of scope; politely declined
    "empty",                # in scope, but the ledger has nothing relevant to say
]


class LedgerSearchHit(BaseModel):
    """One entry returned by hybrid search, with its fused relevance score."""

    record: LedgerEntryRecord = Field(..., description="The stored ledger entry that matched.")
    score: float = Field(..., description="Fused hybrid score: alpha*dense + (1-alpha)*sparse.")
    dense_score: float = Field(0.0, description="Semantic (SBERT/FAISS) component, normalized 0-1.")
    sparse_score: float = Field(0.0, description="Keyword (BM25) component, normalized 0-1.")

    def citation(self) -> str:
        """Judge-readable reference to this exact entry, for grounded answers."""
        r = self.record
        date = f", {r.raw_date}" if r.raw_date else ""
        return f"Entry #{r.id}: {r.customer_name} — ₹{r.amount:,.0f} {r.status}{date}"


class LedgerStats(BaseModel):
    """Deterministic ledger summary. Every number here comes straight from SQL —
    the LLM never computes these, it only phrases them (the anti-hallucination core)."""

    total_outstanding: float = Field(0.0, description="Sum of all unpaid/unknown balances.")
    customer_count: int = Field(0, description="Customers with a non-zero outstanding balance.")
    total_entries: int = Field(0, description="Total ledger entries stored.")
    flagged_count: int = Field(0, description="Entries flagged needs_review by verification.")
    top_defaulters: List[CustomerBalance] = Field(
        default_factory=list, description="Highest outstanding balances, descending."
    )
    monthly_totals: List[MonthlyTotal] = Field(
        default_factory=list, description="Per-month entry totals (best-effort date parsing)."
    )

    def headline_lines(self) -> List[str]:
        """Authoritative facts block injected into the LLM prompt as ground truth."""
        lines = [
            f"Total outstanding across all customers: ₹{self.total_outstanding:,.2f}",
            f"Customers with an outstanding balance: {self.customer_count}",
            f"Total entries on record: {self.total_entries} ({self.flagged_count} flagged for review)",
        ]
        for i, d in enumerate(self.top_defaulters[:5], 1):
            lines.append(f"Top defaulter #{i}: {d.name} owes ₹{d.unpaid_total:,.2f}")
        return lines


class InsightAnswer(BaseModel):
    """The Insights Agent's reply to one question, tagged with how it was made."""

    question: str = Field(..., description="The question asked.")
    answer: str = Field(..., description="The answer text, in the user's language.")
    source: AnswerSource = Field(..., description="How this answer was produced.")
    is_blocked: bool = Field(False, description="True if the guardrail refused the question.")
    degraded: bool = Field(False, description="True if produced without the LLM (fallback/mock).")
    citations: List[str] = Field(
        default_factory=list, description="Human-readable references to the entries used."
    )
    hit_ids: List[int] = Field(
        default_factory=list, description="Entry row ids retrieved (for UI highlighting)."
    )


# ── Reminder (Phase 5) ───────────────────────────────────────────────────────
# The Reminder Agent drafts copy-paste-ready payment reminders. The amount
# always comes from get_all_balances() (deterministic SQL) — the LLM only
# phrases the message, exactly like the Insights Agent's number discipline.

ReminderSourceLiteral = Literal["llm", "template"]


class ReminderDraft(BaseModel):
    """A drafted reminder message for one customer with an outstanding balance."""

    customer_name: str = Field(..., description="Customer display name.")
    amount: float = Field(..., description="Exact outstanding balance (from get_all_balances()).")
    message: str = Field(..., description="The drafted bilingual (Hindi/English) reminder text.")
    source: ReminderSourceLiteral = Field(
        ..., description="'llm' if Groq drafted it, 'template' if it used the fixed fallback."
    )
    degraded: bool = Field(
        False, description="True when the template fallback was used (no LLM available)."
    )
    since_date: Optional[str] = Field(
        None, description="Raw date string of the earliest unpaid entry found, if any."
    )


# ── Orchestrator (Phase 5) ───────────────────────────────────────────────────
# core/orchestrator.py runs Vision -> Verification -> Ledger for one photographed
# page and returns this summary. Every stage is best-effort: a failure is
# recorded in stage_errors, never raised, so the pipeline always returns.

class PageResult(BaseModel):
    """End-to-end result of running one page through the sequential pipeline."""

    source_image: str = Field(..., description="Path/filename of the processed photo.")
    extraction: PageExtraction = Field(..., description="Vision Agent output (possibly degraded).")
    verification: VerificationResult = Field(..., description="Verification Agent's audit.")
    ingest: Optional[IngestSummary] = Field(
        None, description="Ledger ingest summary, or null if ingest could not run."
    )
    degraded: bool = Field(
        False, description="True if any stage ran in a degraded/fallback mode."
    )
    stage_errors: List[str] = Field(
        default_factory=list, description="One message per stage that failed or degraded."
    )
