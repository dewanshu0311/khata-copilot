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
