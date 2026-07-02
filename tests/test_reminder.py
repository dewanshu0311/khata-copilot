"""
Phase 5 tests — the Reminder Agent drafts payment reminders from the ledger.

Every case uses an in-memory database (':memory:') and KHATA_MOCK=1 (no real
Groq calls), so the suite runs offline in milliseconds.

Run from the project root:
    .venv\\Scripts\\python.exe -m pytest tests/test_reminder.py -v
"""
from __future__ import annotations

import os

from agents.reminder_agent import draft_reminder_for, draft_reminders
from agents.verification_agent import audit
from core.db import connect, ingest_page
from core.schemas import LedgerEntry, PageExtraction

os.environ["KHATA_MOCK"] = "1"  # force template fallback for the whole file


def _entry(name="Ramesh", amount=500.0, date="05/01/2026", status="unpaid", confidence=0.95):
    return LedgerEntry(
        name=name, amount=amount, date=date, status=status,
        confidence=confidence, raw_text=f"{name} {amount}",
    )


def _db():
    return connect(":memory:")


# ── 1. A customer with a balance gets a draft ────────────────────────────────
def test_customer_with_balance_gets_draft():
    conn = _db()
    page = PageExtraction(
        source_image="p1.jpg", entries=[_entry("Ramesh", 750, status="unpaid")],
        overall_confidence=0.95,
    )
    ingest_page(conn, page, audit(page))

    drafts = draft_reminders(conn)
    assert len(drafts) == 1
    assert drafts[0].customer_name == "Ramesh"
    assert drafts[0].amount == 750


# ── 2. Template fallback works under KHATA_MOCK=1 ────────────────────────────
def test_template_fallback_under_mock():
    conn = _db()
    page = PageExtraction(
        source_image="p1.jpg", entries=[_entry("Sita", 300, status="unpaid")],
        overall_confidence=0.95,
    )
    ingest_page(conn, page, audit(page))

    drafts = draft_reminders(conn)
    assert drafts[0].source == "template"
    assert drafts[0].degraded is True
    assert drafts[0].message  # non-empty


# ── 3. A fully-paid customer gets no reminder ────────────────────────────────
def test_fully_paid_customer_gets_no_reminder():
    conn = _db()
    page = PageExtraction(
        source_image="p1.jpg",
        entries=[
            _entry("Mohan", 500, status="unpaid"),
            _entry("Gita", 200, status="paid"),
        ],
        overall_confidence=0.95,
    )
    ingest_page(conn, page, audit(page))

    drafts = draft_reminders(conn)
    names = {d.customer_name for d in drafts}
    assert "Mohan" in names
    assert "Gita" not in names


# ── 4. The draft contains the exact amount ───────────────────────────────────
def test_draft_contains_exact_amount():
    conn = _db()
    page = PageExtraction(
        source_image="p1.jpg", entries=[_entry("Kavita", 1234.5, status="unpaid")],
        overall_confidence=0.95,
    )
    ingest_page(conn, page, audit(page))

    draft = draft_reminders(conn)[0]
    assert "1,234.50" in draft.message


# ── 5. Single-customer helper honors an injected complete_fn (llm source) ───
def test_draft_reminder_for_uses_injected_llm():
    conn = _db()
    fake_llm = lambda system, user, **kw: "Namaste Ramesh, aap par 500 baki hai. / Hi Ramesh, you owe 500."
    draft = draft_reminder_for("Ramesh", 500.0, conn, complete_fn=fake_llm)
    assert draft.source == "llm"
    assert draft.degraded is False
    assert draft.amount == 500.0
