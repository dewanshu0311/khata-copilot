"""
Phase 3 tests — the Ledger Agent persists verified extractions into SQLite.

Every case uses an in-memory database (':memory:'), so the suite never
touches khata.db and runs offline in milliseconds.

Run from the project root:
    .venv\\Scripts\\python.exe -m pytest tests/test_ledger.py -v
"""
from __future__ import annotations

from agents.verification_agent import audit
from core.db import (
    connect,
    get_all_balances,
    get_customer_balance,
    get_entries_needing_review,
    get_monthly_totals,
    ingest_page,
)
from core.schemas import LedgerEntry, PageExtraction


# ── Helpers ──────────────────────────────────────────────────────────────────
def _entry(name="Ramesh", amount=100.0, date="05/01/2026", status="unpaid", confidence=0.95):
    return LedgerEntry(
        name=name, amount=amount, date=date, status=status,
        confidence=confidence, raw_text=f"{name} {amount}",
    )


def _db():
    return connect(":memory:")


# ── 1. Ingest a verified page ────────────────────────────────────────────────
def test_ingest_stores_entries():
    conn = _db()
    page = PageExtraction(
        source_image="p1.jpg",
        entries=[_entry("Ramesh", 500, status="unpaid"), _entry("Sita", 300, status="paid")],
        overall_confidence=0.95,
    )
    result = audit(page)
    summary = ingest_page(conn, page, result)

    assert summary.inserted == 2
    assert summary.updated == 0
    balances = {b.name: b for b in get_all_balances(conn)}
    assert balances["Ramesh"].unpaid_total == 500
    assert "Sita" not in balances  # fully paid, no outstanding balance


# ── 2. Re-ingesting the same page updates instead of duplicating ────────────
def test_reingest_same_page_dedupes():
    conn = _db()
    page = PageExtraction(
        source_image="p1.jpg",
        entries=[_entry("Ramesh", 500, status="unpaid", confidence=0.4)],
        overall_confidence=0.4,
    )
    result = audit(page)
    ingest_page(conn, page, result)

    # Re-scan: same (name, amount, date, source_image), higher confidence now.
    page2 = PageExtraction(
        source_image="p1.jpg",
        entries=[_entry("Ramesh", 500, status="unpaid", confidence=0.97)],
        overall_confidence=0.97,
    )
    result2 = audit(page2)
    summary2 = ingest_page(conn, page2, result2)

    assert summary2.inserted == 0
    assert summary2.updated == 1
    count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    assert count == 1
    record = get_entries_needing_review(conn)
    assert record == []  # confidence 0.97 is above threshold now, no longer flagged


# ── 3. Balances: unpaid + unknown count as outstanding, paid excluded ───────
def test_balances_exclude_paid_only():
    conn = _db()
    page = PageExtraction(
        source_image="p1.jpg",
        entries=[
            _entry("Gita", 200, status="unpaid"),
            _entry("Gita", 150, status="unknown"),
            _entry("Gita", 100, status="paid"),
        ],
        overall_confidence=0.95,
    )
    result = audit(page)
    ingest_page(conn, page, result)

    balance = get_customer_balance(conn, "gita")  # case-insensitive lookup
    assert balance.unpaid_total == 350  # unpaid + unknown
    assert balance.paid_total == 100
    assert balance.entry_count == 3


# ── 4. Flagged entries are stored with needs_review, queryable ──────────────
def test_flagged_entries_queryable():
    conn = _db()
    page = PageExtraction(
        source_image="p1.jpg",
        entries=[
            _entry("Ramesh", 500, confidence=0.95),   # clean
            _entry("Mohan", -50, confidence=0.95),    # nonpositive_amount -> flagged
        ],
        overall_confidence=0.9,
    )
    result = audit(page)
    ingest_page(conn, page, result)

    flagged = get_entries_needing_review(conn)
    names = {r.customer_name for r in flagged}
    assert names == {"Mohan"}
    assert all(r.needs_review for r in flagged)


# ── 5. A page-level issue (math mismatch) flags every entry on the page ─────
def test_page_level_issue_flags_all_entries():
    conn = _db()
    page = PageExtraction(
        source_image="p1.jpg",
        entries=[_entry("A", 1000, confidence=0.95), _entry("B", 500, confidence=0.95)],
        written_total=2000,  # sum is 1500 -> math_mismatch, page-level
        overall_confidence=0.95,
    )
    result = audit(page)
    ingest_page(conn, page, result)

    flagged = get_entries_needing_review(conn)
    assert len(flagged) == 2


# ── 6. Monthly totals: parseable dates bucket, unparseable falls to unknown ─
def test_monthly_totals():
    conn = _db()
    page = PageExtraction(
        source_image="p1.jpg",
        entries=[
            _entry("A", 100, date="05/01/2026"),
            _entry("B", 200, date="20/01/2026"),
            _entry("C", 300, date="not-a-date"),
        ],
        overall_confidence=0.95,
    )
    result = audit(page)
    ingest_page(conn, page, result)

    totals = {m.month: m for m in get_monthly_totals(conn)}
    assert totals["2026-01"].total == 300
    assert totals["2026-01"].entry_count == 2
    assert totals["unknown"].total == 300
    assert totals["unknown"].entry_count == 1
