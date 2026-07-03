"""
Ledger Agent storage — SQLite via the stdlib sqlite3.

Pipeline role (Phase 3): (PageExtraction, VerificationResult) in -> persisted
ledger rows out. Pure storage logic, no LLM calls.

Contract carried over from Verification: this layer stores what was found,
including problems. A flagged entry is written with needs_review=True, never
silently dropped and never silently trusted.

Dedup: same (customer, amount, raw date string, source image) is one row.
Re-scanning a page updates that row (status/confidence/raw_text/needs_review)
instead of duplicating it, via SQLite's ON CONFLICT ... DO UPDATE.

Customer matching: exact match on name after whitespace/case normalization
only. "Ramesh" and "Ramesh Kumar" are different customers — fuzzy merging is
out of scope for this phase, kept simple and judge-explainable.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from typing import List, Optional, Sequence

from core.config import DB_PATH
from core.schemas import (
    CustomerBalance,
    IngestSummary,
    LedgerEntry,
    LedgerEntryRecord,
    MonthlyTotal,
    PageExtraction,
    VerificationResult,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    normalized_name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_image TEXT NOT NULL UNIQUE,
    verdict TEXT NOT NULL,
    overall_confidence REAL NOT NULL,
    computed_total REAL NOT NULL,
    written_total REAL,
    scanned_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    page_id INTEGER NOT NULL REFERENCES pages(id),
    source_image TEXT NOT NULL,
    amount REAL NOT NULL,
    raw_date TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    raw_text TEXT NOT NULL DEFAULT '',
    needs_review INTEGER NOT NULL,
    scanned_at TEXT NOT NULL,
    UNIQUE(customer_id, amount, raw_date, source_image)
);
"""

# A handful of common raw-date formats a shopkeeper might write. Anything else
# falls into an "unknown" bucket rather than raising — dates are a raw string
# by design (see core/schemas.py), never parsed at extraction time.
_DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y")


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open (and initialize) the ledger database. Pass ':memory:' for tests."""
    conn = sqlite3.connect(db_path if db_path is not None else DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name or "").strip().casefold()


def _get_or_create_customer(conn: sqlite3.Connection, name: str) -> int:
    normalized = _normalize_name(name)
    row = conn.execute(
        "SELECT id FROM customers WHERE normalized_name = ?", (normalized,)
    ).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        "INSERT INTO customers (normalized_name, display_name) VALUES (?, ?)",
        (normalized, name.strip()),
    )
    return cur.lastrowid


def ingest_page(
    conn: sqlite3.Connection, page: PageExtraction, result: VerificationResult
) -> IngestSummary:
    """Persist a verified page. Flagged entries are stored, not dropped or trusted."""
    now = datetime.utcnow().isoformat(timespec="seconds")

    conn.execute(
        """
        INSERT INTO pages (source_image, verdict, overall_confidence, computed_total, written_total, scanned_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_image) DO UPDATE SET
            verdict=excluded.verdict,
            overall_confidence=excluded.overall_confidence,
            computed_total=excluded.computed_total,
            written_total=excluded.written_total,
            scanned_at=excluded.scanned_at
        """,
        (page.source_image, result.verdict, result.overall_confidence,
         result.computed_total, result.written_total, now),
    )
    page_id = conn.execute(
        "SELECT id FROM pages WHERE source_image = ?", (page.source_image,)
    ).fetchone()[0]

    # A page-level issue (entry_index == -1, e.g. math_mismatch) can't be
    # pinned to one line, so it flags every entry on the page for review.
    flagged_indices = {
        i.entry_index for i in result.issues
        if i.severity in ("warning", "error") and i.entry_index >= 0
    }
    page_level_flag = any(
        i.entry_index == -1 and i.severity in ("warning", "error") for i in result.issues
    )

    inserted = updated = 0
    for idx, entry in enumerate(page.entries):
        customer_id = _get_or_create_customer(conn, entry.name)
        needs_review = page_level_flag or idx in flagged_indices
        raw_date = entry.date or ""

        existing = conn.execute(
            """SELECT id FROM entries
               WHERE customer_id = ? AND amount = ? AND raw_date = ? AND source_image = ?""",
            (customer_id, entry.amount, raw_date, page.source_image),
        ).fetchone()

        if existing:
            conn.execute(
                """UPDATE entries SET page_id=?, status=?, confidence=?, raw_text=?,
                       needs_review=?, scanned_at=?
                   WHERE id=?""",
                (page_id, entry.status, entry.confidence, entry.raw_text,
                 int(needs_review), now, existing[0]),
            )
            updated += 1
        else:
            conn.execute(
                """INSERT INTO entries
                       (customer_id, page_id, source_image, amount, raw_date, status,
                        confidence, raw_text, needs_review, scanned_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (customer_id, page_id, page.source_image, entry.amount, raw_date,
                 entry.status, entry.confidence, entry.raw_text, int(needs_review), now),
            )
            inserted += 1

    conn.commit()
    return IngestSummary(
        source_image=page.source_image, inserted=inserted, updated=updated,
        page_verdict=result.verdict,
    )


def _row_to_record(row: Sequence) -> LedgerEntryRecord:
    return LedgerEntryRecord(
        id=row[0], customer_name=row[1], amount=row[2], raw_date=row[3] or None,
        status=row[4], confidence=row[5], raw_text=row[6], source_image=row[7],
        needs_review=bool(row[8]), scanned_at=row[9],
    )


def get_customer_balance(conn: sqlite3.Connection, name: str) -> CustomerBalance:
    """Outstanding total for one customer. 'unknown' status counts as outstanding."""
    row = conn.execute(
        "SELECT id, display_name FROM customers WHERE normalized_name = ?",
        (_normalize_name(name),),
    ).fetchone()
    if not row:
        return CustomerBalance(name=name.strip(), unpaid_total=0.0, paid_total=0.0, entry_count=0)

    customer_id, display_name = row
    entries = conn.execute(
        "SELECT amount, status FROM entries WHERE customer_id = ?", (customer_id,)
    ).fetchall()
    unpaid_total = round(sum(a for a, s in entries if s != "paid"), 2)
    paid_total = round(sum(a for a, s in entries if s == "paid"), 2)
    return CustomerBalance(
        name=display_name, unpaid_total=unpaid_total, paid_total=paid_total,
        entry_count=len(entries),
    )


def get_all_balances(conn: sqlite3.Connection) -> List[CustomerBalance]:
    """Every customer with an outstanding balance, sorted highest first."""
    balances = []
    for customer_id, display_name in conn.execute("SELECT id, display_name FROM customers"):
        entries = conn.execute(
            "SELECT amount, status FROM entries WHERE customer_id = ?", (customer_id,)
        ).fetchall()
        unpaid_total = round(sum(a for a, s in entries if s != "paid"), 2)
        if unpaid_total <= 0:
            continue
        paid_total = round(sum(a for a, s in entries if s == "paid"), 2)
        balances.append(CustomerBalance(
            name=display_name, unpaid_total=unpaid_total, paid_total=paid_total,
            entry_count=len(entries),
        ))
    balances.sort(key=lambda b: b.unpaid_total, reverse=True)
    return balances


def get_entries_needing_review(conn: sqlite3.Connection) -> List[LedgerEntryRecord]:
    rows = conn.execute(
        """SELECT e.id, c.display_name, e.amount, e.raw_date, e.status, e.confidence,
                  e.raw_text, e.source_image, e.needs_review, e.scanned_at
           FROM entries e JOIN customers c ON c.id = e.customer_id
           WHERE e.needs_review = 1"""
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def get_all_entries(conn: sqlite3.Connection) -> List[LedgerEntryRecord]:
    """Every stored ledger entry (newest first) — the corpus the Insights Agent searches."""
    rows = conn.execute(
        """SELECT e.id, c.display_name, e.amount, e.raw_date, e.status, e.confidence,
                  e.raw_text, e.source_image, e.needs_review, e.scanned_at
           FROM entries e JOIN customers c ON c.id = e.customer_id
           ORDER BY e.id DESC"""
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def _parse_month(raw_date: str) -> str:
    if not raw_date:
        return "unknown"
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw_date.strip(), fmt).strftime("%Y-%m")
        except ValueError:
            continue
    return "unknown"


# ── Demo data (Phase 7) ──────────────────────────────────────────────────────
# A small, fixed set of clean entries so a judge demo of Insights/Reminders
# doesn't depend on a live Gemini/Groq call or accumulated scan clutter. Goes
# through the same ingest_page() path as a real scan (verdict "accept", no
# issues) rather than raw SQL, so it stays consistent with real data shape.
_DEMO_PAGES: list[tuple[str, list[dict]]] = [
    ("demo_page_1.jpg", [
        {"name": "Ramesh Kumar", "amount": 1200.0, "date": "5 Jan", "status": "unpaid", "confidence": 0.95, "raw_text": "Ramesh Kumar 1200 udhaar 5 Jan"},
        {"name": "Sita Devi", "amount": 450.0, "date": "6 Jan", "status": "paid", "confidence": 0.92, "raw_text": "Sita Devi 450 jama 6 Jan"},
        {"name": "Mohan Lal", "amount": 3200.0, "date": "8 Jan", "status": "unpaid", "confidence": 0.9, "raw_text": "Mohan Lal 3200 baki 8 Jan"},
    ]),
    ("demo_page_2.jpg", [
        {"name": "Anita Sharma", "amount": 800.0, "date": "10 Jan", "status": "unpaid", "confidence": 0.88, "raw_text": "Anita Sharma 800 udhaar 10 Jan"},
        {"name": "Ramesh Kumar", "amount": 600.0, "date": "15 Jan", "status": "paid", "confidence": 0.94, "raw_text": "Ramesh Kumar 600 jama 15 Jan"},
        {"name": "Mohan Lal", "amount": 1500.0, "date": "18 Jan", "status": "unknown", "confidence": 0.55, "raw_text": "Mohan Lal 1500 ? 18 Jan"},
    ]),
]


def clear_all(conn: sqlite3.Connection) -> None:
    """Wipe every ledger table. Used by 'Load Demo Data' to start from a clean slate."""
    conn.executescript("DELETE FROM entries; DELETE FROM pages; DELETE FROM customers;")
    conn.commit()


def seed_demo_data(conn: sqlite3.Connection) -> IngestSummary:
    """Clear the ledger and load a fixed, known-good set of demo entries.

    Lets a judge demo run Insights/Reminders instantly without a real scan.
    """
    clear_all(conn)
    total_inserted = total_updated = 0
    for source_image, raw_entries in _DEMO_PAGES:
        entries = [LedgerEntry(**e) for e in raw_entries]
        page = PageExtraction(
            source_image=source_image, entries=entries,
            overall_confidence=sum(e.confidence for e in entries) / len(entries),
            notes="Demo data (Load Demo Data button) — not a real scan.",
        )
        verification = VerificationResult(
            source_image=source_image, verdict="accept",
            overall_confidence=page.overall_confidence,
            computed_total=sum(e.amount for e in entries),
        )
        summary = ingest_page(conn, page, verification)
        total_inserted += summary.inserted
        total_updated += summary.updated
    return IngestSummary(
        source_image="demo_data", inserted=total_inserted, updated=total_updated,
        page_verdict="accept",
    )


def get_monthly_totals(conn: sqlite3.Connection) -> List[MonthlyTotal]:
    """Sum of entry amounts per calendar month, best-effort parsed from raw dates."""
    buckets: dict[str, dict[str, float]] = {}
    for raw_date, amount in conn.execute("SELECT raw_date, amount FROM entries"):
        month = _parse_month(raw_date)
        bucket = buckets.setdefault(month, {"total": 0.0, "count": 0})
        bucket["total"] += amount
        bucket["count"] += 1

    totals = [
        MonthlyTotal(month=month, total=round(v["total"], 2), entry_count=int(v["count"]))
        for month, v in buckets.items()
    ]
    totals.sort(key=lambda m: m.month)
    return totals
