"""
Phase 8 tests — billing/sales as a SEPARATE axis from udhaar credit.

Covers the two-axis model end to end: entry_type defaulting/validation, type
separation (udhaar and sale never mix), udhaar-scoped balances (sales excluded
from outstanding), the sales summary (total/completed/pending), sales_this_month,
the unclassified rollup (unknown-TYPE excluded from BOTH totals), the
deterministic sales router (incl. the "sell in total" collision both directions),
and the info-severity unknown_type verification flag (verdict stays accept).

Every case uses an in-memory DB and needs no network / API key: the guardrail and
the deterministic router both resolve locally, so ask() never touches the LLM.

Run from the project root:
    .venv\\Scripts\\python.exe -m pytest tests/test_billing.py -v
"""
from __future__ import annotations

from datetime import datetime

from agents.insights_agent import ask, compute_stats
from agents.verification_agent import audit
from core.db import (
    connect,
    get_all_balances,
    get_all_entries,
    get_customer_balance,
    get_sales_summary,
    get_sales_this_month,
    get_unpaid_invoices,
    ingest_page,
)
from core.schemas import LedgerEntry, PageExtraction


# ── Helpers ──────────────────────────────────────────────────────────────────
def _entry(name, amount, status="unpaid", entry_type="udhaar", date="05/01/2026", confidence=0.95):
    return LedgerEntry(
        name=name, amount=amount, date=date, status=status,
        entry_type=entry_type, confidence=confidence,
        raw_text=f"{name} {amount} {entry_type}",
    )


def _db(*entries, source_image="p.jpg"):
    conn = connect(":memory:")
    page = PageExtraction(source_image=source_image, entries=list(entries), overall_confidence=0.95)
    ingest_page(conn, page, audit(page))
    return conn


def _codes(result):
    return {i.code for i in result.issues}


# ── Schema: default + validator + persistence ────────────────────────────────
def test_entry_type_defaults_to_udhaar():
    # Every row predates billing unless told otherwise — the backward-compat default.
    assert LedgerEntry(name="X", amount=100, confidence=0.9).entry_type == "udhaar"


def test_entry_type_validator_coerces():
    assert LedgerEntry(name="X", amount=1, confidence=0.9, entry_type="SALE").entry_type == "sale"
    assert LedgerEntry(name="X", amount=1, confidence=0.9, entry_type="garbage").entry_type == "udhaar"


def test_entry_type_persists_through_db():
    conn = _db(
        _entry("A", 100, "unpaid", "udhaar"),
        _entry("B", 200, "paid", "sale"),
        _entry("C", 300, "unknown", "unknown"),
    )
    types = {r.customer_name: r.entry_type for r in get_all_entries(conn)}
    assert types == {"A": "udhaar", "B": "sale", "C": "unknown"}


# ── Type separation + udhaar-scoped balances ─────────────────────────────────
def test_udhaar_and_sale_do_not_mix():
    # Same customer, one credit line and one sale line — they must not blur.
    conn = _db(
        _entry("Ramesh", 1000, "unpaid", "udhaar"),
        _entry("Ramesh", 700, "unpaid", "sale"),
    )
    bal = get_customer_balance(conn, "Ramesh")
    assert bal.unpaid_total == 1000       # only the udhaar line counts as owed credit
    assert bal.entry_count == 1           # balance is udhaar-scoped
    summary = get_sales_summary(conn)
    assert summary.total_sales == 700     # only the sale line
    assert summary.pending_total == 700


def test_outstanding_excludes_sales_and_unclassified():
    conn = _db(
        _entry("A", 500, "unpaid", "udhaar"),
        _entry("B", 900, "unpaid", "sale"),      # pending invoice, NOT outstanding credit
        _entry("C", 300, "unknown", "unknown"),  # unclassified, NOT outstanding
    )
    balances = {b.name: b for b in get_all_balances(conn)}
    assert set(balances) == {"A"}               # only the udhaar customer surfaces
    assert balances["A"].unpaid_total == 500
    stats = compute_stats(conn)
    assert stats.total_outstanding == 500       # sales & unclassified excluded
    assert stats.customer_count == 1


# ── Sales summary + unpaid invoices ──────────────────────────────────────────
def test_sales_summary_totals():
    conn = _db(
        _entry("Karan", 2500, "paid", "sale"),
        _entry("Deepak", 1800, "unpaid", "sale"),
        _entry("Gopal", 300, "unknown", "sale"),    # unknown STATUS -> counts as pending
        _entry("Ramesh", 1000, "unpaid", "udhaar"),  # udhaar -> ignored by sales summary
    )
    s = get_sales_summary(conn)
    assert s.total_sales == 4600                 # 2500 + 1800 + 300 (udhaar excluded)
    assert s.completed_total == 2500             # only the paid sale
    assert s.pending_total == 2100               # 1800 unpaid + 300 unknown-status
    assert (s.sale_count, s.completed_count, s.pending_count) == (3, 1, 2)


def test_unpaid_invoices_are_pending_sales_only():
    conn = _db(
        _entry("Karan", 2500, "paid", "sale"),      # completed -> excluded
        _entry("Deepak", 1800, "unpaid", "sale"),   # pending -> included
        _entry("Gopal", 300, "unknown", "sale"),    # unknown status -> included
        _entry("Ramesh", 1000, "unpaid", "udhaar"),  # udhaar -> excluded
    )
    inv = get_unpaid_invoices(conn)
    assert {r.customer_name for r in inv} == {"Deepak", "Gopal"}
    assert all(r.entry_type == "sale" and r.status != "paid" for r in inv)


# ── sales_this_month (best-effort date parsing) ──────────────────────────────
def test_sales_this_month_excludes_other_months_and_udhaar():
    this_month = datetime.now().strftime("%d/%m/%Y")
    conn = _db(
        _entry("Karan", 2500, "paid", "sale", date=this_month),
        _entry("Deepak", 1800, "unpaid", "sale", date=this_month),
        _entry("OldSale", 999, "paid", "sale", date="05/01/2020"),      # different month
        _entry("Ramesh", 1000, "unpaid", "udhaar", date=this_month),     # udhaar excluded
    )
    assert get_sales_this_month(conn) == 4300           # 2500 + 1800
    assert compute_stats(conn).sales_this_month == 4300  # surfaced identically in stats


# ── Unclassified rollup (honesty rule) ───────────────────────────────────────
def test_unclassified_rollup_excluded_from_both():
    conn = _db(
        _entry("A", 500, "unpaid", "udhaar"),
        _entry("B", 900, "unpaid", "sale"),
        _entry("Mystery", 400, "unpaid", "unknown"),  # unknown TYPE -> in neither total
    )
    stats = compute_stats(conn)
    assert stats.unclassified_total == 400
    assert stats.unclassified_count == 1
    assert stats.total_outstanding == 500                 # not in udhaar
    assert get_sales_summary(conn).total_sales == 900     # not in sales


# ── Deterministic sales router ───────────────────────────────────────────────
def test_sales_router_is_deterministic():
    conn = _db(
        _entry("Karan", 2500, "paid", "sale"),
        _entry("Deepak", 1800, "unpaid", "sale"),
    )
    for q in ["kitni bikri hui?", "how much did I sell?", "aaj ki bikri", "total sales?"]:
        ans = ask(q, conn)
        assert ans.source == "deterministic", q
        assert ans.is_blocked is False
        assert "4,300" in ans.answer      # total sales 2500 + 1800
        assert "1,800" in ans.answer      # still pending as invoices


def test_router_collision_both_directions():
    conn = _db(
        _entry("Karan", 2500, "paid", "sale"),
        _entry("Ramesh", 1000, "unpaid", "udhaar"),
    )
    # "sell ... in total" must route to SALES, not the outstanding total.
    sell = ask("how much did I sell in total?", conn)
    assert sell.source == "deterministic"
    assert "2,500" in sell.answer
    assert "owe" not in sell.answer.lower()

    # "total outstanding" must still route to the UDHAAR total, not sales.
    out = ask("what is the total outstanding?", conn)
    assert out.source == "deterministic"
    assert "1,000" in out.answer
    assert "sold" not in out.answer.lower() and "sell" not in out.answer.lower()


# ── Verification: unknown_type is info, verdict stays accept ──────────────────
def test_unknown_type_flag_is_info_and_verdict_accepts():
    # An otherwise-clean line whose only problem is an undetermined type.
    page = PageExtraction(
        source_image="p.jpg",
        entries=[_entry("Clear", 500, "unpaid", "unknown", confidence=0.95)],
        overall_confidence=0.95,
    )
    result = audit(page)
    assert "unknown_type" in _codes(result)
    issue = next(i for i in result.issues if i.code == "unknown_type")
    assert issue.severity == "info"
    assert result.verdict == "accept"                 # info-only never forces review
    assert result.error_count == 0 and result.warning_count == 0
