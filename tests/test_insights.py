"""
Phase 4 tests — hybrid search + the Insights Agent.

Everything runs OFFLINE and makes NO real API calls:
  - deterministic stats and the mini-router never touch the LLM;
  - the guardrail test is refused by a local keyword pattern (no LLM);
  - the fallback test forces KHATA_MOCK=1 so Groq returns None;
  - the semantic-search test SKIPS gracefully if the SBERT model isn't available
    (e.g. no network to download it), so the suite is green offline.

Run from the project root:
    .venv\\Scripts\\python.exe -m pytest tests/test_insights.py -v
"""
from __future__ import annotations

import pytest

from agents.insights_agent import ask, compute_stats
from agents.verification_agent import audit
from core.db import connect, get_all_entries, ingest_page
from core.schemas import LedgerEntry, PageExtraction
from core.search import LedgerSearchIndex


# ── Helpers ──────────────────────────────────────────────────────────────────
def _entry(name, amount, date, status, raw_text):
    return LedgerEntry(
        name=name, amount=amount, date=date, status=status,
        confidence=0.95, raw_text=raw_text,
    )


def _seed():
    """In-memory ledger with hand-known balances:

        Ramesh Kumar : 1200 + 300 unpaid          -> owes 1500  (top defaulter)
        Sita Devi    : 450 paid                    -> owes 0     (excluded)
        Mohan        : 800 unpaid                  -> owes 800
        Gita         : 200 unknown                 -> owes 200   (unknown counts)

        total outstanding = 2500 across 3 customers.
    """
    conn = connect(":memory:")
    page = PageExtraction(
        source_image="ledger1.jpg",
        entries=[
            _entry("Ramesh Kumar", 1200, "05/01/2026", "unpaid", "Ramesh Kumar 1200 udhaar 5 Jan"),
            _entry("Ramesh Kumar", 300, "12/01/2026", "unpaid", "Ramesh Kumar 300 udhaar 12 Jan"),
            _entry("Sita Devi", 450, "06/01/2026", "paid", "Sita Devi 450 jama 6 Jan"),
            _entry("Mohan", 800, "06/01/2026", "unpaid", "Mohan 800 baaki 6 Jan"),
            _entry("Gita", 200, "07/01/2026", "unknown", "Gita 200 6 Jan"),
        ],
        overall_confidence=0.95,
    )
    ingest_page(conn, page, audit(page))
    return conn


# ── 1. Deterministic stats match hand-computed values ────────────────────────
def test_deterministic_stats_match_hand_computed():
    stats = compute_stats(_seed())
    assert stats.total_outstanding == 2500
    assert stats.customer_count == 3
    assert stats.total_entries == 5
    assert stats.top_defaulters[0].name == "Ramesh Kumar"
    assert stats.top_defaulters[0].unpaid_total == 1500


# ── 2. The mini-router answers money questions deterministically (no LLM) ─────
def test_router_total_outstanding_is_deterministic():
    ans = ask("What is the total outstanding amount?", _seed())
    assert ans.source == "deterministic"
    assert ans.is_blocked is False
    assert "2,500" in ans.answer


def test_router_top_defaulter_is_deterministic():
    ans = ask("Who owes me the most?", _seed())
    assert ans.source == "deterministic"
    assert "Ramesh Kumar" in ans.answer


def test_router_top_n_honors_requested_count():
    # "top 2" must list exactly the two biggest (Ramesh, Mohan) and NOT the third
    # (Gita) — the old code always returned a fixed 3 regardless of the number asked.
    ans = ask("give me the top 2 people who owe me the most", _seed())
    assert ans.source == "deterministic"
    assert "Ramesh Kumar" in ans.answer and "Mohan" in ans.answer
    assert "Gita" not in ans.answer
    assert len(ans.citations) == 2


def test_router_top_n_more_than_exist_says_so():
    # Only 3 customers have a balance; asking for 5 should honestly say "Only 3".
    ans = ask("top 5 defaulters", _seed())
    assert ans.source == "deterministic"
    assert "Only 3" in ans.answer
    assert all(n in ans.answer for n in ("Ramesh Kumar", "Mohan", "Gita"))


def test_router_least_is_deterministic():
    # The mirror question the old router couldn't answer at all — Gita owes the least.
    ans = ask("who owes me the least?", _seed())
    assert ans.source == "deterministic"
    assert "Gita" in ans.answer
    assert "200" in ans.answer


# ── 3. Hybrid search: exact name (BM25, always) + semantic query (SBERT) ──────
def test_search_finds_customer_by_exact_name():
    index = LedgerSearchIndex(get_all_entries(_seed()))
    hits = index.search("Mohan", k=3)
    assert hits, "expected at least one hit for an exact customer name"
    assert hits[0].record.customer_name == "Mohan"


def test_search_finds_owing_customers_by_semantic_query():
    index = LedgerSearchIndex(get_all_entries(_seed()))
    if not index.dense_enabled:
        pytest.skip("SBERT model unavailable (offline); dense semantic search not exercised")
    # Query shares NO literal tokens with the entries, so BM25 contributes nothing
    # and the dense half must carry it: an owing customer should rank above the
    # single fully-paid one purely by meaning.
    hits = index.search("which people still have an unsettled debt to repay", k=5)
    assert hits
    assert hits[0].record.status != "paid"


# ── 4. Guardrail refuses an obviously out-of-scope question (local, no LLM) ───
def test_guardrail_refuses_out_of_scope():
    ans = ask("What is the capital of France?", _seed())
    assert ans.is_blocked is True
    assert ans.source == "guardrail_refusal"
    assert "khata" in ans.answer.lower()


# ── 5. Extractive fallback when the LLM is unavailable (mock mode) ────────────
def test_extractive_fallback_when_llm_unavailable(monkeypatch):
    monkeypatch.setenv("KHATA_MOCK", "1")  # forces groq_chat() to return None
    conn = _seed()
    index = LedgerSearchIndex(get_all_entries(conn))
    ans = ask("Tell me about Ramesh's dues", conn, index=index)

    assert ans.source == "extractive_fallback"
    assert ans.degraded is True
    assert "Ramesh" in ans.answer
    assert "Entry #" in ans.answer          # answer cites the entries it used
    assert ans.hit_ids                       # retrieved at least one entry
