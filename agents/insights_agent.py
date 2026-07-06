"""
Insights Agent — answers a shopkeeper's plain-language questions about their ledger.

Pipeline role (Phase 4): a question (+ the SQLite ledger) in -> a grounded,
source-tagged InsightAnswer out.

The design is deliberately split so money numbers can never be hallucinated:

  1. DETERMINISTIC stats (total outstanding, top defaulters, monthly totals) are
     computed straight from SQL by compute_stats(). No LLM ever touches them.
  2. A tiny ROUTER answers the unambiguous money questions directly from SQL —
     total outstanding, sales, and ranking ("top N who owe most", "who owes the
     least", honoring the requested count) — source="deterministic".
  3. Everything else goes through RAG: a GUARDRAIL (out-of-scope questions are
     politely refused), hybrid RETRIEVAL of the relevant entries, then Groq
     generates an answer GROUNDED in those entries plus the exact pre-computed
     totals (injected as authoritative context so it phrases numbers, never invents).
  4. If Groq is unavailable (mock mode, no key, error) we fall back to an
     EXTRACTIVE answer built from the retrieved entries — the demo never dies.

Every InsightAnswer carries a `source` tag, so a grounded answer, a deterministic
answer, a fallback, and a refusal never look alike.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

from core import prompts
from core.config import SEARCH_TOP_K
from core.db import (
    get_all_balances,
    get_all_entries,
    get_entries_needing_review,
    get_monthly_totals,
    get_sales_summary,
    get_sales_this_month,
)
from core.groq_client import groq_chat
from core.schemas import InsightAnswer, LedgerStats
from core.search import LedgerSearchIndex, extractive_answer

# A completion callable: (system, user, **kwargs) -> answer text or None. Defaults
# to core.groq_client.groq_chat; injectable so tests run without real API calls.
CompleteFn = Callable[..., Optional[str]]


# ── Deterministic stats (pure SQL, never the LLM) ────────────────────────────
def compute_stats(conn) -> LedgerStats:
    """Summarize the ledger straight from the database. Every number is exact —
    the LLM never computes these. Udhaar credit and sales are separate axes:
    total_outstanding is udhaar-only; the sales fields cover the billing axis;
    unknown-TYPE entries are counted in neither (the honesty rule)."""
    balances = get_all_balances(conn)  # udhaar-scoped, sorted desc, zero balances excluded
    all_entries = get_all_entries(conn)
    unclassified = [e for e in all_entries if e.entry_type == "unknown"]
    sales = get_sales_summary(conn)
    return LedgerStats(
        total_outstanding=round(sum(b.unpaid_total for b in balances), 2),
        customer_count=len(balances),
        total_entries=len(all_entries),
        flagged_count=len(get_entries_needing_review(conn)),
        top_defaulters=balances[:5],
        monthly_totals=get_monthly_totals(conn),
        total_sales=sales.total_sales,
        sales_this_month=get_sales_this_month(conn),
        unpaid_invoices_total=sales.pending_total,
        unclassified_total=round(sum(e.amount for e in unclassified), 2),
        unclassified_count=len(unclassified),
    )


# ── Guardrail (rescoped from my Rag-Assistant-masterclass build) ─────────────
# Clearly-out-of-scope topics. Kept narrow and non-overlapping with ledger words
# so a real khata question is never caught here by accident.
_OOS_PATTERNS = (
    r"\b(weather|temperature|forecast|barish|mausam)\b",
    r"\b(capital of|president|prime minister|population of|who invented|when did)\b",
    r"\b(joke|shayari|poem|kavita|recipe|movie|film|song|gaana|kahani)\b",
    r"\b(cricket|football|ipl|world ?cup|kohli|messi)\b",
    r"\b(write|create|generate|debug|fix)\b.*\b(code|program|script|python|java|sql|html|essay)\b",
    r"\b\d+\s*[\+\-\*x/]\s*\d+\b",  # arithmetic puzzle, e.g. "2+2"
    r"\b(stock price|bitcoin|crypto|sensex|nifty|share market)\b",
)
# Ledger vocabulary (Latin + Devanagari) that marks a question as in-scope.
_IN_SCOPE_HINTS = (
    "udhaar", "udhar", "उधार", "baki", "baaki", "बाकी", "jama", "जमा",
    "owe", "owes", "owed", "due", "dues", "pending", "paid", "unpaid",
    "balance", "outstanding", "total", "kitna", "kitni", "kitne", "kaun", "kis",
    "customer", "grahak", "ग्राहक", "khata", "ledger", "payment", "paisa",
    "rupay", "rupaye", "₹", "defaulter", "month", "mahina", "महीना",
    "sabse", "zyada", "kam", "how much", "how many", "who ", "amount", "list",
    "most", "least", "lowest", "smallest", "highest", "top", "biggest",
    # Sales / billing vocabulary (Phase 8) — a kirana store sells goods too.
    "bikri", "बिक्री", "becha", "bechi", "bech", "sale", "sales", "sold", "sell",
    "selling", "bill", "invoice", "revenue",
)


def _matches_any(text: str, patterns) -> bool:
    return any(re.search(p, text) for p in patterns)


def _local_guardrail(question: str, known_names: List[str]) -> Optional[bool]:
    """Fast, no-LLM scope check. True=in, False=out, None=undecided (ask the LLM)."""
    text = question.lower()
    if _matches_any(text, _OOS_PATTERNS):
        return False
    if any(hint in text for hint in _IN_SCOPE_HINTS):
        return True
    if any(name and name.lower() in text for name in known_names):
        return True
    return None


def _classify_scope(question: str, complete_fn: CompleteFn) -> Optional[bool]:
    """LLM scope classifier. Returns True/False, or None when the LLM is unavailable."""
    raw = complete_fn(
        "You are a precise scope classifier. Reply with exactly one word.",
        prompts.INSIGHTS_GUARDRAIL_PROMPT.format(question=question),
        max_tokens=8,
    )
    if not raw:
        return None
    verdict = raw.strip().upper()
    if "OUT_OF_SCOPE" in verdict:  # check OUT first — "IN_SCOPE" is a substring of it
        return False
    if "IN_SCOPE" in verdict:
        return True
    return None


def _is_in_scope(question: str, known_names: List[str], complete_fn: CompleteFn) -> bool:
    """Two-tier guardrail: local keywords first, then the LLM only if still unsure."""
    local = _local_guardrail(question, known_names)
    if local is not None:
        return local
    verdict = _classify_scope(question, complete_fn)  # None if no LLM (mock/offline)
    if verdict is not None:
        return verdict
    # No clear signal and no LLM to arbitrate: lean permissive so an offline/mock
    # demo still answers from the ledger. A clear OOS pattern already refused above.
    return True


# ── Deterministic mini-router (keeps the money numbers off the LLM path) ─────
_TOTAL_PATTERNS = (
    r"\btotal\b.*\b(outstanding|due|owed|udhaar|udhar|baki|baaki|pending|balance)\b",
    r"\b(outstanding|udhaar|udhar|baki|baaki)\b.*\btotal\b",
    r"\bhow much\b.*\b(total|outstanding|altogether|in all)\b",
    r"\b(kitna|kitni|kitne)\b.*\b(total|udhaar|udhar|baki|baaki)\b",
)
_TOP_PATTERNS = (
    r"\b(top|biggest|largest|highest)\b.*\b(defaulters?|debtors?|borrowers?)\b",
    r"\b(top|biggest|largest|highest)\b.*\bowes?\b",           # "top 5 who owe..."
    r"\bwho\b.*\bowes?\b.*\b(most|the most|highest|maximum)\b",
    r"\b(sabse zyada|sabse bada)\b.*\b(udhaar|udhar|baki|baaki)\b",
    r"\bkaun\b.*\bsabse zyada\b",
)
# "who owes the LEAST" — the mirror of _TOP_PATTERNS. Checked separately so it is
# never grabbed by the top branch (top requires most/highest, never least).
_LEAST_PATTERNS = (
    r"\b(lowest|smallest)\b.*\b(defaulters?|debtors?|borrowers?|balances?|udhaar|udhar|baki|baaki)\b",
    r"\bwho\b.*\bowes?\b.*\b(least|the least|lowest|smallest|minimum)\b",
    r"\b(sabse kam|sabse chhota|sabse chota)\b.*\b(udhaar|udhar|baki|baaki)\b",
    r"\bkaun\b.*\bsabse kam\b",
)


def _parse_count(question: str, cap: int = 10) -> Optional[int]:
    """Pull the requested list size out of a question ("top 5", "3 customers").
    Returns None when no count is given (caller then uses its singular default),
    else the count clamped to [1, cap] so a huge number can't blow up the answer."""
    text = question.lower()
    m = re.search(r"\b(?:top|first|list|show|give(?:\s+me)?|name)\D{0,12}(\d{1,3})\b", text)
    if not m:
        m = re.search(
            r"\b(\d{1,3})\s+(?:people|persons?|customers?|names?|log|logo|naam|"
            r"defaulters?|debtors?|borrowers?)\b",
            text,
        )
    if not m:
        return None
    return max(1, min(int(m.group(1)), cap))
# Sales / billing questions (Phase 8). Any explicit sales word routes here; kept
# separate from _TOTAL_PATTERNS (which is udhaar-outstanding) and CHECKED FIRST in
# the router so "how much did I sell in total" isn't grabbed by the total pattern.
_SALES_PATTERNS = (
    r"\b(bikri|becha|bechi|bech)\b",                 # Hindi/Hinglish: sale / sold
    r"\b(sold|sell|selling|sale|sales|revenue)\b",   # English
)


def _sales_answer(question: str, stats: LedgerStats) -> InsightAnswer:
    """Answer a sales question straight from SQL-computed sales stats (no LLM).
    'received' is total_sales minus pending invoices — deterministic arithmetic,
    not an LLM guess."""
    if not stats.total_sales:
        return InsightAnswer(
            question=question, answer="No sales are recorded in your ledger yet.",
            source="deterministic",
        )
    received = round(stats.total_sales - stats.unpaid_invoices_total, 2)
    answer = (
        f"Your recorded sales total ₹{stats.total_sales:,.2f} — ₹{received:,.2f} received "
        f"and ₹{stats.unpaid_invoices_total:,.2f} still pending as invoices."
    )
    if stats.sales_this_month:
        answer += f" This month: ₹{stats.sales_this_month:,.2f}."
    return InsightAnswer(question=question, answer=answer, source="deterministic")


def _ranked_answer(question: str, balances: list, *, least: bool) -> InsightAnswer:
    """Build a deterministic 'top/least N' answer from the FULL sorted balance list
    (highest-first). Honors an explicit count from the question ("top 5"); with no
    count it gives the single most/least, plus two for context on the 'most' side."""
    if not balances:
        return InsightAnswer(
            question=question, answer="No customer currently has an outstanding balance.",
            source="deterministic",
        )
    ordered = list(reversed(balances)) if least else balances  # smallest-first for 'least'
    n = _parse_count(question)
    if n is None:
        one = ordered[0]
        superlative = "least" if least else "most"
        answer = f"{one.name} owes you the {superlative}, at ₹{one.unpaid_total:,.2f}."
        if not least:  # a little context is natural for "who owes the most"
            others = ordered[1:3]
            if others:
                answer += " Next: " + ", ".join(
                    f"{d.name} (₹{d.unpaid_total:,.0f})" for d in others
                ) + "."
        shown = ordered[:3]
    else:
        shown = ordered[:n]
        listed = "; ".join(
            f"{i}. {d.name} — ₹{d.unpaid_total:,.2f}" for i, d in enumerate(shown, 1)
        )
        label = "lowest" if least else "top"
        if len(shown) < n:  # they asked for more than exist — say so, don't pad
            answer = (
                f"Only {len(shown)} customer(s) have an outstanding balance — "
                f"{label} {len(shown)}: {listed}."
            )
        else:
            answer = f"{label.capitalize()} {len(shown)} by outstanding balance: {listed}."
    return InsightAnswer(
        question=question, answer=answer, source="deterministic",
        citations=[f"{d.name}: ₹{d.unpaid_total:,.2f}" for d in shown],
    )


def _deterministic_router(question: str, stats: LedgerStats, conn) -> Optional[InsightAnswer]:
    """Answer the unambiguous money questions straight from SQL-computed figures.

    Returns None for anything nuanced, which then flows to RAG+LLM. Deterministic
    answers are templated English; a nuanced Hindi question instead goes to the LLM
    (which still receives these exact totals as ground truth, so numbers stay honest).
    """
    text = question.lower()
    if _matches_any(text, _SALES_PATTERNS):
        return _sales_answer(question, stats)
    if _matches_any(text, _TOTAL_PATTERNS):
        answer = (
            f"Your customers owe you ₹{stats.total_outstanding:,.2f} in total, across "
            f"{stats.customer_count} customer(s) with a pending balance."
        )
        return InsightAnswer(question=question, answer=answer, source="deterministic")
    # Ranking questions use the FULL balance list (not the capped top_defaulters), so
    # "top 10" is answerable and "who owes least" is exact — check least before top
    # since "least" is the more specific word.
    if _matches_any(text, _LEAST_PATTERNS):
        return _ranked_answer(question, get_all_balances(conn), least=True)
    if _matches_any(text, _TOP_PATTERNS):
        return _ranked_answer(question, get_all_balances(conn), least=False)
    return None


# ── RAG context formatting ───────────────────────────────────────────────────
def _build_context(hits) -> str:
    """Render retrieved entries as citation-tagged lines for the answer prompt."""
    if not hits:
        return "(no matching entries found)"
    lines = []
    for h in hits:
        flag = " | FLAGGED: needs review" if h.record.needs_review else ""
        lines.append(f"{h.citation()} | source image: {h.record.source_image}{flag}")
    return "\n".join(lines)


# ── Public entry point ───────────────────────────────────────────────────────
def ask(
    question: str,
    conn,
    *,
    index: Optional[LedgerSearchIndex] = None,
    complete_fn: Optional[CompleteFn] = None,
    k: int = SEARCH_TOP_K,
) -> InsightAnswer:
    """Answer one question about the ledger. Always returns a tagged InsightAnswer."""
    question = (question or "").strip()
    if not question:
        return InsightAnswer(
            question="", answer="Please ask a question about your ledger.", source="empty",
        )

    complete_fn = complete_fn or groq_chat
    stats = compute_stats(conn)
    known_names = [d.name for d in stats.top_defaulters]

    # 1) Guardrail — refuse out-of-scope questions politely.
    if not _is_in_scope(question, known_names, complete_fn):
        return InsightAnswer(
            question=question, answer=prompts.INSIGHTS_REFUSAL_MESSAGE,
            source="guardrail_refusal", is_blocked=True,
        )

    # 2) Deterministic router — money numbers answered without the LLM.
    routed = _deterministic_router(question, stats, conn)
    if routed is not None:
        return routed

    # 3) Retrieve relevant entries via hybrid search.
    if index is None:
        index = LedgerSearchIndex(get_all_entries(conn))
    hits = index.search(question, k=k)
    if not hits:
        return InsightAnswer(
            question=question,
            answer="I couldn't find anything about that in your ledger yet.",
            source="empty",
        )
    citations = [h.citation() for h in hits]
    hit_ids = [h.record.id for h in hits]

    # 4) Generate a grounded answer; fall back to extractive if the LLM is down.
    user_prompt = prompts.INSIGHTS_ANSWER_PROMPT.format(
        stats="\n".join(stats.headline_lines()),
        context=_build_context(hits),
        question=question,
    )
    answer = complete_fn(prompts.INSIGHTS_SYSTEM_PROMPT, user_prompt)
    if answer:
        return InsightAnswer(
            question=question, answer=answer.strip(), source="llm",
            citations=citations, hit_ids=hit_ids,
        )
    return InsightAnswer(
        question=question, answer=extractive_answer(question, hits),
        source="extractive_fallback", degraded=True, citations=citations, hit_ids=hit_ids,
    )


class InsightsAgent:
    """Stateful convenience wrapper: builds the search index once, reuses it.

    The Streamlit UI (Phase 6) holds one of these; call refresh_index() after new
    pages are ingested. compute_stats() is exposed for the deterministic stat cards.
    """

    def __init__(self, conn, *, complete_fn: Optional[CompleteFn] = None, build_index: bool = True):
        self.conn = conn
        self.complete_fn = complete_fn or groq_chat
        self.index = LedgerSearchIndex(get_all_entries(conn)) if build_index else None

    def stats(self) -> LedgerStats:
        return compute_stats(self.conn)

    def refresh_index(self) -> None:
        self.index = LedgerSearchIndex(get_all_entries(self.conn))

    def ask(self, question: str) -> InsightAnswer:
        return ask(question, self.conn, index=self.index, complete_fn=self.complete_fn)
