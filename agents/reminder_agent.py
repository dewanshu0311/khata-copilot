"""
Reminder Agent — drafts polite, copy-to-clipboard payment reminders.

Pipeline role (Phase 5): the ledger (via get_all_balances) in -> a ReminderDraft
per customer with an outstanding balance out.

Same anti-hallucination discipline as the Insights Agent: the AMOUNT is always
the exact figure from core.db.get_all_balances() (deterministic SQL). The LLM
(via core.groq_client, reused — not duplicated) only phrases the wording; it is
told to copy the amount verbatim. If the LLM is unavailable (mock mode, no key,
failed call) a fixed bilingual TEMPLATE fills in instead, so drafting never
fails. Every draft is tagged source="llm" or source="template", exactly like
InsightAnswer.source.

DRAFT ONLY: this module never sends a message anywhere (no WhatsApp/SMS
integration) — that is explicitly out of scope for this phase.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from core import prompts
from core.db import get_all_balances, get_all_entries
from core.groq_client import groq_chat
from core.schemas import ReminderDraft

# Same completion-callable shape as the Insights Agent: (system, user, **kwargs)
# -> answer text or None. Defaults to core.groq_client.groq_chat; injectable so
# tests run without real API calls.
CompleteFn = Callable[..., Optional[str]]


def _find_since_date(conn, customer_name: str) -> Optional[str]:
    """Earliest recorded raw_date for this customer's unpaid entries, if any.

    Never inferred or computed — just the earliest ledger entry that actually
    carries a date, in the words the shopkeeper wrote it. None if no unpaid
    entry has a date, which is honest: we simply don't say "since" at all.
    """
    dated = [
        e for e in get_all_entries(conn)
        if e.customer_name == customer_name and e.status != "paid" and e.raw_date
    ]
    if not dated:
        return None
    return min(dated, key=lambda e: e.id).raw_date


def _template_message(name: str, amount: float, since_date: Optional[str]) -> str:
    since_hi = f" ({since_date} se)" if since_date else ""
    since_en = f" since {since_date}" if since_date else ""
    return prompts.REMINDER_TEMPLATE.format(
        name=name, amount=amount, since_clause_hi=since_hi, since_clause_en=since_en,
    )


def draft_reminder_for(
    name: str, amount: float, conn, *, complete_fn: Optional[CompleteFn] = None,
    since_date: Optional[str] = None,
) -> ReminderDraft:
    """Draft one reminder. `amount` must already be the exact deterministic balance."""
    complete_fn = complete_fn or groq_chat
    since_clause = f"outstanding since {since_date}" if since_date else "no specific date on record"

    llm_text = complete_fn(
        prompts.REMINDER_SYSTEM_PROMPT,
        prompts.REMINDER_DRAFT_PROMPT.format(name=name, amount=f"{amount:,.2f}", since_clause=since_clause),
    )
    if llm_text:
        return ReminderDraft(
            customer_name=name, amount=amount, message=llm_text.strip(),
            source="llm", since_date=since_date,
        )
    return ReminderDraft(
        customer_name=name, amount=amount,
        message=_template_message(name, amount, since_date),
        source="template", degraded=True, since_date=since_date,
    )


def draft_reminders(conn, *, complete_fn: Optional[CompleteFn] = None) -> List[ReminderDraft]:
    """Draft a reminder for every customer with a positive outstanding balance.

    Fully-paid customers (unpaid_total <= 0, already excluded by
    get_all_balances()) get no draft at all — there is nothing to remind them of.
    """
    drafts = []
    for balance in get_all_balances(conn):
        since_date = _find_since_date(conn, balance.name)
        drafts.append(draft_reminder_for(
            balance.name, balance.unpaid_total, conn,
            complete_fn=complete_fn, since_date=since_date,
        ))
    return drafts
