"""
Verification Agent — audits a PageExtraction and FLAGS problems.

Pipeline role (Phase 2): PageExtraction in -> VerificationResult out.

Core contract (from the spec): this agent FLAGS, it NEVER silently fixes data.
Self-flagging is the product feature — a wrong number a human is told to check
is far safer than a wrong number presented as correct.

Two layers:
  1. AUDIT — pure-Python checks (math, plausibility, completeness, confidence).
     No LLM involved; every rule is judge-explainable.
  2. SELF-CORRECTION — if the audit suggests the whole PAGE was likely misread
     (math badly off, or too many low-confidence entries), the agent may ask the
     Vision Agent for ONE bounded re-read with targeted feedback, then re-audit
     and keep whichever result is better. Bounded by MAX_VERIFICATION_RETRIES so
     it can never loop, and it degrades gracefully in mock mode / on any error.

Shape adapted (not code-copied) from project-overwatch's self_correction_loop:
produce -> critique -> conditional bounded re-produce with feedback injected.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from core.config import (
    LOW_CONF_FRACTION_TRIGGER,
    MATH_MISMATCH_REL_TRIGGER,
    MATH_MISMATCH_TOLERANCE,
    MAX_PLAUSIBLE_AMOUNT,
    MAX_VERIFICATION_RETRIES,
    REVIEW_CONFIDENCE_THRESHOLD,
)
from core.schemas import EntryIssue, PageExtraction, VerificationResult

# Type of the vision callable we re-invoke for a correction re-read. Injected as
# a parameter (default: the real vision agent) so tests can pass a fake and we
# avoid a hard import cycle.
ExtractFn = Callable[..., PageExtraction]


# ── Layer 1: the audit (pure Python, no LLM) ─────────────────────────────────
def _audit_entries(page: PageExtraction, threshold: float) -> List[EntryIssue]:
    """Per-entry plausibility, completeness and confidence checks."""
    issues: List[EntryIssue] = []
    for idx, entry in enumerate(page.entries):
        name = entry.name or "(blank)"

        if not entry.name or entry.name.upper() == "UNREADABLE":
            issues.append(EntryIssue(
                entry_index=idx, entry_name=name, code="missing_name", severity="warning",
                message="Customer name is blank or unreadable.",
            ))

        if entry.amount <= 0:
            issues.append(EntryIssue(
                entry_index=idx, entry_name=name, code="nonpositive_amount", severity="warning",
                message=f"Amount is {entry.amount:g} — zero or negative, likely misread.",
            ))
        elif entry.amount > MAX_PLAUSIBLE_AMOUNT:
            issues.append(EntryIssue(
                entry_index=idx, entry_name=name, code="absurd_amount", severity="warning",
                message=f"Amount ₹{entry.amount:,.0f} is implausibly large for one line.",
            ))

        if not entry.date:
            issues.append(EntryIssue(
                entry_index=idx, entry_name=name, code="missing_date", severity="info",
                message="No date recorded for this entry.",
            ))

        if entry.status == "unknown":
            issues.append(EntryIssue(
                entry_index=idx, entry_name=name, code="unknown_status", severity="info",
                message="Paid/unpaid status could not be determined.",
            ))

        if entry.confidence <= threshold:
            issues.append(EntryIssue(
                entry_index=idx, entry_name=name, code="low_confidence", severity="warning",
                message=f"Reader confidence {entry.confidence:.2f} is at/below the {threshold:.2f} review threshold.",
            ))
    return issues


def _audit_math(page: PageExtraction) -> Optional[EntryIssue]:
    """Compare a written page total against the sum of entry amounts."""
    if page.written_total is None:
        return None
    diff = page.written_total - page.computed_total
    if abs(diff) <= MATH_MISMATCH_TOLERANCE:
        return None
    return EntryIssue(
        entry_index=-1, entry_name="", code="math_mismatch", severity="error",
        message=(
            f"Written total ₹{page.written_total:,.2f} ≠ sum of entries "
            f"₹{page.computed_total:,.2f} (off by ₹{diff:,.2f})."
        ),
    )


def audit(page: PageExtraction, threshold: float = REVIEW_CONFIDENCE_THRESHOLD) -> VerificationResult:
    """Run every check on a page and return a VerificationResult. No LLM, no loop."""
    issues = _audit_entries(page, threshold)

    math_issue = _audit_math(page)
    if math_issue:
        issues.append(math_issue)

    if page.degraded:
        issues.append(EntryIssue(
            entry_index=-1, entry_name="", code="degraded_extraction", severity="warning",
            message="Page came from a degraded extraction (mock or failed/partial read).",
        ))

    total_difference = (
        round(page.written_total - page.computed_total, 2)
        if page.written_total is not None else None
    )

    # needs_review if anything actionable was found, the page is degraded, or the
    # aggregate confidence is itself below the review bar.
    has_actionable = any(i.severity in ("warning", "error") for i in issues)
    below_conf = page.overall_confidence <= threshold
    verdict = "needs_review" if (has_actionable or page.degraded or below_conf) else "accept"

    return VerificationResult(
        source_image=page.source_image,
        verdict=verdict,
        issues=issues,
        overall_confidence=page.overall_confidence,
        computed_total=page.computed_total,
        written_total=page.written_total,
        total_difference=total_difference,
    )


# ── Layer 2: self-correction trigger ─────────────────────────────────────────
def _correction_feedback(page: PageExtraction, result: VerificationResult, threshold: float) -> Optional[str]:
    """Decide whether the PAGE (not just a line) looks misread, and why.

    Returns feedback text to inject into a re-read, or None if a re-read isn't
    warranted. Only whole-page signals qualify — a single soft flag does not.
    """
    # Signal 1: math badly off (relative gap, guarded against divide-by-zero).
    if page.written_total is not None:
        diff = abs(page.written_total - page.computed_total)
        denom = abs(page.written_total) or 1.0
        if diff > MATH_MISMATCH_TOLERANCE and (diff / denom) > MATH_MISMATCH_REL_TRIGGER:
            return (
                f"The amounts you read sum to ₹{page.computed_total:,.2f}, but the total written on "
                f"the page reads ₹{page.written_total:,.2f} — a gap of ₹{diff:,.2f}. "
                "Recount each amount and re-check any ambiguous digits."
            )

    # Signal 2: systemic low confidence across the page.
    if page.entries:
        low = sum(1 for e in page.entries if e.confidence <= threshold)
        if low / len(page.entries) >= LOW_CONF_FRACTION_TRIGGER:
            return (
                f"{low} of {len(page.entries)} entries were read with low confidence. "
                "Re-read the page slowly and re-check smudged or ambiguous handwriting."
            )
    return None


def _is_better(candidate: VerificationResult, incumbent: VerificationResult) -> bool:
    """A re-read wins only if it clearly improves the audit outcome."""
    cand_errors = candidate.error_count + candidate.warning_count
    inc_errors = incumbent.error_count + incumbent.warning_count
    if cand_errors != inc_errors:
        return cand_errors < inc_errors
    return candidate.overall_confidence > incumbent.overall_confidence


# ── Public entry point ───────────────────────────────────────────────────────
def verify_page(
    page: PageExtraction,
    extract_fn: Optional[ExtractFn] = None,
    threshold: float = REVIEW_CONFIDENCE_THRESHOLD,
    max_retries: int = MAX_VERIFICATION_RETRIES,
) -> VerificationResult:
    """Audit a page and, if it looks misread, attempt bounded re-extraction.

    extract_fn: the vision callable used for a correction re-read. Defaults to
        the real vision agent; tests inject a fake. Kept as a parameter to avoid
        an import cycle and to keep the loop testable offline.

    Always returns a VerificationResult — every failure path degrades to the best
    audit we have, never an exception.
    """
    result = audit(page, threshold)
    best_page, best_result = page, result

    if max_retries <= 0:
        return best_result

    if extract_fn is None:
        # Lazy import so importing this module never drags in the SDK, and so the
        # audit-only path works even if vision deps are absent.
        try:
            from agents.vision_agent import extract_page as extract_fn  # type: ignore
        except Exception:
            return best_result

    attempts = 0
    while attempts < max_retries:
        feedback = _correction_feedback(best_page, best_result, threshold)
        if not feedback:
            break  # audit doesn't think a re-read would help — stop.
        attempts += 1
        try:
            reread = extract_fn(best_page.source_image, correction_feedback=feedback)
        except Exception as e:
            best_result.correction_note = (
                f"Attempted {attempts} re-extraction(s); last one errored ({e}). "
                "Kept the original audit."
            )
            break

        reread_result = audit(reread, threshold)
        if _is_better(reread_result, best_result):
            best_page, best_result = reread, reread_result
            best_result.corrected = True
            best_result.correction_note = (
                f"Re-extraction #{attempts} improved the reading; using the corrected page. "
                "Data was re-read, never hand-edited."
            )
            # If the re-read is now clean, no reason to keep spending retries.
            if best_result.verdict == "accept":
                break
        else:
            best_result.corrected = True
            best_result.correction_note = (
                f"Re-extraction #{attempts} did not improve the reading; kept the original audit. "
                "Flagged for human review."
            )
            break  # a non-improving re-read (e.g. mock returns the same page) — don't loop.

    return best_result
