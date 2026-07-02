"""
Sequential orchestrator — runs one photographed khata page through the full
pipeline: Vision -> Verification -> Ledger ingest.

Deliberately hand-rolled and linear (NOT CrewAI, per the project's stack
constraint) — three function calls in a row, each wrapped so a failure in one
stage degrades the result instead of crashing the other two. Reuses each
agent's own graceful-failure contract (vision_agent and verify_page already
never raise on the failure paths they know about); the try/except here is the
last-resort guard for anything unexpected, so process_page() itself never raises.
"""
from __future__ import annotations

from typing import List, Optional

from agents.verification_agent import ExtractFn, verify_page
from core.db import ingest_page
from core.schemas import EntryIssue, PageExtraction, PageResult, VerificationResult


def _default_extract_fn():
    from agents.vision_agent import extract_page  # lazy: keep vision deps optional
    return extract_page


def process_page(
    image_path: str, conn, *, extract_fn: Optional[ExtractFn] = None,
) -> PageResult:
    """Run Vision -> Verification -> Ledger ingest for one page. Always returns a PageResult."""
    extract_fn = extract_fn or _default_extract_fn()
    stage_errors: List[str] = []
    degraded = False

    try:
        extraction = extract_fn(image_path)
    except Exception as e:  # noqa: BLE001 — total: this stage must never crash the pipeline
        stage_errors.append(f"Vision stage failed: {e}")
        degraded = True
        extraction = PageExtraction(
            source_image=image_path, entries=[], degraded=True, error=str(e),
        )

    try:
        verification = verify_page(extraction, extract_fn=extract_fn)
    except Exception as e:  # noqa: BLE001
        stage_errors.append(f"Verification stage failed: {e}")
        degraded = True
        verification = VerificationResult(
            source_image=extraction.source_image,
            verdict="needs_review",
            issues=[EntryIssue(
                entry_index=-1, entry_name="", code="degraded_extraction", severity="error",
                message=f"Verification could not run: {e}",
            )],
            overall_confidence=extraction.overall_confidence,
            computed_total=extraction.computed_total,
            written_total=extraction.written_total,
        )

    ingest = None
    try:
        ingest = ingest_page(conn, extraction, verification)
    except Exception as e:  # noqa: BLE001
        stage_errors.append(f"Ledger ingest failed: {e}")
        degraded = True

    degraded = degraded or extraction.degraded

    return PageResult(
        source_image=image_path,
        extraction=extraction,
        verification=verification,
        ingest=ingest,
        degraded=degraded,
        stage_errors=stage_errors,
    )
