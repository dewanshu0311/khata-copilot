"""
Phase 2 tests — the Verification Agent audits and self-corrects, never fixes.

Every case builds a PageExtraction directly (no API needed), so the whole suite
runs offline in milliseconds. The self-correction test drives the loop with a
fake vision callable and also proves mock mode terminates instead of looping.

Run from the project root:
    .venv\\Scripts\\python.exe -m pytest tests/test_verification.py -v
"""
from __future__ import annotations

import os

from agents.verification_agent import audit, verify_page
from core.config import REVIEW_CONFIDENCE_THRESHOLD
from core.schemas import LedgerEntry, PageExtraction


# ── Helpers ──────────────────────────────────────────────────────────────────
def _entry(name="Ramesh", amount=100.0, date="5 Jan", status="unpaid", confidence=0.95):
    return LedgerEntry(
        name=name, amount=amount, date=date, status=status,
        confidence=confidence, raw_text=f"{name} {amount}",
    )


def _codes(result):
    return {i.code for i in result.issues}


# ── 1. Math mismatch: written_total != sum of entries ────────────────────────
def test_math_mismatch_flagged():
    page = PageExtraction(
        source_image="p.jpg",
        entries=[_entry("A", 1000), _entry("B", 500)],  # sum = 1500
        written_total=2000,                             # off by 500
        overall_confidence=0.95,
    )
    result = audit(page)
    assert "math_mismatch" in _codes(result)
    assert result.verdict == "needs_review"
    assert result.total_difference == 500.0        # written - computed
    assert result.error_count >= 1                 # math mismatch is an error


# ── 2. Absurd amount (₹99,00,000) ────────────────────────────────────────────
def test_absurd_amount_flagged():
    page = PageExtraction(
        source_image="p.jpg",
        entries=[_entry("A", 100), _entry("Whale", 9_900_000)],
        overall_confidence=0.95,
    )
    result = audit(page)
    assert "absurd_amount" in _codes(result)
    assert result.verdict == "needs_review"
    # The clean entry must NOT be flagged absurd — only the whale.
    absurd = [i for i in result.issues if i.code == "absurd_amount"]
    assert len(absurd) == 1 and absurd[0].entry_name == "Whale"


# ── 3. Missing date + unknown status ─────────────────────────────────────────
def test_missing_date_and_unknown_status_flagged():
    page = PageExtraction(
        source_image="p.jpg",
        entries=[_entry("A", 100, date=None, status="unknown")],
        overall_confidence=0.95,
    )
    result = audit(page)
    codes = _codes(result)
    assert "missing_date" in codes
    assert "unknown_status" in codes


# ── 4. All-low-confidence page ───────────────────────────────────────────────
def test_all_low_confidence_page():
    low = REVIEW_CONFIDENCE_THRESHOLD - 0.3
    page = PageExtraction(
        source_image="p.jpg",
        entries=[_entry("A", 100, confidence=low), _entry("B", 200, confidence=low)],
        overall_confidence=low,
    )
    result = audit(page)
    low_conf = [i for i in result.issues if i.code == "low_confidence"]
    assert len(low_conf) == 2                       # every entry flagged
    assert result.verdict == "needs_review"


# ── 5. A clean page is accepted ──────────────────────────────────────────────
def test_clean_page_accepted():
    page = PageExtraction(
        source_image="p.jpg",
        entries=[_entry("A", 1000), _entry("B", 500)],
        written_total=1500,                          # matches sum
        overall_confidence=0.95,
    )
    result = audit(page)
    assert result.verdict == "accept"
    assert result.error_count == 0 and result.warning_count == 0


# ── 6. Self-correction: bounded re-read that improves the page ───────────────
def test_self_correction_uses_better_reread():
    bad = PageExtraction(
        source_image="p.jpg",
        entries=[_entry("A", 1000), _entry("B", 500)],  # sum = 1500
        written_total=2000,                             # 25% gap -> triggers re-read
        overall_confidence=0.95,
    )
    good = PageExtraction(
        source_image="p.jpg",
        entries=[_entry("A", 1000), _entry("B", 1000)],  # sum = 2000, now matches
        written_total=2000,
        overall_confidence=0.97,
    )
    calls = {"n": 0}

    def fake_extract(image_path, correction_feedback=None):
        calls["n"] += 1
        assert correction_feedback  # feedback must be injected on the re-read
        return good

    result = verify_page(bad, extract_fn=fake_extract)
    assert calls["n"] == 1                 # exactly one bounded re-extraction
    assert result.corrected is True
    assert result.verdict == "accept"      # corrected page is clean
    assert "math_mismatch" not in _codes(result)


# ── 7. Self-correction terminates when the re-read doesn't help ──────────────
def test_self_correction_no_improvement_does_not_loop():
    bad = PageExtraction(
        source_image="p.jpg",
        entries=[_entry("A", 1000), _entry("B", 500)],
        written_total=2000,
        overall_confidence=0.95,
    )
    calls = {"n": 0}

    def same_bad_extract(image_path, correction_feedback=None):
        calls["n"] += 1
        return bad  # re-read returns the same page (mimics mock mode)

    result = verify_page(bad, extract_fn=same_bad_extract, max_retries=1)
    assert calls["n"] == 1                 # tried once, then stopped — no loop
    assert result.verdict == "needs_review"
    assert "math_mismatch" in _codes(result)


# ── 8. Mock mode end-to-end: real vision agent, no crash, no loop ────────────
def test_self_correction_in_mock_mode(monkeypatch):
    monkeypatch.setenv("KHATA_MOCK", "1")
    from agents import vision_agent

    page = vision_agent.extract_page("sample_data/whatever.jpg")
    assert page.degraded is True
    # Audit + (bounded) self-correction against the real mock vision agent.
    result = verify_page(page)              # extract_fn defaults to the real agent
    assert result.verdict == "needs_review"  # mock page is degraded
    # It must have terminated with a valid result (the assertion above proves no
    # exception/infinite loop occurred).
    assert result.source_image == page.source_image
