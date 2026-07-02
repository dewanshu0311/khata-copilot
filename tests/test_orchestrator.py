"""
Phase 5 tests — the sequential orchestrator runs Vision -> Verification ->
Ledger ingest for one photographed page.

Every case uses an in-memory database (':memory:') and KHATA_MOCK=1 (no real
Gemini/Groq calls), so the suite runs offline in milliseconds.

Run from the project root:
    .venv\\Scripts\\python.exe -m pytest tests/test_orchestrator.py -v
"""
from __future__ import annotations

import os

from core.db import connect, get_all_balances
from core.orchestrator import process_page
from core.schemas import LedgerEntry, PageExtraction

os.environ["KHATA_MOCK"] = "1"  # force the whole pipeline offline


def _db():
    return connect(":memory:")


def _fake_extract(image_path, correction_feedback=None):
    return PageExtraction(
        source_image=image_path,
        entries=[
            LedgerEntry(name="Ramesh", amount=500, date="05/01/2026", status="unpaid",
                        confidence=0.95, raw_text="Ramesh 500"),
            LedgerEntry(name="Sita", amount=300, date="06/01/2026", status="paid",
                        confidence=0.9, raw_text="Sita 300"),
        ],
        overall_confidence=0.93,
    )


# ── 1. process_page runs end-to-end using the default (mock) vision agent ────
def test_process_page_runs_end_to_end_in_mock_mode():
    conn = _db()
    result = process_page("sample_data/page1.jpg", conn)

    assert result.source_image == "sample_data/page1.jpg"
    assert result.extraction.entries  # mock extraction always has entries
    assert result.verification.source_image == "sample_data/page1.jpg"
    assert result.ingest is not None
    assert result.ingest.inserted > 0


# ── 2. A valid summary is returned and entries land in the ledger ───────────
def test_process_page_ingests_entries_with_injected_extractor():
    conn = _db()
    result = process_page("p1.jpg", conn, extract_fn=_fake_extract)

    assert result.ingest.inserted == 2
    balances = {b.name: b for b in get_all_balances(conn)}
    assert balances["Ramesh"].unpaid_total == 500
    assert "Sita" not in balances  # fully paid


# ── 3. A vision-stage failure degrades gracefully instead of crashing ───────
def test_process_page_survives_vision_failure():
    conn = _db()

    def _boom(image_path, correction_feedback=None):
        raise RuntimeError("camera roll corrupted")

    result = process_page("bad.jpg", conn, extract_fn=_boom)

    assert result.degraded is True
    assert result.stage_errors  # at least one recorded failure
    assert result.extraction.entries == []
    assert result.ingest is not None  # ledger ingest still runs on the empty page
