"""
Phase 7 tests — the Vision Agent's reader order: Gemini -> Groq vision -> mock.

Real Gemini/Groq calls are never made here. The Gemini/Groq attempt functions
are monkeypatched at the module level to simulate success/failure, so these
tests run offline and prove the ORCHESTRATION (which reader is tried when, and
in what order) without depending on network access or real keys.

Run from the project root:
    .venv\\Scripts\\python.exe -m pytest tests/test_vision.py -v
"""
from __future__ import annotations

import agents.vision_agent as vision_agent
from core.schemas import PageExtraction


def _fake_extraction(name: str, notes: str = "") -> PageExtraction:
    return PageExtraction(source_image="p.jpg", entries=[], overall_confidence=0.9, notes=notes)


# ── 1. Gemini succeeds -> Groq is never tried ────────────────────────────────
def test_gemini_success_skips_groq(monkeypatch):
    calls = {"groq": 0}

    def fake_gemini(image_path, feedback):
        return _fake_extraction("gemini"), None

    def fake_groq(image_path, feedback):
        calls["groq"] += 1
        return _fake_extraction("groq"), None

    monkeypatch.setattr(vision_agent, "_load_image", lambda p: type("I", (), {"close": lambda self: None})())
    monkeypatch.setattr(vision_agent, "_extract_page_gemini", fake_gemini)
    monkeypatch.setattr(vision_agent, "_extract_page_groq_vision", fake_groq)
    monkeypatch.setenv("KHATA_MOCK", "0")

    result = vision_agent.extract_page("p.jpg")
    assert not result.degraded
    assert calls["groq"] == 0


# ── 2. Gemini fails -> Groq vision fallback is tried and succeeds ───────────
def test_gemini_failure_falls_back_to_groq(monkeypatch):
    def fake_gemini(image_path, feedback):
        return None, "quota exceeded"

    def fake_secondary(image_path, feedback):
        return None, "no secondary vision key configured"

    def fake_groq(image_path, feedback):
        return _fake_extraction("groq", notes="Read via Groq vision fallback"), None

    monkeypatch.setattr(vision_agent, "_load_image", lambda p: type("I", (), {"close": lambda self: None})())
    monkeypatch.setattr(vision_agent, "_extract_page_gemini", fake_gemini)
    monkeypatch.setattr(vision_agent, "_extract_page_secondary", fake_secondary)
    monkeypatch.setattr(vision_agent, "_extract_page_groq_vision", fake_groq)
    monkeypatch.setenv("KHATA_MOCK", "0")

    result = vision_agent.extract_page("p.jpg")
    assert not result.degraded
    assert "Groq vision fallback" in result.notes


# ── 3. Both Gemini and Groq fail -> mock last resort, reason names both ─────
def test_both_readers_fail_falls_back_to_mock(monkeypatch):
    def fake_gemini(image_path, feedback):
        return None, "quota exceeded"

    def fake_secondary(image_path, feedback):
        return None, "no secondary vision key configured"

    def fake_groq(image_path, feedback):
        return None, "no GROQ_API_KEY configured"

    monkeypatch.setattr(vision_agent, "_load_image", lambda p: type("I", (), {"close": lambda self: None})())
    monkeypatch.setattr(vision_agent, "_extract_page_gemini", fake_gemini)
    monkeypatch.setattr(vision_agent, "_extract_page_secondary", fake_secondary)
    monkeypatch.setattr(vision_agent, "_extract_page_groq_vision", fake_groq)
    monkeypatch.setenv("KHATA_MOCK", "0")

    result = vision_agent.extract_page("p.jpg")
    assert result.degraded
    assert "quota exceeded" in result.notes
    assert "GROQ_API_KEY" in result.notes
    assert len(result.entries) == 4  # the canned mock page


# ── 4. KHATA_MOCK=1 short-circuits before either reader is attempted ────────
def test_mock_mode_skips_both_readers(monkeypatch):
    calls = {"gemini": 0, "groq": 0}

    def fake_gemini(image_path, feedback):
        calls["gemini"] += 1
        return _fake_extraction("gemini"), None

    def fake_groq(image_path, feedback):
        calls["groq"] += 1
        return _fake_extraction("groq"), None

    monkeypatch.setattr(vision_agent, "_extract_page_gemini", fake_gemini)
    monkeypatch.setattr(vision_agent, "_extract_page_groq_vision", fake_groq)
    monkeypatch.setenv("KHATA_MOCK", "1")

    result = vision_agent.extract_page("p.jpg")
    assert result.degraded
    assert calls == {"gemini": 0, "groq": 0}


# ── 5. A genuinely bad image path is reported honestly, not masked as mock ──
def test_bad_image_path_is_not_masked_as_mock(monkeypatch):
    monkeypatch.setenv("KHATA_MOCK", "0")
    result = vision_agent.extract_page("this_file_does_not_exist.jpg")
    assert result.degraded
    assert result.error is not None
    assert "Could not open image" in result.error
    assert "MOCK MODE" not in result.notes
