"""
Central configuration constants for Khata Copilot.

Kept tiny and explicit so every value is judge-explainable. Anything an agent
tunes (model names, thresholds, retry counts) lives here — not scattered across
files.
"""
from __future__ import annotations

import os

# ── Models (all free tier) ──────────────────────────────────────────────────
# Gemini handles handwriting vision; Groq handles later reasoning phases.
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Confidence / review policy ───────────────────────────────────────────────
# Entries at or below this confidence get flagged for one-tap human review in
# the UI. The self-flagging IS the feature — we never silently "fix" data.
REVIEW_CONFIDENCE_THRESHOLD = float(os.getenv("REVIEW_CONFIDENCE_THRESHOLD", "0.8"))

# ── Retry / resilience ───────────────────────────────────────────────────────
MAX_VISION_RETRIES = int(os.getenv("MAX_VISION_RETRIES", "2"))
# Cooldown applied to a key after a 429/quota error (seconds). Gemini free-tier
# quotas reset per-minute, so ~62s matches the window.
KEY_COOLDOWN_SECONDS = float(os.getenv("KEY_COOLDOWN_SECONDS", "62.0"))

# ── Verification / audit policy (Phase 2) ────────────────────────────────────
# The Verification Agent AUDITS a PageExtraction — it flags problems, it never
# silently rewrites data. These constants tune what counts as a problem.
#
# How many times the agent may ask the Vision Agent to re-read a page it
# believes was misread. Bounded to prevent loops / burning free-tier quota.
MAX_VERIFICATION_RETRIES = int(os.getenv("MAX_VERIFICATION_RETRIES", "1"))
# An amount above this (₹10 lakh) is implausible for a single khata line and
# gets flagged for human review.
MAX_PLAUSIBLE_AMOUNT = float(os.getenv("MAX_PLAUSIBLE_AMOUNT", "1000000.0"))
# Written-vs-computed total gaps at or below this (rupees) are rounding noise,
# not a real mismatch.
MATH_MISMATCH_TOLERANCE = float(os.getenv("MATH_MISMATCH_TOLERANCE", "1.0"))
# A math mismatch only triggers a re-extraction when the gap exceeds this
# fraction of the written total (e.g. 0.2 = the sum is off by >20%). A small
# gap is flagged but not worth spending a re-read on.
MATH_MISMATCH_REL_TRIGGER = float(os.getenv("MATH_MISMATCH_REL_TRIGGER", "0.2"))
# If at least this fraction of entries are below the confidence threshold, the
# page was likely misread as a whole and is worth one re-extraction.
LOW_CONF_FRACTION_TRIGGER = float(os.getenv("LOW_CONF_FRACTION_TRIGGER", "0.5"))

# ── Ledger storage (Phase 3) ─────────────────────────────────────────────────
# SQLite file the Ledger Agent persists verified extractions into. Gitignored —
# every environment builds its own from scanned pages.
DB_PATH = os.getenv("KHATA_DB_PATH", "khata.db")

# ── Demo safety ──────────────────────────────────────────────────────────────
# When set to "1" (or when no GEMINI key exists) the vision agent returns a
# canned extraction so the CLI/UI run fully offline. Always flagged degraded.
def mock_mode_enabled() -> bool:
    return os.getenv("KHATA_MOCK", "0").strip() == "1"
