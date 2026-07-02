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

# ── Demo safety ──────────────────────────────────────────────────────────────
# When set to "1" (or when no GEMINI key exists) the vision agent returns a
# canned extraction so the CLI/UI run fully offline. Always flagged degraded.
def mock_mode_enabled() -> bool:
    return os.getenv("KHATA_MOCK", "0").strip() == "1"
