"""
Vision Agent — reads a handwritten khata page into structured JSON.

Pipeline role (Phase 1): photo in -> validated PageExtraction out.

Design principles (from the spec):
  - Never fake functionality: confidence scores are reported honestly; the
    self-flagging of low-confidence entries IS the feature.
  - Never crash mid-demo: JSON-parse retries, Gemini key rotation on 429/quota
    (via core.key_manager), and a mock fallback so the CLI/UI run fully offline.

Uses Gemini free tier (gemini-2.0-flash). google-generativeai is imported
lazily so mock mode works even if the SDK isn't installed yet.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import ValidationError

from core import prompts
from core.config import (
    GEMINI_VISION_MODEL,
    KEY_COOLDOWN_SECONDS,
    MAX_VISION_RETRIES,
    mock_mode_enabled,
)
from core.key_manager import get_next_key, has_keys, mark_key_exhausted
from core.schemas import LedgerEntry, PageExtraction

_SERVICE = "GEMINI"
_RATE_LIMIT_MARKERS = ("429", "rate limit", "ratelimit", "quota", "resource_exhausted", "resource exhausted")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _is_rate_limit(error: Exception) -> bool:
    msg = str(error).lower()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def _strip_code_fences(text: str) -> str:
    """Gemini occasionally wraps JSON in ```json ... ``` despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _build_entries(raw_entries: list) -> tuple[list[LedgerEntry], list[str]]:
    """Validate entries defensively so one bad row can't discard the whole page."""
    entries: list[LedgerEntry] = []
    problems: list[str] = []
    for i, raw in enumerate(raw_entries or []):
        if not isinstance(raw, dict):
            problems.append(f"entry {i} was not an object")
            continue
        try:
            entries.append(LedgerEntry(**raw))
        except ValidationError:
            # Salvage what we can at low confidence rather than dropping it.
            salvage = LedgerEntry(
                name=str(raw.get("name") or "UNREADABLE"),
                amount=raw.get("amount", 0),
                date=raw.get("date"),
                status=raw.get("status", "unknown"),
                confidence=0.2,
                raw_text=str(raw.get("raw_text") or raw),
            )
            entries.append(salvage)
            problems.append(f"entry {i} failed strict validation and was flagged low-confidence")
    return entries, problems


def _extraction_from_payload(payload: dict, image_path: str, degraded: bool = False) -> PageExtraction:
    entries, problems = _build_entries(payload.get("entries", []))
    notes = str(payload.get("notes") or "").strip()
    if problems:
        notes = (notes + " | " if notes else "") + "; ".join(problems)
    return PageExtraction(
        source_image=image_path,
        entries=entries,
        written_total=payload.get("written_total"),
        overall_confidence=payload.get("overall_confidence", 0.0),
        notes=notes,
        degraded=degraded,
    )


# ── Mock mode (offline demo safety) ──────────────────────────────────────────
def _mock_extraction(image_path: str, reason: str) -> PageExtraction:
    """A canned, clearly-degraded extraction so the pipeline runs with no key."""
    payload = {
        "entries": [
            {"name": "Ramesh Kumar", "amount": 1200, "date": "5 Jan",
             "status": "unpaid", "confidence": 0.93, "raw_text": "Ramesh Kumar 1200 udhaar 5 Jan"},
            {"name": "Sita Devi", "amount": 450, "date": "6 Jan",
             "status": "paid", "confidence": 0.88, "raw_text": "Sita Devi 450 jama 6 Jan"},
            {"name": "Mohan", "amount": 800, "date": "6 Jan",
             "status": "unpaid", "confidence": 0.61, "raw_text": "Mohan 8?0 baki 6 Jan (digit unclear)"},
            {"name": "UNREADABLE", "amount": 0, "date": None,
             "status": "unknown", "confidence": 0.25, "raw_text": "smudged line"},
        ],
        "written_total": 2450,
        "notes": f"MOCK MODE ({reason}). Synthetic data — not a real reading.",
    }
    return _extraction_from_payload(payload, image_path, degraded=True)


# ── Gemini call ──────────────────────────────────────────────────────────────
def _load_image(image_path: str):
    from PIL import Image
    return Image.open(image_path)


def _configure_model(genai):
    """(Re)configure genai with the next key and return a fresh model."""
    api_key = get_next_key(_SERVICE)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        GEMINI_VISION_MODEL,
        system_instruction=prompts.VISION_SYSTEM_PROMPT,
    )
    return model, api_key


def extract_page(image_path: str) -> PageExtraction:
    """Read a handwritten khata page into a validated PageExtraction.

    Always returns a PageExtraction — on unrecoverable failure it is flagged
    degraded with an error message, never an exception, so the demo survives.
    """
    if mock_mode_enabled():
        return _mock_extraction(image_path, reason="KHATA_MOCK=1")
    if not has_keys(_SERVICE):
        return _mock_extraction(image_path, reason="no GEMINI_API_KEY configured")

    try:
        import google.generativeai as genai
    except ImportError:
        return _mock_extraction(image_path, reason="google-generativeai not installed")

    try:
        image = _load_image(image_path)
    except Exception as e:
        return PageExtraction(
            source_image=image_path, entries=[], degraded=True,
            error=f"Could not open image: {e}",
            notes="Check the file path and that it is a valid image.",
        )

    generation_config = {"temperature": 0.1, "response_mime_type": "application/json"}
    prompt = prompts.VISION_EXTRACTION_PROMPT
    last_error: Optional[str] = None

    model, api_key = _configure_model(genai)

    for attempt in range(MAX_VISION_RETRIES + 1):
        try:
            response = model.generate_content([prompt, image], generation_config=generation_config)
            text = _strip_code_fences(response.text or "")
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError("Model did not return a JSON object")
            return _extraction_from_payload(payload, image_path, degraded=False)

        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON from model: {e}"
            # Nudge the model to emit clean JSON on the retry.
            prompt = prompts.VISION_EXTRACTION_PROMPT + "\n\nIMPORTANT: Your previous reply was not valid JSON. Return ONLY the JSON object."

        except Exception as e:
            last_error = str(e)
            if _is_rate_limit(e):
                mark_key_exhausted(_SERVICE, api_key, cooldown_seconds=KEY_COOLDOWN_SECONDS)
                model, api_key = _configure_model(genai)  # rotate to next key
            else:
                break  # non-retryable (bad request, network) — stop trying

    return PageExtraction(
        source_image=image_path, entries=[], degraded=True,
        error=last_error or "Unknown extraction failure",
        notes="Vision extraction failed after retries. Try again or re-photograph the page.",
    )
