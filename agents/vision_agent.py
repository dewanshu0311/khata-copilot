"""
Vision Agent — reads a handwritten khata page into structured JSON.

Pipeline role (Phase 1): photo in -> validated PageExtraction out.

Design principles (from the spec):
  - Never fake functionality: confidence scores are reported honestly; the
    self-flagging of low-confidence entries IS the feature.
  - Never crash mid-demo: JSON-parse retries, key rotation on 429/quota (via
    core.key_manager), a Groq vision fallback, and a mock last resort so the
    CLI/UI run fully offline.

Reader order (Phase 7): Gemini (primary, best handwriting reader) -> Groq
vision (fallback, only tried when Gemini is unavailable or exhausted) -> mock
(last resort, only when BOTH real readers are unavailable). Gemini free-tier
quota errors are exactly what motivated adding the fallback, so a quota-out
Gemini key no longer forces the whole page into mock mode.

google-generativeai and groq are imported lazily so mock mode works even if
neither SDK is installed yet.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Optional

from pydantic import ValidationError

from core import prompts
from core.config import (
    GEMINI_VISION_MODEL,
    GROQ_VISION_MODEL,
    KEY_COOLDOWN_SECONDS,
    MAX_VISION_RETRIES,
    mock_mode_enabled,
)
from core.key_manager import get_next_key, has_keys, mark_key_exhausted
from core.schemas import LedgerEntry, PageExtraction

_SERVICE = "GEMINI"
_GROQ_SERVICE = "GROQ"
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
            # Retyped (Phase 8) from a udhaar/jama line to a completed CASH SALE, so
            # the mock exercises the new sale axis (sale + paid = completed bill) in
            # the Scan preview. Still 4 entries — the len==4 mock test stays alive.
            {"name": "Sita Devi", "amount": 450, "date": "6 Jan",
             "status": "paid", "entry_type": "sale", "confidence": 0.88,
             "raw_text": "Sita Devi 450 cash sale (bill) 6 Jan"},
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


def _build_prompt(correction_feedback: Optional[str]) -> str:
    prompt = prompts.VISION_EXTRACTION_PROMPT
    if correction_feedback:
        # A targeted re-read requested by the Verification Agent's self-correction
        # loop. Appended so the base honesty contract still applies.
        prompt += "\n\n" + prompts.VERIFICATION_CORRECTION_PROMPT.format(
            feedback=correction_feedback
        )
    return prompt


def _extract_page_gemini(image_path: str, correction_feedback: Optional[str]) -> tuple[Optional[PageExtraction], Optional[str]]:
    """Try Gemini. Returns (extraction, None) on success, or (None, reason) so
    the caller knows to try the Groq vision fallback next."""
    if not has_keys(_SERVICE):
        return None, "no GEMINI_API_KEY configured"
    try:
        import google.generativeai as genai
    except ImportError:
        return None, "google-generativeai not installed"

    try:
        image = _load_image(image_path)
    except Exception as e:
        return None, f"could not open image: {e}"

    generation_config = {"temperature": 0.1, "response_mime_type": "application/json"}
    prompt = _build_prompt(correction_feedback)
    last_error: Optional[str] = None

    model, api_key = _configure_model(genai)

    for _attempt in range(MAX_VISION_RETRIES + 1):
        try:
            response = model.generate_content([prompt, image], generation_config=generation_config)
            text = _strip_code_fences(response.text or "")
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError("Model did not return a JSON object")
            return _extraction_from_payload(payload, image_path, degraded=False), None

        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON from model: {e}"
            # Nudge the model to emit clean JSON on the retry.
            prompt = _build_prompt(correction_feedback) + "\n\nIMPORTANT: Your previous reply was not valid JSON. Return ONLY the JSON object."

        except Exception as e:
            last_error = str(e)
            if _is_rate_limit(e):
                mark_key_exhausted(_SERVICE, api_key, cooldown_seconds=KEY_COOLDOWN_SECONDS)
                model, api_key = _configure_model(genai)  # rotate to next key
            else:
                break  # non-retryable (bad request, network) — stop trying

    return None, last_error or "unknown Gemini failure"


# ── Groq vision fallback (Phase 7) ───────────────────────────────────────────
# Only reached when Gemini is unavailable or exhausted. Groq's vision model is
# NOT as strong as Gemini on messy handwriting — this is a "keep the demo
# alive" fallback, not a replacement, and the returned notes say so honestly.
def _extract_page_groq_vision(image_path: str, correction_feedback: Optional[str]) -> tuple[Optional[PageExtraction], Optional[str]]:
    """Try Groq vision. Returns (extraction, None) on success, or (None, reason)."""
    if not has_keys(_GROQ_SERVICE):
        return None, "no GROQ_API_KEY configured"
    try:
        from groq import Groq
    except ImportError:
        return None, "groq package not installed"

    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        return None, f"could not open image: {e}"

    ext = image_path.rsplit(".", 1)[-1].lower() if "." in image_path else "jpeg"
    mime = "image/png" if ext == "png" else "image/jpeg"
    prompt = _build_prompt(correction_feedback)
    messages = [
        {"role": "system", "content": prompts.VISION_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]},
    ]

    last_error: Optional[str] = None
    for _attempt in range(MAX_VISION_RETRIES + 1):
        api_key = get_next_key(_GROQ_SERVICE)
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=messages,
                temperature=0.1,
            )
            text = _strip_code_fences(response.choices[0].message.content or "")
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError("Model did not return a JSON object")
            extraction = _extraction_from_payload(payload, image_path, degraded=False)
            extraction.notes = (
                (extraction.notes + " | " if extraction.notes else "")
                + "Read via Groq vision fallback (Gemini was unavailable) — this reader is "
                "less reliable than Gemini on messy handwriting, review flagged entries carefully."
            )
            return extraction, None

        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON from Groq vision: {e}"

        except Exception as e:
            last_error = str(e)
            if _is_rate_limit(e):
                mark_key_exhausted(_GROQ_SERVICE, api_key, cooldown_seconds=KEY_COOLDOWN_SECONDS)
                continue  # next iteration picks the next available key
            break  # non-retryable — stop trying

    return None, last_error or "unknown Groq vision failure"


def extract_page(image_path: str, correction_feedback: Optional[str] = None) -> PageExtraction:
    """Read a handwritten khata page into a validated PageExtraction.

    Always returns a PageExtraction — on unrecoverable failure it is flagged
    degraded with an error message, never an exception, so the demo survives.

    Reader order: Gemini (primary) -> Groq vision (fallback) -> mock (last
    resort, only if both real readers are unavailable/fail).

    correction_feedback: optional text from the Verification Agent (Phase 2)
        describing a problem it found (e.g. a math mismatch). When present it is
        appended to the extraction prompt as a targeted re-read hint. Defaults to
        None so all existing callers are unaffected. In mock mode it changes
        nothing (the canned page is returned regardless), which is exactly what
        lets the self-correction loop terminate instead of looping forever.
    """
    if mock_mode_enabled():
        return _mock_extraction(image_path, reason="KHATA_MOCK=1")

    try:
        _load_image(image_path).close()
    except Exception as e:
        # A bad file path/corrupt image is not an API problem — no fallback
        # reader can fix it, so say so plainly instead of masking it as mock data.
        return PageExtraction(
            source_image=image_path, entries=[], degraded=True,
            error=f"Could not open image: {e}",
            notes="Check the file path and that it is a valid image.",
        )

    gemini_result, gemini_reason = _extract_page_gemini(image_path, correction_feedback)
    if gemini_result is not None:
        return gemini_result

    groq_result, groq_reason = _extract_page_groq_vision(image_path, correction_feedback)
    if groq_result is not None:
        return groq_result

    return _mock_extraction(
        image_path,
        reason=f"Gemini failed ({gemini_reason}); Groq vision fallback also failed ({groq_reason})",
    )
