"""
Groq chat helper — one small, resilient entry point for every LLM reasoning call.

Shared by Phase 4 (Insights Agent) and Phase 5 (Reminder Agent) so the
key-rotation, mock-mode and retry logic lives in exactly one place.

Contract (the important part): groq_chat() NEVER raises and NEVER blocks a demo.
It returns the answer text, or None when the LLM is unavailable — mock mode, no
GROQ key, the `groq` package missing, or the call failing after its bounded
retries. Callers treat None as "use your non-LLM fallback", so the pipeline
degrades instead of crashing.

Key rotation reuses core.key_manager (the same round-robin + cooldown pool that
drives the Vision Agent), so multiple free-tier Groq keys are load-balanced and
a 429'd key is skipped until it cools down.
"""
from __future__ import annotations

from typing import List, Optional

from core.config import (
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    KEY_COOLDOWN_SECONDS,
    MAX_GROQ_RETRIES,
    mock_mode_enabled,
)
from core.key_manager import get_next_key, has_keys, mark_key_exhausted

try:  # rich is nice-to-have; never let its absence break a reasoning call
    from rich.console import Console
    _console = Console()
    def _log(msg: str) -> None:
        _console.print(msg)
except Exception:  # pragma: no cover
    def _log(msg: str) -> None:
        print(msg)

_SERVICE = "GROQ"
# Same rate-limit fingerprints the Vision Agent watches for, so a 429 from Groq
# triggers key rotation instead of a hard failure.
_RATE_LIMIT_MARKERS = (
    "429", "rate limit", "ratelimit", "rate_limit", "quota",
    "resource_exhausted", "resource exhausted", "try again in",
)


def _is_rate_limit(error: Exception) -> bool:
    msg = str(error).lower()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def groq_available() -> bool:
    """True only if a real Groq call could actually be made right now.

    Lets callers (e.g. the guardrail) skip the LLM path cheaply instead of
    building a request that is guaranteed to return None.
    """
    if mock_mode_enabled() or not has_keys(_SERVICE):
        return False
    try:
        import groq  # noqa: F401
    except ImportError:
        return False
    return True


def groq_chat(
    system: str,
    user: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> Optional[str]:
    """Run one system+user chat completion on Groq's llama-3.3-70b-versatile.

    Returns the reply text, or None when the LLM is unavailable or the call
    fails after `max_retries` key-rotating attempts. Never raises.
    """
    if mock_mode_enabled() or not has_keys(_SERVICE):
        return None
    try:
        from groq import Groq
    except ImportError:
        return None

    temperature = GROQ_TEMPERATURE if temperature is None else temperature
    max_tokens = GROQ_MAX_TOKENS if max_tokens is None else max_tokens
    max_retries = MAX_GROQ_RETRIES if max_retries is None else max_retries

    messages: List[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_error: Optional[Exception] = None
    for _attempt in range(max_retries + 1):
        api_key = get_next_key(_SERVICE)  # advances the round-robin pointer each call
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
            return text or None
        except Exception as e:  # noqa: BLE001 — deliberately total: we degrade, never crash
            last_error = e
            if _is_rate_limit(e):
                mark_key_exhausted(_SERVICE, api_key, cooldown_seconds=KEY_COOLDOWN_SECONDS)
                continue  # next iteration picks the next available key
            break  # non-retryable (bad request, network) — stop and fall back

    _log(f"[dim]Groq call failed ({last_error}); caller will fall back.[/dim]")
    return None
