"""
Key Rotation Manager — round-robin API key cycling with per-key cooldown.

Ported (near-verbatim) from project-overwatch/main_workflow/key_manager.py.
It is service-agnostic, so the same module drives both:
  - GEMINI  (vision, Phase 1)  — free tier ~15 RPM / 1500 RPD per key
  - GROQ    (reasoning, later) — free tier ~30 RPM per key

Env vars (either plural or singular, comma-separated for multiple keys):
    GEMINI_API_KEYS=key1,key2,key3      or   GEMINI_API_KEY=key1
    GROQ_API_KEYS=key1,key2             or   GROQ_API_KEY=key1

On a 429/quota error a caller marks the current key exhausted; it is skipped
until its cooldown expires. If every key is cooling down we return the one that
frees up soonest, so the pipeline degrades instead of crashing mid-demo.
"""
from __future__ import annotations

import os
import threading
import time

from dotenv import load_dotenv

try:  # rich is nice-to-have; never let its absence break key rotation
    from rich.console import Console
    _console = Console()
    def _log(msg: str) -> None:
        _console.print(msg)
except Exception:  # pragma: no cover
    def _log(msg: str) -> None:
        print(msg)

load_dotenv()

# Per-key cooldown tracking: {service: {key_index: cooldown_expiry_timestamp}}
_cooldowns: dict[str, dict[int, float]] = {}
# Round-robin pointer: {service: next_index_to_try}
_pointers: dict[str, int] = {}
_lock = threading.Lock()
_key_cache: dict[str, list[str]] = {}


def _load_keys(service: str) -> list[str]:
    """Load all keys for a service from env (plural preferred, then singular)."""
    keys_str = os.getenv(f"{service}_API_KEYS") or os.getenv(f"{service}_API_KEY")
    if not keys_str:
        return []
    return [k.strip() for k in keys_str.split(",") if k.strip() and not k.strip().startswith("your_")]


def _get_all_keys(service: str) -> list[str]:
    """Cache-friendly key loader (reads env once per service)."""
    if service not in _key_cache:
        _key_cache[service] = _load_keys(service)
    return _key_cache[service]


def has_keys(service: str) -> bool:
    """True if at least one real key is configured for the service."""
    return bool(_get_all_keys(service))


def get_next_key(service: str) -> str:
    """Get the next available API key using round-robin rotation.

    Advances the pointer each call so load is spread evenly, skipping any key in
    cooldown. If all keys are cooling down, returns the soonest-to-recover one.
    """
    keys = _get_all_keys(service)
    if not keys:
        return "dummy"

    now = time.time()
    with _lock:
        _cooldowns.setdefault(service, {})
        _pointers.setdefault(service, 0)

        n = len(keys)
        start = _pointers[service] % n
        for offset in range(n):
            idx = (start + offset) % n
            if now >= _cooldowns[service].get(idx, 0):
                _pointers[service] = idx + 1
                return keys[idx]

        # All keys cooling down — pick the one that expires soonest.
        soonest_idx = min(_cooldowns[service], key=lambda i: _cooldowns[service][i])
        wait = _cooldowns[service][soonest_idx] - now
        _pointers[service] = soonest_idx + 1
        _log(f"[yellow]All {service} keys cooling down. Shortest wait: {wait:.1f}s[/yellow]")
        return keys[soonest_idx]


def mark_key_exhausted(service: str, key: str, cooldown_seconds: float = 62.0) -> None:
    """Mark a key rate-limited so it is skipped for cooldown_seconds.

    If the exact key string is unknown (empty), infer the most-recently-used key
    from the pointer — matches Overwatch's behaviour.
    """
    keys = _get_all_keys(service)
    if not keys:
        return

    with _lock:
        if key in keys:
            idx = keys.index(key)
        else:
            ptr = _pointers.get(service, 0)
            idx = (ptr - 1) % len(keys)
        _cooldowns.setdefault(service, {})[idx] = time.time() + cooldown_seconds
        remaining = sum(
            1 for i in range(len(keys)) if time.time() >= _cooldowns[service].get(i, 0)
        )
    _log(
        f"[dim]Key #{idx + 1}/{len(keys)} for {service} exhausted. "
        f"Cooling {cooldown_seconds:.0f}s. {remaining} keys still available.[/dim]"
    )


def get_key_pool_status(service: str) -> dict:
    """Return status of all keys for monitoring/debugging (keys masked)."""
    keys = _get_all_keys(service)
    now = time.time()
    status = []
    for i, key in enumerate(keys):
        masked = key[:6] + "..." + key[-4:] if len(key) > 12 else "***"
        expiry = _cooldowns.get(service, {}).get(i, 0)
        status.append({
            "index": i,
            "key": masked,
            "available": now >= expiry,
            "cooldown_remaining": max(0.0, expiry - now),
        })
    return {
        "service": service,
        "total_keys": len(keys),
        "available_keys": sum(1 for s in status if s["available"]),
        "pointer": _pointers.get(service, 0) % max(len(keys), 1),
        "keys": status,
    }
