"""Per-process spend cap on paid API calls (``max_usd_per_run``).

A "run" is the process. Every paid call is admitted BEFORE any request leaves
the process: the run's recorded spend, plus the estimates of calls in flight
and of native batches submitted but not yet harvested, plus this call's own
estimate, must stay within the cap. Otherwise the call raises
``SpendCapExceededError`` and nothing is sent. A call already admitted always
finishes; the cap never stops a request halfway.

Scope: API spend only. Self-served routes (``jurisdiction == "self"``: local
models, SLURM-served endpoints) are never checked or counted.

The cap, highest precedence first:

1. ``LLM_UTILS_MAX_USD_PER_RUN`` (a number of US dollars, or ``none`` for no
   cap): the switch a launcher sets for an approved sweep;
2. ``spend_cap.max_usd_per_run`` in the project's ``llm_utils.yaml`` (or a
   ``configure(spend_cap={...})`` call);
3. the packaged default in ``defaults.yaml``.

The per-call estimate is the service's ``_estimate_cost_usd``: character-count
input tokens and ``max_tokens`` output per request unless the caller passes
``expected_output_tokens``. It is an upper bound for most calls, so the cap
errs toward refusing; reasoning-model callers pass a realistic
``expected_output_tokens`` to avoid refusals caused by a large ``max_tokens``.
"""
from __future__ import annotations

import contextlib
import math
import os
import threading
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from . import config
from .exceptions import SpendCapExceededError

MAX_USD_ENV = "LLM_UTILS_MAX_USD_PER_RUN"

_lock = threading.Lock()
_spent = 0.0
_reserved = 0.0
_committed: Dict[str, float] = {}
_local = threading.local()


@dataclass(frozen=True)
class SpendStatus:
    """The run's spend position in US dollars. ``cap`` None = no cap."""
    spent: float
    reserved: float
    committed: float
    cap: Optional[float]

    @property
    def remaining(self) -> Optional[float]:
        if self.cap is None:
            return None
        return max(0.0, self.cap - self.spent - self.reserved - self.committed)


def _parse_cap(raw, where: str) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("none", "off", "unlimited"):
            return None
        try:
            raw = float(text)
        except ValueError:
            raise ValueError(f"{where}: spend cap must be a dollar amount or 'none', got {raw!r}") from None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0:
        raise ValueError(f"{where}: spend cap must be a finite amount >= 0, got {raw!r}")
    return float(raw)


def max_usd_per_run() -> Optional[float]:
    """The cap in force for this process (None = no cap)."""
    raw = os.getenv(MAX_USD_ENV)
    if raw is not None and raw.strip():
        return _parse_cap(raw, MAX_USD_ENV)
    return _parse_cap(config.get("spend_cap.max_usd_per_run"), "spend_cap.max_usd_per_run")


def spend_status() -> SpendStatus:
    """Recorded spend, in-flight reservations and unharvested batch estimates."""
    cap = max_usd_per_run()
    with _lock:
        return SpendStatus(_spent, _reserved, sum(_committed.values()), cap)


def reset_run_spend() -> None:
    """Zero the run's counters (tests; a long-lived process starting a new run)."""
    global _spent, _reserved
    with _lock:
        _spent = 0.0
        _reserved = 0.0
        _committed.clear()


def record(cost: float) -> None:
    """Add a completed call's actual cost (called by ``_record_usage``)."""
    global _spent
    if cost:
        with _lock:
            _spent += cost


def admit(estimate: float, what: str) -> float:
    """Reserve ``estimate`` for a call about to be sent, or raise
    ``SpendCapExceededError`` without reserving anything."""
    global _reserved
    cap = max_usd_per_run()
    estimate = max(0.0, float(estimate))
    with _lock:
        if cap is not None:
            committed = sum(_committed.values())
            if _spent + _reserved + committed + estimate > cap:
                raise SpendCapExceededError(
                    f"{what}: estimated ${estimate:.4f} on top of this run's "
                    f"${_spent:.4f} spent, ${_reserved:.4f} in flight and "
                    f"${committed:.4f} in unharvested batches would pass the spend cap "
                    f"${cap:.2f} (spend_cap.max_usd_per_run). Nothing was sent. For an "
                    f"approved larger run set {MAX_USD_ENV}=<dollars> in its launcher, "
                    f"or spend_cap.max_usd_per_run in the project's llm_utils.yaml; "
                    f"reasoning-model callers can pass expected_output_tokens to "
                    f"tighten the estimate.",
                    estimate=estimate, spent=_spent, cap=cap)
        _reserved += estimate
    return estimate


def release(estimate: float) -> None:
    """Drop a reservation once its call has finished (its real cost was recorded)."""
    global _reserved
    with _lock:
        _reserved = max(0.0, _reserved - estimate)


def commit_batch(batch_id: str, estimate: float) -> None:
    """Hold a submitted native batch's estimate until it is harvested."""
    with _lock:
        _committed[str(batch_id)] = max(0.0, float(estimate))


def settle_batch(batch_id: str) -> None:
    """A harvested batch's real cost is recorded; drop its held estimate."""
    with _lock:
        _committed.pop(str(batch_id), None)


def in_guard() -> bool:
    return getattr(_local, "depth", 0) > 0


@contextlib.contextmanager
def guarded() -> Iterator[None]:
    """Mark this thread as inside an admitted call, so nested seam calls
    (``chat`` -> ``batch_chat``, an override calling ``super()``) are not
    admitted twice."""
    _local.depth = getattr(_local, "depth", 0) + 1
    try:
        yield
    finally:
        _local.depth -= 1
