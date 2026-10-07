"""Default durable usage record: one call-ledger row per recorded call.

When no consumer usage hook is installed (``BaseLLMService.set_usage_hook``)
and the optional ``agent_manager`` package is importable, ``_record_usage``
hands each completed call to ``record_call`` below, which appends one row to
agent_manager's call ledger; ``_record_failure`` hands each failed call to it
too (status error or timeout, the error class, zero tokens and cost). Without
``agent_manager`` this module is inert: llm_utils never depends on it.

A consumer that installs its own hook and still wants the ledger row chains to
``record_call(service, ...)`` from inside that hook.

Launch point (the ledger's name for "which code made this call"), in order:
1. the explicit label: ``service.launch_point`` (settable directly or via
   ``LLMServiceFactory.create(..., launch_point=...)``);
2. the calling module's file, resolved to a census id through agent_manager's
   census (exact file match first, then a unique directory row);
3. the calling module's dotted name;
4. ``DEFAULT_LAUNCH_POINT`` when no caller frame is found (e.g. a worker thread).

The census lookup (``census.resolve``) also gives the row's ``repo``. When
agent_manager refuses it because its control data is unavailable (the
published oikos map, ``OikosUnavailable``), the row is still written, with
``repo`` null, the module-name launch point, and ``ATTRIBUTION_UNAVAILABLE``
noted in ``error_class``, so it never reads as a lookup that found nothing.
Only successful lookups are reused, each for ``call_ledger.attribution_ttl_s``
seconds (package config).

Recording never raises into the caller.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Optional

from . import config

# Ledger vocabulary (agent_manager config): surface of every call made
# through this package, and the host literal for direct API callers.
SURFACE = "llm_api"
HOST = "api"
# Call outcome (the ledger's `status` vocabulary): a completed call is ok; a
# failed one (BaseLLMService._record_failure) is error, or timeout when a
# deadline expired.
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"
# The census row of this package itself: the fallback when no caller is found.
DEFAULT_LAUNCH_POINT = "llm_utils.library"
# Note written into a row's error_class when agent_manager refused the census
# lookup (``attribution_unavailable:<error class>``), after the call's own
# error class on a failed call. The ledger has no attribution-status field and
# rejects unknown fields; agent_manager's own writers note an ok row's
# degraded record the same way (``transcript_unreadable:<error>``), and its
# audit counts error classes only on failed rows.
ATTRIBUTION_UNAVAILABLE = "attribution_unavailable"
# Distinct caller files whose resolution is kept (the former lru_cache size).
_CACHE_SIZE = 512

# Frames skipped when walking out to the caller: this package (and any
# consumer wrapper whose module path carries an ``llm_utils`` component), plus
# the stdlib machinery that sits between a caller and a coroutine or thread.
_SKIP_PREFIXES = (
    "asyncio", "concurrent", "threading", "contextlib", "functools",
    "runpy", "anyio", "tenacity", "importlib",
)


def _ledger():
    """agent_manager.ledger, or None when the package is not installed."""
    try:
        from agent_manager import ledger  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — optional dependency; any import failure = absent
        return None
    return ledger


def available() -> bool:
    return _ledger() is not None


def _skipped(module: str) -> bool:
    if not module:
        return True
    parts = module.split(".")
    if "llm_utils" in parts:
        return True
    return any(module == p or module.startswith(p + ".") for p in _SKIP_PREFIXES)


def _module_name(frame) -> str:
    g = frame.f_globals
    name = g.get("__name__") or ""
    if name == "__main__":
        spec = g.get("__spec__")
        if spec is not None and getattr(spec, "name", None):
            return spec.name
        path = g.get("__file__") or frame.f_code.co_filename
        if not path or path.startswith("<"):  # `python -c`, REPL
            return name
        return os.path.splitext(os.path.basename(path))[0]
    return name


def caller() -> tuple[Optional[str], Optional[str]]:
    """(module name, file path) of the first frame outside llm_utils and the
    async/thread machinery, or (None, None)."""
    frame = sys._getframe(1)
    while frame is not None:
        module = _module_name(frame)
        if not _skipped(module):
            return module, frame.f_code.co_filename
        frame = frame.f_back
    return None, None


_cache: dict = {}  # path -> (monotonic expiry, census.resolve result)
_cache_lock = threading.Lock()


def clear_cache() -> None:
    """Forget every reused census resolution."""
    with _cache_lock:
        _cache.clear()


def census_id(path: Optional[str]) -> Optional[str]:
    """The census row id covering ``path``: the row naming that exact file,
    else the single directory row containing it; None when unresolved."""
    return _lookup(path)[0].get("launch_point")


def repo_of(path: Optional[str]) -> Optional[str]:
    """The oikos repo containing ``path`` per agent_manager, else None."""
    return _lookup(path)[0].get("repo")


# Kept from the lru_cache era: callers and tests reset through either name.
census_id.cache_clear = clear_cache  # type: ignore[attr-defined]
repo_of.cache_clear = clear_cache  # type: ignore[attr-defined]


def _refusals() -> tuple:
    """The errors with which ``census.resolve`` refuses unavailable control
    data: ``AgentManagerError`` (its documented seam contract; the published
    map raises ``OikosUnavailable`` since agent_manager 1.27.0) and datastore's
    ``LiveHomeUnderTest``, which resolve re-raises beside it."""
    found = []
    try:
        from agent_manager.config import AgentManagerError  # type: ignore[import-not-found]
        found.append(AgentManagerError)
    except Exception:  # noqa: BLE001 — optional dependency
        pass
    try:
        from datastore.home import LiveHomeUnderTest  # type: ignore[import-not-found]
        found.append(LiveHomeUnderTest)
    except Exception:  # noqa: BLE001 — optional dependency
        pass
    return tuple(found)


def _lookup(path: Optional[str]) -> tuple[dict, Optional[str]]:
    """(agent_manager's ``census.resolve`` result, refusal error class or None).

    A refused lookup gives ({}, the error's class name); no census seam or any
    other failure gives ({}, None), best effort as before. Only a successful
    result is reused, until ``call_ledger.attribution_ttl_s`` passes."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(path)
        if hit is not None and hit[0] > now:
            return hit[1], None
    try:
        from agent_manager import census  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 — no census seam: nothing to attribute
        return {}, None
    try:
        found = census.resolve(path) or {}
    except _refusals() as exc:
        return {}, type(exc).__name__
    except Exception:  # noqa: BLE001 — other resolution failures stay best effort
        return {}, None
    try:
        ttl = float(config.get("call_ledger.attribution_ttl_s"))
    except Exception:  # noqa: BLE001 — unreadable config: use the result, keep nothing
        return found, None
    if ttl > 0:
        with _cache_lock:
            _cache.pop(path, None)
            while len(_cache) >= _CACHE_SIZE:
                _cache.pop(next(iter(_cache)))  # oldest stored first
            _cache[path] = (now + ttl, found)
    return found, None


def launch_point_for(service: Any) -> str:
    return _resolve(service)[0]


def _resolve(service: Any) -> tuple[str, Optional[str], Optional[str]]:
    """(launch point, repo or None, census refusal error class or None) for a
    call made by ``service``."""
    explicit = getattr(service, "launch_point", None)
    module, path = caller()
    found, refused = _lookup(path)
    repo = found.get("repo")
    if explicit:
        return str(explicit), repo, refused
    if module is None:
        return DEFAULT_LAUNCH_POINT, None, refused
    return found.get("launch_point") or module, repo, refused


def record_call(service: Any, input_tokens: int, output_tokens: int,
                cost: float, *, is_test: bool = False,
                label: Optional[str] = None, status: str = STATUS_OK,
                error_class: Optional[str] = None) -> Optional[dict]:
    """Append one call-ledger row for a call; return the row, or None when
    agent_manager is absent or recording failed. Never raises.

    A completed call is ``status="ok"``; a failed one passes ``status``
    ("error" | "timeout") and ``error_class`` with zero tokens and cost. A
    refused census lookup adds ``attribution_unavailable:<error class>`` to
    ``error_class`` (after the call's own class, joined by "; ")."""
    try:
        ledger = _ledger()
        if ledger is None:
            return None
        model = getattr(service, "model", None)
        model_id = getattr(model, "model_id", None) or (str(model) if model is not None else None)
        purpose = label if label is not None else getattr(service, "usage_label", None)
        launch_point, repo, refused = _resolve(service)
        if refused is not None:  # never read as a lookup that found nothing
            note = f"{ATTRIBUTION_UNAVAILABLE}:{refused}"
            error_class = note if error_class is None else f"{error_class}; {note}"
        fields = dict(
            launch_point=launch_point, surface=SURFACE, host=HOST,
            model=model_id, repo=repo, purpose=purpose,
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            cost_usd=float(cost) if cost is not None else None, status=status,
        )
        if error_class is not None:  # absent = null in the ledger row
            fields["error_class"] = error_class
        return ledger.record(**fields)
    except Exception:  # noqa: BLE001 — recording never breaks the caller
        return None
