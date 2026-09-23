"""Default durable usage record: one call-ledger row per recorded call.

When no consumer usage hook is installed (``BaseLLMService.set_usage_hook``)
and the optional ``agent_manager`` package is importable, ``_record_usage``
hands each call to ``record_call`` below, which appends one row to
agent_manager's call ledger. Without ``agent_manager`` this module is inert:
llm_utils never depends on it.

A consumer that installs its own hook and still wants the ledger row chains to
``record_call(service, ...)`` from inside that hook.

Launch point (the ledger's name for "which code made this call"), in order:
1. the explicit label: ``service.launch_point`` (settable directly or via
   ``LLMServiceFactory.create(..., launch_point=...)``);
2. the calling module's file, resolved to a census id through agent_manager's
   census (exact file match first, then a unique directory row);
3. the calling module's dotted name;
4. ``DEFAULT_LAUNCH_POINT`` when no caller frame is found (e.g. a worker thread).

Recording never raises into the caller.
"""
from __future__ import annotations

import functools
import os
import sys
from typing import Any, Optional

# Ledger vocabulary (agent_manager config): surface of every call made
# through this package, and the host literal for direct API callers.
SURFACE = "llm_api"
HOST = "api"
# The census row of this package itself: the fallback when no caller is found.
DEFAULT_LAUNCH_POINT = "llm_utils.library"

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


@functools.lru_cache(maxsize=512)
def census_id(path: Optional[str]) -> Optional[str]:
    """The census row id covering ``path``: the row naming that exact file,
    else the single directory row containing it; None when unresolved."""
    return _census_resolve(path).get("launch_point")


def _census_resolve(path: Optional[str]) -> dict:
    """agent_manager's public ``census.resolve`` seam; empty when unavailable."""
    try:
        from agent_manager import census  # type: ignore[import-not-found]

        return census.resolve(path) or {}
    except Exception:  # noqa: BLE001 — resolution is best effort
        return {}


@functools.lru_cache(maxsize=512)
def repo_of(path: Optional[str]) -> Optional[str]:
    """The oikos repo containing ``path`` per agent_manager, else None."""
    return _census_resolve(path).get("repo")


def launch_point_for(service: Any) -> str:
    return _resolve(service)[0]


def _resolve(service: Any) -> tuple[str, Optional[str]]:
    """(launch point, repo or None) for a call made by ``service``."""
    explicit = getattr(service, "launch_point", None)
    module, path = caller()
    repo = repo_of(path)
    if explicit:
        return str(explicit), repo
    if module is None:
        return DEFAULT_LAUNCH_POINT, None
    return census_id(path) or module, repo


def record_call(service: Any, input_tokens: int, output_tokens: int,
                cost: float, *, is_test: bool = False,
                label: Optional[str] = None) -> Optional[dict]:
    """Append one call-ledger row for a completed call; return the row, or
    None when agent_manager is absent or recording failed. Never raises."""
    try:
        ledger = _ledger()
        if ledger is None:
            return None
        model = getattr(service, "model", None)
        model_id = getattr(model, "model_id", None) or (str(model) if model is not None else None)
        purpose = label if label is not None else getattr(service, "usage_label", None)
        launch_point, repo = _resolve(service)
        return ledger.record(
            launch_point=launch_point, surface=SURFACE, host=HOST,
            model=model_id, repo=repo, purpose=purpose,
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            cost_usd=float(cost) if cost is not None else None, status="ok",
        )
    except Exception:  # noqa: BLE001 — recording never breaks the caller
        return None
