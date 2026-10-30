"""Default usage record: one dict per recorded call, handed to a registered callback.

A caller registers one plain function for the whole process with
``set_usage_callback(fn)``; ``set_usage_callback(None)`` removes it. When no
consumer usage hook is installed (``BaseLLMService.set_usage_hook``),
``_record_usage`` hands each completed call to ``report`` below and
``_record_failure`` hands each failed call to it too (status error or timeout,
the error class, zero tokens and cost). With no callback registered nothing is
built and nothing is written. llm_utils never knows who consumes the record.

The record (one dict per call, a fresh dict each time):

- ``launch_point``: which code made the call, in order: the explicit label
  ``service.launch_point`` (settable directly or via
  ``LLMServiceFactory.create(..., launch_point=...)``); the calling module's
  dotted name; ``DEFAULT_LAUNCH_POINT`` when no caller frame is found (e.g. a
  worker thread).
- ``launch_point_source``: ``"explicit"``, ``"module"`` or ``"default"``, so a
  consumer that maps caller files to its own names knows when to replace it.
- ``caller_file``: the calling module's file, or None.
- ``surface`` / ``host``: ``SURFACE`` and ``HOST`` (every call through this
  package is a direct API call).
- ``model``: the model id; ``repo``: always None (this package cannot know
  it; a consumer fills it from ``caller_file``); ``purpose``: the service's
  ``usage_label``.
- ``input_tokens``, ``output_tokens``, ``cost_usd``, ``is_test``.
- ``status``: ``STATUS_OK``, ``STATUS_ERROR`` or ``STATUS_TIMEOUT``;
  ``error_class``: the exception class name on a failed call, else None.

The callback must be cheap. Reporting never raises into the caller: an
exception from the callback is swallowed, with one warning per process.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Optional

from ._logging import get_logger

logger = get_logger(__name__)

# Surface of every call made through this package, and the host literal for
# direct API callers.
SURFACE = "llm_api"
HOST = "api"
# Call outcome: a completed call is ok; a failed one
# (BaseLLMService._record_failure) is error, or timeout when a deadline expired.
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"
# The launch point when no caller is found.
DEFAULT_LAUNCH_POINT = "llm_utils.library"

# Frames skipped when walking out to the caller: this package (and any
# consumer wrapper whose module path carries an ``llm_utils`` component), plus
# the stdlib machinery that sits between a caller and a coroutine or thread.
_SKIP_PREFIXES = (
    "asyncio", "concurrent", "threading", "contextlib", "functools",
    "runpy", "anyio", "tenacity", "importlib",
)

_callback: Optional[Callable[[dict], None]] = None
_warned = False


def set_usage_callback(callback: Optional[Callable[[dict], None]]) -> None:
    """Register the process-wide usage callback (one dict per recorded call);
    None removes it. A later registration replaces the earlier one."""
    global _callback, _warned
    if callback is not None and not callable(callback):
        raise TypeError(f"usage callback must be callable, got {type(callback).__name__}")
    _callback = callback
    _warned = False


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


def report(service: Any, input_tokens: int, output_tokens: int,
           cost: float, *, is_test: bool = False,
           label: Optional[str] = None, status: str = STATUS_OK,
           error_class: Optional[str] = None) -> Optional[dict]:
    """Hand one call's record to the registered callback; return the record,
    or None when no callback is registered or reporting failed. Never raises.

    A completed call is ``status="ok"``; a failed one passes ``status``
    ("error" | "timeout") and ``error_class`` with zero tokens and cost."""
    global _warned
    callback = _callback
    if callback is None:
        return None
    try:
        model = getattr(service, "model", None)
        model_id = getattr(model, "model_id", None) or (str(model) if model is not None else None)
        purpose = label if label is not None else getattr(service, "usage_label", None)
        explicit = getattr(service, "launch_point", None)
        module, path = caller()
        if explicit:
            launch_point, source = str(explicit), "explicit"
        elif module is not None:
            launch_point, source = module, "module"
        else:
            launch_point, source = DEFAULT_LAUNCH_POINT, "default"
        record = dict(
            launch_point=launch_point, launch_point_source=source,
            caller_file=path, surface=SURFACE, host=HOST,
            model=model_id, repo=None, purpose=purpose,
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            cost_usd=float(cost) if cost is not None else None,
            is_test=bool(is_test), status=status, error_class=error_class,
        )
        callback(record)
        return record
    except Exception as e:  # noqa: BLE001 — reporting never breaks the caller
        if not _warned:
            _warned = True
            logger.warning(f"usage callback failed ({str(e)[:90]}) — "
                           f"calls proceed unreported this process")
        return None
