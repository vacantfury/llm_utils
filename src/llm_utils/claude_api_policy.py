"""Anthropic API calls are opt-in per process.

A model served by the Anthropic API itself (provider ANTHROPIC, ``ClaudeService``)
is refused unless the process sets ``LLM_UTILS_ALLOW_CLAUDE_API=1``. The
deployment decides which processes may spend on the Anthropic API (for example,
research runs only) by exporting the variable in those launchers; every other
process fails closed when the service is built, before any request is sent.
Claude served by other routes (Bedrock, OpenAI-compatible routers) is not
covered: those routes bill their own accounts.
"""
from __future__ import annotations

import os

from .llm_model import Provider

ALLOW_ENV = "LLM_UTILS_ALLOW_CLAUDE_API"


class ClaudeAPINotAllowed(PermissionError):
    """A paid Claude API service was requested in a process that has not opted in."""


def is_claude_model(model) -> bool:
    """A Claude-family model by registry metadata: provider Anthropic, family claude,
    or Claude weights on another route."""
    family = (getattr(model, "family", None) or "").lower()
    weights = (getattr(model, "weights", None) or "").lower()
    model_id = str(getattr(model, "model_id", "") or "").lower()
    return (getattr(model, "provider", None) == Provider.ANTHROPIC or family == "claude"
            or "claude" in weights or "claude" in model_id)


def claude_api_allowed() -> bool:
    """True only when this process opted in with ``LLM_UTILS_ALLOW_CLAUDE_API=1``."""
    return os.getenv(ALLOW_ENV, "").strip() == "1"


def require_claude_api_allowed(model) -> None:
    """Raise ClaudeAPINotAllowed for an Anthropic API model unless this process opted in."""
    if getattr(model, "provider", None) == Provider.ANTHROPIC and not claude_api_allowed():
        raise ClaudeAPINotAllowed(
            f"{getattr(model, 'model_id', model)}: Anthropic API calls are off in this "
            f"process. Set {ALLOW_ENV}=1 only in the launchers allowed to spend on the "
            "Claude API; elsewhere run Claude through Claude Code (`claude -p`)."
        )
