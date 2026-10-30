"""Mainland-China endpoints are opt-in per process.

A request to a route whose endpoint processes data under PRC jurisdiction
(``Provider.jurisdiction == "prc"``: the direct DeepSeek, Z.AI and Moonshot
endpoints) is refused unless the process sets ``LLM_UTILS_ALLOW_PRC_ENDPOINTS=1``.
Setting it declares that the process sends only public or synthetic data
(research evals, judge sweeps over public content, dev tests). Personal data
about the owner or people he knows never goes to these endpoints; such a task
uses the same open weights on a US route (``LLMModel.us_route``, e.g. OpenRouter).

The check runs before any paid entry point sends a request. Building a service
and its balance lookups carry no content and stay allowed, so credit monitors
work without the opt-in.
"""
from __future__ import annotations

import os

ALLOW_ENV = "LLM_UTILS_ALLOW_PRC_ENDPOINTS"


class PRCEndpointNotAllowed(PermissionError):
    """A mainland-China endpoint was asked to process a request in a process that has not opted in."""


def is_prc_route(model) -> bool:
    """True when the model's endpoint processes data under PRC jurisdiction."""
    return getattr(model, "jurisdiction", None) == "prc"


def prc_endpoints_allowed() -> bool:
    """True only when this process opted in with ``LLM_UTILS_ALLOW_PRC_ENDPOINTS=1``."""
    return os.getenv(ALLOW_ENV, "").strip() == "1"


def require_prc_allowed(model) -> None:
    """Raise PRCEndpointNotAllowed for a PRC-jurisdiction route unless this process opted in."""
    if is_prc_route(model) and not prc_endpoints_allowed():
        raise PRCEndpointNotAllowed(
            f"{getattr(model, 'model_id', model)}: mainland-China endpoints are off in this "
            f"process. Set {ALLOW_ENV}=1 only in launchers whose calls carry public or "
            "synthetic data; for anything about a person, use the model's US route "
            "(LLMModel.us_route, e.g. OpenRouter)."
        )
