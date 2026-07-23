"""
Base abstract class for LLM services with usage tracking.

All services implement a single public method: ``batch_chat``.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Awaitable, Callable, List, Optional, Tuple, TypeVar

from ._logging import get_logger

logger = get_logger(__name__)

_T = TypeVar("_T")


# Substrings we treat as rate-limit / quota errors (case-insensitive).
# Covers OpenAI 429s, Google RESOURCE_EXHAUSTED, Anthropic 429s, and
# vLLM rate-limit responses. Extend here if a new provider surfaces a
# different message format.
_RATE_LIMIT_PATTERNS = (
    "429",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "rate-limit",
    "too many requests",
    "resource_exhausted",
    "resource exhausted",
    "quota exceeded",
    "quota_exceeded",
    "overloaded",
    "throttle",
)


def is_rate_limit_error(exc: BaseException) -> bool:
    """Heuristic: does the exception look like a rate-limit / quota error?"""
    err = str(exc).lower()
    return any(p in err for p in _RATE_LIMIT_PATTERNS)


# ---------------------------------------------------------------------------
# Account-fatal error detection (bad key / no credits)
# ---------------------------------------------------------------------------
# These are ACCOUNT-GLOBAL failures: the API key is invalid, or the account is
# out of credits. They will NOT recover mid-run, so a service must fail-fast
# (raise an AccountFatalError that aborts the whole run) instead of retrying
# every cell and grinding each into a mechanism-error. See exceptions.py.

# Substrings that mean the API KEY itself is bad (invalid / revoked / 401).
# Case-insensitive. Kept phrase-specific: a false positive here ABORTS the run.
_INVALID_CREDENTIAL_PATTERNS = (
    "invalid_api_key",
    "invalid api key",
    "incorrect api key",          # OpenAI: "Incorrect API key provided"
    "invalid x-api-key",          # Anthropic
    "api key not valid",          # Google: "API key not valid. Please pass a valid API key."
    "api_key_invalid",            # Google error status
    "authentication_error",
    "authentication error",
    "error code: 401",
    "http 401",
    "status code 401",
)

# Substrings that mean the account is OUT OF CREDITS / over its billing quota.
# NOTE: several providers surface this as an HTTP 429 (same status as a transient
# rate-limit), so `is_credit_exhausted_error` MUST be consulted BEFORE the
# rate-limit retry branch, or an exhausted account gets pointlessly retried and
# then mechanism-errored. Case-insensitive; phrase-specific to avoid false hits.
_CREDIT_EXHAUSTED_PATTERNS = (
    "insufficient_quota",             # OpenAI billing (code; comes back as a 429)
    "exceeded your current quota",    # OpenAI billing (message)
    "credit balance is too low",      # Anthropic
    "insufficient balance",           # DeepSeek ("Insufficient Balance")
    "payment required",               # generic HTTP 402
    "error code: 402",
    "billing_not_active",             # some OpenAI-compatible endpoints
    "billing_hard_limit_reached",     # OpenAI hard billing cap
    "account is not active",          # xAI / some compatible endpoints
    "arrearage",                      # Z.AI / some CN endpoints (owed balance)
)


def is_invalid_credential_error(exc: BaseException) -> bool:
    """Heuristic: did the provider reject the API key (invalid / revoked / 401)?"""
    err = str(exc).lower()
    return any(p in err for p in _INVALID_CREDENTIAL_PATTERNS)


def is_credit_exhausted_error(exc: BaseException) -> bool:
    """Heuristic: is the account out of credits / over its billing quota?

    Consulted BEFORE `is_rate_limit_error` because several providers report this
    as a 429 that would otherwise be mistaken for a transient rate-limit.
    """
    err = str(exc).lower()
    return any(p in err for p in _CREDIT_EXHAUSTED_PATTERNS)


# ---------------------------------------------------------------------------
# Mechanism-error sentinel
# ---------------------------------------------------------------------------
# A response wrapped with this sentinel marks a genuine MECHANISM / processing
# failure: the API call did NOT produce a valid model output (context-overflow,
# network/connection error, timeout, rate-limit exhaustion, a failed/missing
# batch item). This is *not* a refusal — a refusal is a successful call that
# returns refusal text (or empty content). Services emit it only from their
# failure paths (the `except` handler, or an errored batch item). Downstream,
# rows whose response carries this sentinel are flagged
# `is_correctly_processed=False` and EXCLUDED from metric denominators, so an
# untestable prompt is never miscounted as a defeated attack.
#
# The null byte makes the marker impossible to confuse with real model output.
MECHANISM_ERROR_SENTINEL = "\x00__MECHANISM_ERROR__\x00"


def make_mechanism_error(message: str) -> str:
    """Wrap a failure message as a mechanism-error sentinel response."""
    return f"{MECHANISM_ERROR_SENTINEL}{message}"


def is_mechanism_error(response: Any) -> bool:
    """True iff `response` is a mechanism-error sentinel (a real processing
    failure, NOT a refusal)."""
    return isinstance(response, str) and response.startswith(
        MECHANISM_ERROR_SENTINEL)


def strip_mechanism_error(response: str) -> str:
    """Return the human-readable failure message (sentinel prefix removed)."""
    if is_mechanism_error(response):
        return response[len(MECHANISM_ERROR_SENTINEL):]
    return response


def _backoff_seconds(attempt: int, base: float = 2.0, max_wait: float = 60.0) -> float:
    """Exponential backoff with jitter, capped at `max_wait`."""
    return min(base ** attempt + random.random() * 2, max_wait)


@dataclass
class UsageStats:
    """Tracks inference count, token usage, and cost."""
    inference_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    def record(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        self.inference_count += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost += cost

    def to_dict(self) -> dict:
        return asdict(self)


class BaseLLMService(ABC):
    """Abstract base class for LLM services.

    Tracks two usage accumulators:
    - algorithm_usage: only non-test calls (optimization algorithm cost)
    - total_usage: all calls (algorithm + test evaluation)
    """

    def __init__(
        self,
        max_concurrency: int = 20,
        max_retries: int = 5,
        batch_poll_interval: int = 30,
        batch_timeout: int = 3600,
    ):
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.batch_poll_interval = batch_poll_interval
        self.batch_timeout = batch_timeout
        self.algorithm_usage = UsageStats()
        self.total_usage = UsageStats()

    def _record_usage(
        self, input_tokens: int, output_tokens: int, cost: float, is_test: bool,
    ) -> None:
        self.total_usage.record(input_tokens, output_tokens, cost)
        if not is_test:
            self.algorithm_usage.record(input_tokens, output_tokens, cost)

    def _check_fatal_error(self, error: Exception, model_id: str) -> None:
        """Raise ``FatalModelError`` for 404 / model-not-found errors."""
        error_str = str(error).lower()
        if "not found" in error_str or "does not exist" in error_str or "404" in str(error):
            from .exceptions import FatalModelError
            raise FatalModelError(f"Model {model_id} not found") from error

    def _raise_if_account_fatal(self, error: BaseException) -> None:
        """Convert an account-global failure into a fatal exception that ABORTS
        the whole run. Call this from a service's error handler BEFORE the
        rate-limit retry branch (credit-exhaustion arrives as a 429 for several
        providers, so it must be caught first). No-op for any other error.

        Distinct from ``_check_fatal_error`` (per-MODEL 404): a bad key or an
        empty balance dooms every task on this provider, so retrying other cells
        is wasted wall-clock and the run should stop with an actionable message.
        """
        provider = getattr(self.model, "provider", None)
        provider_name = getattr(provider, "value", None) or self.__class__.__name__
        detail = str(error)[:200]
        if is_credit_exhausted_error(error):
            from .exceptions import CreditsExhaustedError
            raise CreditsExhaustedError(
                f"{provider_name}: account is out of credits / over billing quota "
                f"({detail}). Top up this provider's account, then rerun — the run "
                f"was aborted so no cells are miscounted as defeated attacks."
            ) from error
        if is_invalid_credential_error(error):
            from .exceptions import InvalidCredentialError
            raise InvalidCredentialError(
                f"{provider_name}: API key rejected — invalid or revoked "
                f"({detail}). Fix the key (check the env var) "
                f"for this provider, then rerun — the run was aborted."
            ) from error

    def get_usage(self) -> dict:
        return {
            "algorithm": self.algorithm_usage.to_dict(),
            "total": self.total_usage.to_dict(),
        }

    def reset_usage(self) -> None:
        self.algorithm_usage = UsageStats()
        self.total_usage = UsageStats()

    # ------------------------------------------------------------------
    # Rate-limit retry helpers (shared across all services).
    # ------------------------------------------------------------------

    def _retry_rate_limit_sync(
        self,
        fn: Callable[[], _T],
        label: str,
        *,
        max_retries: Optional[int] = None,
        max_wait_seconds: float = 60.0,
    ) -> _T:
        """Call `fn()`; on rate-limit error, sleep + retry up to max_retries.

        Used by batch-API services (Google, Anthropic) whose `client.batches.*`
        calls are blocking. Non-rate-limit exceptions propagate immediately.
        """
        retries = self.max_retries if max_retries is None else max_retries
        for attempt in range(retries + 1):
            try:
                return fn()
            except Exception as e:
                if is_rate_limit_error(e) and attempt < retries:
                    wait = _backoff_seconds(attempt, max_wait=max_wait_seconds)
                    logger.warning(
                        f"{label}: rate-limit, retry {attempt + 1}/{retries} "
                        f"after {wait:.1f}s — {str(e)[:120]}")
                    time.sleep(wait)
                    continue
                raise
        # Unreachable — loop either returns or raises.
        raise RuntimeError(f"{label}: exhausted retries")

    async def _retry_rate_limit_async(
        self,
        fn: Callable[[], Awaitable[_T]],
        label: str,
        *,
        max_retries: Optional[int] = None,
        max_wait_seconds: float = 60.0,
    ) -> _T:
        """Async variant of `_retry_rate_limit_sync` for per-call async services
        (OpenAI, vLLM/NU_CLUSTER)."""
        retries = self.max_retries if max_retries is None else max_retries
        for attempt in range(retries + 1):
            try:
                return await fn()
            except Exception as e:
                if is_rate_limit_error(e) and attempt < retries:
                    wait = _backoff_seconds(attempt, max_wait=max_wait_seconds)
                    logger.warning(
                        f"{label}: rate-limit, retry {attempt + 1}/{retries} "
                        f"after {wait:.1f}s — {str(e)[:120]}")
                    await asyncio.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"{label}: exhausted retries")

    @abstractmethod
    def batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        system_message: Optional[str] = None,
        is_test: bool = False,
        **kwargs,
    ) -> List[Tuple[str, str]]:
        """Process conversations in batch.

        Args:
            conversations: List of ``(id, messages)`` tuples where *messages*
                is a list of ``(text, image_or_None)`` tuples.
            system_message: Optional system instruction prepended to each
                conversation.
            is_test: If True usage is only counted in ``total_usage``.
            **kwargs: Model-specific overrides (temperature, max_tokens …).

        Returns:
            List of ``(id, response_text)`` tuples in the same order as input.
        """
        raise NotImplementedError
