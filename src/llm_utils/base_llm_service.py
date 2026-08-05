"""
Base abstract class for LLM services with usage tracking.

The core public method every service implements is ``batch_chat``; the base
also declares the single-prompt conveniences (``chat`` / ``achat``),
``chat_structured``, ``batch_chat_with_logprobs``, and the resumable batch
trio (``submit_batch_chat`` / ``batch_chat_status`` / ``harvest_batch_chat``)
— each implemented per provider where the serving route supports it.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
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
    "throttl",            # AWS ThrottlingException / "throttled" / "throttle"
    "toomanyrequests",    # AWS TooManyRequestsException (no spaces)
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
# failure paths (the `except` handler, or an errored batch item). Callers
# should check `is_mechanism_error(resp)` to distinguish a real failure (retry
# / escalate / exclude from result denominators) from a valid model response
# (which may itself be a refusal).
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


# --- Native-batch auto-routing heuristics ----------------------------------
# Shared by every service with a native batch API (OpenAI, Anthropic, Google).
# Deliberately crude: routing only needs the cost estimate right within
# ~2-3x. Tunables surface as constructor params (a library's config seam);
# tuning the defaults against real ledger data is filed in the repo TODO.
_DEFAULT_BATCH_THRESHOLD_USD = 1.0   # est. job cost at/above which auto mode batches
_EST_CHARS_PER_TOKEN = 4             # rough text-token estimate
_EST_IMAGE_TOKENS = 1000             # flat per-image token estimate


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

    # The native batch APIs (OpenAI Batch, Anthropic Message Batches, Google
    # batch mode) all bill at 50% of the realtime list prices the registry
    # stores, so every batch-path cost recording multiplies by this factor.
    # Registry prices stay realtime list — never store batch prices there.
    BATCH_COST_DISCOUNT: float = 0.5

    # Consumer-installable usage hook — the seam for durable accounting (e.g. a
    # cost ledger). `_record_usage` is the ONE choke point every provider
    # funnels through, so a registered hook sees every call; the in-memory
    # UsageStats above die with the process, the hook is how a consumer
    # persists spend. Signature:
    #   hook(model, input_tokens, output_tokens, cost_usd, is_test=..., label=...)
    _usage_hook: Optional[Callable[..., None]] = None
    _usage_hook_warned: bool = False

    @classmethod
    def set_usage_hook(cls, hook: Callable[..., None]) -> None:
        """Register a callable invoked after every recorded call (all services).

        The hook must be cheap; exceptions it raises are swallowed (one stderr
        warning per process) — accounting must never kill a call.
        """
        cls._usage_hook = hook

    @classmethod
    def clear_usage_hook(cls) -> None:
        """Unregister the usage hook."""
        cls._usage_hook = None

    def __init__(
        self,
        max_concurrency: int = 20,
        max_retries: int = 5,
        batch_poll_interval: int = 30,
        batch_timeout: int = 3600,
        use_batch_api: Optional[bool] = None,
        batch_threshold_usd: Optional[float] = None,
    ):
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.batch_poll_interval = batch_poll_interval
        self.batch_timeout = batch_timeout
        # Native-batch routing (services with a native batch API only):
        #   None  → auto: batch iff estimated job cost ≥ batch_threshold_usd
        #   True  → always use the native batch API
        #   False → never (always the concurrent realtime path)
        self.use_batch_api = use_batch_api
        self.batch_threshold_usd = (
            _DEFAULT_BATCH_THRESHOLD_USD
            if batch_threshold_usd is None else batch_threshold_usd)
        self.algorithm_usage = UsageStats()
        self.total_usage = UsageStats()
        # Free-form accounting tag for the usage hook (set via
        # LLMServiceFactory.create(..., label=...)). None = unlabeled.
        self.usage_label: Optional[str] = None

    def _record_usage(
        self, input_tokens: int, output_tokens: int, cost: float, is_test: bool,
    ) -> None:
        self.total_usage.record(input_tokens, output_tokens, cost)
        if not is_test:
            self.algorithm_usage.record(input_tokens, output_tokens, cost)
        if BaseLLMService._usage_hook is not None:
            try:
                BaseLLMService._usage_hook(
                    getattr(self, "model", None),
                    input_tokens, output_tokens, cost,
                    is_test=is_test, label=self.usage_label,
                )
            except Exception as e:  # noqa: BLE001 — accounting never kills a call
                if not BaseLLMService._usage_hook_warned:
                    BaseLLMService._usage_hook_warned = True
                    logger.warning(
                        f"usage hook failed ({str(e)[:90]}) — "
                        f"calls proceed unrecorded this process")

    def _accepts_temperature(self) -> bool:
        """Single source of truth for the temperature quirk across ALL providers.

        Newer reasoning models — OpenAI GPT-5 / o-series, Anthropic Opus 4.7+,
        Moonshot Kimi K3 — reject a custom ``temperature`` with a 400 and must
        be sent none. A model declares this once in the registry via
        ``ModelQuirk.NO_CUSTOM_TEMPERATURE``; every service gates its
        temperature through here, so adding a new such model is a one-line
        registry change.
        """
        from .llm_model import ModelQuirk
        return not self.model.has_quirk(ModelQuirk.NO_CUSTOM_TEMPERATURE)

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

    # How this provider's remaining-credit balance can be queried, statically
    # discoverable so a monitor can branch policy WITHOUT a network call:
    #   "api_key"        → get_account_status() works with the service's own key
    #   "management_key" → the provider has an endpoint but it needs a separate
    #                      management/admin credential (not implemented here)
    #   None             → no balance endpoint exists (postpaid billing, or the
    #                      provider simply doesn't expose one) — a consumer must
    #                      estimate from its own usage ledger (the usage hook)
    #                      and rely on CreditsExhaustedError as the reactive
    #                      backstop.
    BALANCE_QUERY_VIA: Optional[str] = None

    def get_account_status(self) -> "AccountStatus":
        """Point-in-time credit state from the provider's own billing endpoint.

        Complements `get_usage` (in-memory, this process only): this queries
        the ACCOUNT — spend from every machine and session shows up here.
        Default is unsupported; services whose provider exposes a balance
        endpoint usable with the normal API key override `_fetch_account_status`.
        Network/auth failures propagate — a monitor must see a failed check
        as failed, never as "no balance data".
        """
        from .account_status import AccountStatus
        provider = getattr(self, "SERVICE_NAME", type(self).__name__)
        fetch = getattr(self, "_fetch_account_status", None)
        if fetch is None:
            return AccountStatus(provider=provider, supported=False)
        return fetch()

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
        (OpenAI, vLLM/SLURM_CLUSTER)."""
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

    # ------------------------------------------------------------------
    # Native-batch auto-routing (shared by OpenAI / Anthropic / Google).
    # ------------------------------------------------------------------

    def _supports_native_batch(self) -> bool:
        """Whether this service can reach a native batch API. Overridden by
        the services that have one; the default keeps every other route on
        the realtime path."""
        return False

    def _estimate_cost_usd(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        max_tokens: int,
    ) -> float:
        """Crude worst-case job cost over seam-format conversations: chars/4
        (+ flat per image) input tokens, full ``max_tokens`` output per
        request. Only needs to be right within ~2-3x to route correctly."""
        in_tokens = 0
        for _cid, messages in conversations:
            for text, image in messages:
                in_tokens += len(text or "") // _EST_CHARS_PER_TOKEN
                if image is not None:
                    images = image if isinstance(image, list) else [image]
                    in_tokens += _EST_IMAGE_TOKENS * sum(
                        1 for img in images if img is not None)
        out_tokens = len(conversations) * max_tokens
        return (
            in_tokens * self.model.input_price
            + out_tokens * self.model.output_price
        ) / 1_000_000

    def _route_to_native_batch(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        max_tokens: int,
    ) -> bool:
        """Decide realtime vs native batch for this job (see ``use_batch_api``)."""
        if not self._supports_native_batch():
            return False
        if self.use_batch_api is not None:
            return self.use_batch_api
        est = self._estimate_cost_usd(conversations, max_tokens)
        to_batch = est >= self.batch_threshold_usd
        logger.info(
            f"auto-route: est. job cost ${est:.2f} "
            f"{'≥' if to_batch else '<'} ${self.batch_threshold_usd:.2f} → "
            f"{'native batch (50% price)' if to_batch else 'realtime'}")
        return to_batch

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

    def batch_chat_with_logprobs(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        system_message: Optional[str] = None,
        is_test: bool = False,
        top_logprobs: int = 5,
        **kwargs,
    ) -> List[Tuple[str, str, Optional[dict]]]:
        """``batch_chat`` plus per-token logprobs for each sampled response.

        A separate method on purpose: ``batch_chat``'s ``(id, text)`` return
        shape is load-bearing for every consumer, so logprobs ride a THIRD
        element here instead of widening it. Each result is
        ``(id, text, logprobs)`` where *logprobs* is the OpenAI-schema payload
        (``{"content": [{"token", "logprob", "top_logprobs": [...]}, ...]}``,
        one entry per generated token, each with the ``top_logprobs``
        alternatives), or None when the call failed (mechanism error) or the
        server returned no logprobs.

        Implemented only on ``SlurmClusterService`` (vLLM's OpenAI-compatible
        endpoint returns logprobs when asked). Anthropic and Google APIs do
        not return logprobs at all, so their services keep this default (a
        permanent gap, not a TODO); the OpenAI/compatible API services simply
        haven't needed it yet.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose token logprobs "
            "(only the vLLM cluster serving route implements this today)"
        )

    # ------------------------------------------------------------------
    # Resumable batch trio — implemented by services with a native batch
    # API (OpenAI, Anthropic, Google). Declared here so a duck-typed caller
    # gets a clean NotImplementedError instead of an AttributeError.
    # ------------------------------------------------------------------

    def submit_batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Submit a native batch WITHOUT waiting; returns the provider batch
        id. Persist the id anywhere and harvest from a later process."""
        raise NotImplementedError(
            f"{type(self).__name__} has no native batch API "
            "(resumable batches exist on OpenAI, Anthropic, and Google services)"
        )

    def batch_chat_status(self, batch_id: str) -> str:
        """The provider's status string for a previously submitted batch."""
        raise NotImplementedError(
            f"{type(self).__name__} has no native batch API"
        )

    def harvest_batch_chat(
        self, batch_id: str, *, is_test: bool = False,
    ) -> Optional[List[Tuple[str, str]]]:
        """Results of a previously submitted batch, or None while running."""
        raise NotImplementedError(
            f"{type(self).__name__} has no native batch API"
        )

    # ------------------------------------------------------------------
    # Single-prompt convenience. `achat` offloads the sync call to a
    # worker thread so it composes with an async runtime without the
    # nested-event-loop error `batch_chat`'s internal `asyncio.run` raises.
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        *,
        is_test: bool = False,
        **kwargs,
    ) -> str:
        """One prompt → one response string."""
        out = self.batch_chat(
            [("_one", [(prompt, None)])],
            system_message=system_message,
            is_test=is_test,
            **kwargs,
        )
        if not out:
            return make_mechanism_error("batch_chat returned no results")
        return out[0][1]

    async def achat(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        *,
        is_test: bool = False,
        **kwargs,
    ) -> str:
        """Async one-prompt call — runs the sync call in a worker thread."""
        return await asyncio.to_thread(
            self.chat, prompt, system_message, is_test=is_test, **kwargs
        )

    def chat_structured(
        self,
        prompt: str,
        output_schema: Any,
        system_message: Optional[str] = None,
        *,
        is_test: bool = False,
        **kwargs,
    ) -> Any:
        """One prompt → one instance of `output_schema` (a pydantic BaseModel
        subclass), via the provider's structured-output / parse API.

        Unlike `chat` (which reports API failures as mechanism-error strings),
        this RAISES on failure after retries — a typed return has no
        error-string channel. May return None when the model's output failed
        schema validation. Implemented per provider; no generic fallback.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support structured output yet"
        )
