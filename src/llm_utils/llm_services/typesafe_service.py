"""TypeSafe AI "System One" typed evaluation (model family Jev).

Not a text generator: one request sends a ``state`` (text, a JSON object, or a
list of messages; state plus the longest question <= 32k tokens) and named
typed questions, and gets one typed answer per question, all evaluated in one
pass:

- ``noul``: a yes/no question -> probability of yes;
- ``choice``: pick one of up to 255 named options -> the choice, a confidence
  and per-option probabilities;
- ``score``: rate on 2-10 ordered levels -> the probability-weighted score, a
  confidence and per-level probabilities.

Wire: ``POST {base}/v1/systemone`` with ``Authorization: Bearer
$TYPESAFE_API_KEY`` (request/response schema from the provider's OpenAPI, as
shipped in its ``typesafe-sdk`` package); raw HTTP through httpx2, no SDK
dependency. Retries 408/429/5xx (529 = overloaded), honoring Retry-After.
Billing is per input token (output tokens are free); each call records usage
through the same choke point as every other service, so the usage hook or the
default call ledger gets its cost row.

The caller pins the model id (``LLMModel.JEV_1_13_0``) so behavior does not
move under an alias. Data handling: the direct API has no per-request
zero-retention switch, so ``evaluate(..., personal_data=True)`` is refused
unless the deployment built the service with ``zero_retention=True``, a
statement that the account has a written zero-retention agreement.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import httpx2

from ..base_llm_service import BaseLLMService, _backoff_seconds
from ..constants import TYPESAFE_API_URL
from ..exceptions import (
    BrokerModeUnsupportedError, CreditsExhaustedError, FatalModelError,
    InvalidCredentialError, RetentionPolicyError,
)
from ..llm_model import LLMModel
from .._logging import get_logger

logger = get_logger(__name__)

MAX_CHOICE_OPTIONS = 255     # provider limit
SCORE_LEVELS = (2, 10)       # provider limits, inclusive
_RETRY_STATUSES = frozenset({408, 429}) | frozenset(range(500, 600))
_MAX_RETRY_WAIT_S = 30.0     # a longer Retry-After is not waited out: the error surfaces
# Only failures before the request reached the server are retried; a read
# timeout after sending may already have been billed.
_RETRY_TRANSPORT = (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.PoolTimeout)

JSONContent = Union[str, Dict[str, Any], list]


# ---------------------------------------------------------------------------
# Question builders (plain dicts in the wire format; dicts may also be passed
# directly)
# ---------------------------------------------------------------------------

def noul_question(instructions: Optional[JSONContent] = None, *,
                  true: Optional[JSONContent] = None,
                  false: Optional[JSONContent] = None) -> dict:
    """A yes/no question; ``true``/``false`` optionally describe each outcome."""
    q: dict = {"type": "noul"}
    if instructions is not None:
        q["instructions"] = instructions
    criteria = {k: v for k, v in (("true", true), ("false", false)) if v is not None}
    if criteria:
        q["criteria"] = criteria
    return q


def choice_question(criteria: Union[Mapping[str, Optional[JSONContent]], Sequence[str]],
                    instructions: Optional[JSONContent] = None) -> dict:
    """Pick one option. ``criteria`` maps option name -> description (or None),
    or is a plain list of option names. 2 to 255 options."""
    if not isinstance(criteria, Mapping):
        criteria = {name: None for name in criteria}
    if not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
        raise ValueError(f"a choice question needs 2..{MAX_CHOICE_OPTIONS} options, "
                         f"got {len(criteria)}")
    q: dict = {"type": "choice", "criteria": dict(criteria)}
    if instructions is not None:
        q["instructions"] = instructions
    return q


def score_question(levels: Sequence[JSONContent],
                   instructions: Optional[JSONContent] = None) -> dict:
    """Rate on ordered levels; level i (from 0) is described by ``levels[i]``.
    2 to 10 levels."""
    lo, hi = SCORE_LEVELS
    if isinstance(levels, (str, bytes)) or not lo <= len(levels) <= hi:
        raise ValueError(f"a score question needs {lo}..{hi} levels")
    q: dict = {"type": "score", "criteria": list(levels)}
    if instructions is not None:
        q["instructions"] = instructions
    return q


def _check_questions(questions: Mapping[str, dict]) -> None:
    if not questions:
        raise ValueError("at least one question is required")
    for name, q in questions.items():
        kind = q.get("type") if isinstance(q, Mapping) else None
        if kind not in ("noul", "choice", "score"):
            raise ValueError(f"question {name!r}: type must be noul, choice or score")
        if kind == "choice":
            n = len(q.get("criteria") or {})
            if not 2 <= n <= MAX_CHOICE_OPTIONS:
                raise ValueError(f"question {name!r}: 2..{MAX_CHOICE_OPTIONS} options, got {n}")
        if kind == "score":
            n = len(q.get("criteria") or [])
            if not SCORE_LEVELS[0] <= n <= SCORE_LEVELS[1]:
                raise ValueError(f"question {name!r}: {SCORE_LEVELS[0]}..{SCORE_LEVELS[1]} "
                                 f"levels, got {n}")


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoulAnswer:
    noul: float                       # probability of yes, 0..1


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class ScoreAnswer:
    score: float                      # probability-weighted level, may be fractional
    confidence: float
    probabilities: Dict[str, float]
    legend: Dict[str, Any] = field(default_factory=dict)


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


@dataclass(frozen=True)
class Evaluation:
    """One typed evaluation: answers keyed by question name."""
    model: str                        # the model that answered (resolves aliases)
    answers: Dict[str, Answer]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float


def _parse_answer(name: str, raw: Mapping[str, Any]) -> Answer:
    kind = raw.get("type")
    if kind == "noul":
        return NoulAnswer(float(raw["noul"]))
    if kind == "choice":
        return ChoiceAnswer(str(raw["choice"]), float(raw["confidence"]),
                            {k: float(v) for k, v in raw["probabilities"].items()})
    if kind == "score":
        return ScoreAnswer(float(raw["score"]), float(raw["confidence"]),
                           {k: float(v) for k, v in raw["probabilities"].items()},
                           dict(raw.get("legend") or {}))
    raise ValueError(f"answer {name!r}: unknown type {kind!r}")


class TypeSafeHTTPError(RuntimeError):
    """A non-success answer from the TypeSafe API after retries."""

    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class TypeSafeService(BaseLLMService):
    """Typed evaluation on TypeSafe's System One API (``evaluate``); it has no
    chat interface, so ``batch_chat`` / ``chat`` raise ``NotImplementedError``."""

    API_KEY_ENV = "TYPESAFE_API_KEY"
    BASE_URL = TYPESAFE_API_URL
    SERVICE_NAME = "TypeSafe"
    # evaluate is a paid entry point: admitted against the run spend cap.
    _SPEND_GUARDED = BaseLLMService._SPEND_GUARDED + ("evaluate",)

    def __init__(self, model: LLMModel, **kwargs):
        super().__init__(max_concurrency=kwargs.pop("max_concurrency", 8),
                         max_retries=kwargs.pop("max_retries", 3))
        if os.getenv("LLM_UTILS_BROKER_URL") is not None:
            raise BrokerModeUnsupportedError(
                "TypeSafe is not a broker route; unset LLM_UTILS_BROKER_URL for this process")
        self.model = model
        self.max_tokens = 0            # no text output; keeps the base estimator total
        self.api_key = kwargs.get("api_key") or os.getenv(self.API_KEY_ENV)
        if not self.api_key:
            raise ValueError(f"{self.SERVICE_NAME} API key not found. Set {self.API_KEY_ENV} "
                             "in the environment or pass the api_key parameter")
        self.base_url = (kwargs.get("base_url") or self.BASE_URL).rstrip("/")
        self.call_timeout: float = kwargs.get("call_timeout", 30.0)
        # Set only when the account has a written zero-retention agreement.
        self.zero_retention: bool = bool(kwargs.get("zero_retention", False))
        self._client = httpx2.Client(timeout=self.call_timeout, follow_redirects=False)

    def close(self) -> None:
        self._client.close()
        super().close()

    # -- no chat surface ----------------------------------------------------

    def batch_chat(self, conversations, system_message=None, is_test=False, **kwargs):
        raise NotImplementedError(
            "TypeSafe Jev answers typed questions, it does not chat: use evaluate()")

    # -- cost estimate for the spend cap -------------------------------------

    def _spend_estimate(self, name: str, args: tuple, kwargs: dict) -> float:
        if name != "evaluate":
            return super()._spend_estimate(name, args, kwargs)
        state = args[0] if args else kwargs.get("state", "")
        questions = args[1] if len(args) > 1 else kwargs.get("questions", {})
        chars = len(json.dumps(state, default=str)) + len(json.dumps(questions, default=str))
        return (chars // 4) * self.model.input_price / 1_000_000

    # -- the call -------------------------------------------------------------

    def evaluate(self, state: JSONContent, questions: Mapping[str, dict], *,
                 personal_data: bool = False, is_test: bool = False) -> Evaluation:
        """Ask typed ``questions`` about ``state`` in one request.

        ``personal_data=True`` declares that ``state`` carries personal data;
        it is refused (``RetentionPolicyError``, before sending) unless this
        service was built with ``zero_retention=True``. Raises on failure after
        retries: ``InvalidCredentialError`` (401), ``CreditsExhaustedError``
        (402), ``FatalModelError`` (unknown model), ``ValueError`` (422
        validation), else ``TypeSafeHTTPError`` / the transport exception.
        """
        if personal_data and not self.zero_retention:
            raise RetentionPolicyError(
                "personal data needs a zero-retention route: the direct TypeSafe API "
                "has no per-request retention switch; build the service with "
                "zero_retention=True only under a written zero-retention agreement")
        _check_questions(questions)
        body = {"state": state, "model": self.model.model_id, "questions": dict(questions)}
        started = time.monotonic()
        try:
            payload = self._post_with_retries(body)
        except Exception as e:
            self._record_failure(e, is_test=is_test)
            raise
        # Record the billed usage first: a response we cannot parse was still paid for.
        usage = payload.get("usage") or {}
        in_tok = int(usage.get("input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        cost = (in_tok * self.model.input_price + out_tok * self.model.output_price) / 1_000_000
        self._record_usage(in_tok, out_tok, cost, is_test)
        answers = {name: _parse_answer(name, raw) for name, raw in payload["answers"].items()}
        return Evaluation(str(payload.get("model", self.model.model_id)), answers,
                          in_tok, out_tok, cost, time.monotonic() - started)

    async def aevaluate(self, state: JSONContent, questions: Mapping[str, dict], *,
                        personal_data: bool = False, is_test: bool = False) -> Evaluation:
        """Async ``evaluate`` (runs the sync call in a worker thread)."""
        return await asyncio.to_thread(self.evaluate, state, questions,
                                       personal_data=personal_data, is_test=is_test)

    def _post_with_retries(self, body: dict) -> dict:
        url = f"{self.base_url}/v1/systemone"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json", "Accept": "application/json"}
        attempt = 0
        while True:
            try:
                resp = self._client.post(url, json=body, headers=headers)
            except _RETRY_TRANSPORT as e:
                if attempt >= self.max_retries:
                    raise
                wait = _backoff_seconds(attempt, max_wait=_MAX_RETRY_WAIT_S)
                logger.warning(f"{self.SERVICE_NAME} {type(e).__name__}; retry {attempt + 1}/"
                               f"{self.max_retries} in {wait:.1f}s")
                time.sleep(wait)
                attempt += 1
                continue
            code = resp.status_code
            if 200 <= code < 300:
                return resp.json()
            asked = _retry_after_s(resp)
            if (code in _RETRY_STATUSES and attempt < self.max_retries
                    and (asked is None or asked <= _MAX_RETRY_WAIT_S)):
                wait = asked if asked is not None else _backoff_seconds(attempt, max_wait=_MAX_RETRY_WAIT_S)
                logger.warning(f"{self.SERVICE_NAME} HTTP {code}; retry {attempt + 1}/"
                               f"{self.max_retries} in {wait:.1f}s")
                time.sleep(wait)
                attempt += 1
                continue
            raise _http_error(code, resp)


def _retry_after_s(resp) -> Optional[float]:
    for header, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = resp.headers.get(header)
        if value:
            try:
                return max(0.0, float(value) / scale)
            except ValueError:
                return None
    return None


def _http_error(code: int, resp) -> Exception:
    try:
        detail = resp.json().get("detail")
    except Exception:  # noqa: BLE001 — a non-JSON error body
        detail = None
    message = str(detail)[:300] if detail else (resp.text or "")[:200]
    if code == 401:
        return InvalidCredentialError(f"TypeSafe rejected the API key: {message}")
    if code == 402:
        return CreditsExhaustedError(f"TypeSafe account cannot pay: {message}")
    if code == 404:
        return FatalModelError(f"TypeSafe model or endpoint not found: {message}")
    if code == 422:
        return ValueError(f"TypeSafe rejected the request: {message}")
    return TypeSafeHTTPError(code, message)
