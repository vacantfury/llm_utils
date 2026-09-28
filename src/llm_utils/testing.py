"""Test kit for consumers of llm_utils: an offline fake service, a factory
patch, and the exception-hierarchy contract check.

Import these explicitly in your own test suite; nothing here registers itself
as a pytest plugin.

    from llm_utils import LLMModel
    from llm_utils.testing import FakeService, use_fake_service, check_exception_contract

    def test_judge_parses_verdicts():
        fake = FakeService(LLMModel.GPT_5_MINI, responses={"q1": "unsafe", "q2": "safe"})
        with use_fake_service(fake):
            run_my_judge(...)                 # calls LLMServiceFactory.create(...)
        assert [c.conversation_id for c in fake.calls] == ["q1", "q2"]

    def test_llm_utils_exception_contract():
        check_exception_contract()            # raises AssertionError on a break

``FakeService`` is a real ``BaseLLMService`` subclass: ``chat`` / ``achat`` /
``batch_chat`` / ``chat_structured`` follow the same return and error
contract as the provider services, usage is counted in ``get_usage()`` from
the model's registry prices, a consumer usage hook (if installed) is called,
and paid-model fakes go through the run spend cap. It never writes the
default call ledger, so a test run leaves no durable rows.
"""
from __future__ import annotations

import contextlib
import inspect
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from . import spend as _spend
from .base_llm_service import BaseLLMService, make_mechanism_error
from .claude_api_policy import ClaudeAPINotAllowed
from .exceptions import (
    AccountFatalError, BrokerError, BrokerModeUnsupportedError, CreditsExhaustedError,
    FatalModelError, InvalidCredentialError, RetentionPolicyError, SpendCapExceededError,
)
from .llm_model import LLMModel

# Rough token count for fake usage, matching the cost estimator's heuristic.
_CHARS_PER_TOKEN = 4

Responder = Callable[[str, str], Any]


@dataclass(frozen=True)
class FakeCall:
    """One request the fake received."""
    method: str                       # "batch_chat" | "chat_structured"
    conversation_id: str
    messages: Tuple[Tuple[str, Any], ...]
    system_message: Optional[str]
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """The last message's text (the prompt, for single-turn calls)."""
        return self.messages[-1][0] if self.messages else ""


class FakeService(BaseLLMService):
    """Offline stand-in for any llm_utils service.

    ``responses`` decides each answer:

    - a string: every request gets it;
    - a list (or any iterable): answers in order, one per request; running out
      raises ``AssertionError``;
    - a dict: keyed by conversation id, else by the request's last message
      text; unmatched requests get ``default``;
    - a callable ``(text, conversation_id) -> answer``.

    An answer that is an exception instance behaves like a provider failure:
    ``AccountFatalError`` and ``FatalModelError`` subclasses are raised (as the
    real services do); any other exception becomes a mechanism-error string
    and one failed-call record. For ``chat_structured`` the answer is returned
    as is (pass the pydantic instance you want back).
    """

    def __init__(
        self,
        model: Optional[LLMModel] = None,
        responses: Union[str, List[Any], Dict[str, Any], Responder, None] = None,
        *,
        default: Any = "fake response",
        max_tokens: int = 4096,
        expected_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(max_concurrency=1, max_retries=0)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = kwargs.get("temperature", 0.0)
        self.expected_output_tokens = expected_output_tokens
        self.default = default
        self.calls: List[FakeCall] = []
        self._responses = responses
        if responses is not None and not isinstance(responses, (str, dict)) and not callable(responses):
            self._responses = deque(responses)

    # -- answers -----------------------------------------------------------

    def _answer(self, text: str, cid: str) -> Any:
        r = self._responses
        if r is None:
            return self.default
        if isinstance(r, str):
            return r
        if isinstance(r, dict):
            if cid in r:
                return r[cid]
            return r.get(text, self.default)
        if isinstance(r, deque):
            if not r:
                raise AssertionError(
                    f"FakeService: no scripted response left for request {cid!r} ({text[:60]!r})")
            return r.popleft()
        return r(text, cid)

    def _usage(self, text: str, answer: str) -> Tuple[int, int, float]:
        in_tok = max(1, len(text) // _CHARS_PER_TOKEN)
        out_tok = max(1, len(answer) // _CHARS_PER_TOKEN)
        cost = 0.0
        if self.model is not None:
            cost = (in_tok * self.model.input_price + out_tok * self.model.output_price) / 1_000_000
        return in_tok, out_tok, cost

    # -- the service seam ---------------------------------------------------

    def batch_chat(self, conversations, system_message=None, is_test=False, **kwargs):
        out = []
        for cid, messages in conversations:
            call = FakeCall("batch_chat", cid, tuple(messages), system_message, dict(kwargs))
            self.calls.append(call)
            text = " ".join(str(t or "") for t, _ in messages)
            answer = self._answer(call.text, cid)
            if isinstance(answer, BaseException):
                self._record_failure(answer, is_test=is_test)
                if isinstance(answer, (AccountFatalError, FatalModelError)):
                    raise answer
                out.append((cid, make_mechanism_error(f"{type(answer).__name__}: {answer}")))
                continue
            answer = str(answer)
            self._record_usage(*self._usage(text, answer), is_test)
            out.append((cid, answer))
        return out

    def chat_structured(self, prompt, output_schema, system_message=None, *, is_test=False, **kwargs):
        call = FakeCall("chat_structured", "_one", ((prompt, None),), system_message,
                        {"output_schema": output_schema, **kwargs})
        self.calls.append(call)
        answer = self._answer(prompt, "_one")
        if isinstance(answer, BaseException):
            self._record_failure(answer, is_test=is_test)
            raise answer
        self._record_usage(*self._usage(prompt, str(answer)), is_test)
        return answer

    # -- accounting: never the durable default ledger -----------------------

    def _record_usage(self, input_tokens, output_tokens, cost, is_test):
        if BaseLLMService._usage_hook is not None:
            return super()._record_usage(input_tokens, output_tokens, cost, is_test)
        self.total_usage.record(input_tokens, output_tokens, cost)
        if not is_test:
            self.algorithm_usage.record(input_tokens, output_tokens, cost)
        if self._spend_capped():
            _spend.record(cost)

    def _record_failure(self, error, *, status=None, is_test=False):
        if BaseLLMService._usage_hook is not None:
            super()._record_failure(error, status=status, is_test=is_test)


@contextlib.contextmanager
def use_fake_service(
    fake: Union[BaseLLMService, Callable[..., BaseLLMService]],
) -> Iterator[Union[BaseLLMService, Callable[..., BaseLLMService]]]:
    """Make ``LLMServiceFactory.create`` return ``fake`` inside the block.

    ``fake`` is one service instance (returned for every create; its ``model``
    is filled from the first call when it has none) or a callable
    ``(model, **kwargs) -> service`` to build one per call. String models are
    resolved to ``LLMModel`` first, as the real factory does. The original
    ``create`` is restored on exit, even after an exception.
    """
    from .llm_service_factory import LLMServiceFactory

    original = LLMServiceFactory.__dict__["create"]

    def create(cls, model, *, label=None, launch_point=None, **kwargs):
        if isinstance(model, str):
            model = LLMModel.from_string(model)
        if isinstance(fake, BaseLLMService):
            service = fake
            if getattr(service, "model", None) is None:
                service.model = model
        else:
            service = fake(model, **kwargs)
        if label:
            service.usage_label = label
        if launch_point:
            service.launch_point = launch_point
        return service

    LLMServiceFactory.create = classmethod(create)
    try:
        yield fake
    finally:
        LLMServiceFactory.create = original


# ---------------------------------------------------------------------------
# Exception-hierarchy contract
# ---------------------------------------------------------------------------

# Every exception consumers catch, and the base each must keep. Runners catch
# AccountFatalError to abort a whole run and FatalModelError to drop one
# model; a flattened hierarchy (the v2.1.0 break) silently turns an abort into
# a grind of mechanism errors.
EXCEPTION_CONTRACT: Dict[type, type] = {
    FatalModelError: Exception,
    AccountFatalError: Exception,
    InvalidCredentialError: AccountFatalError,
    CreditsExhaustedError: AccountFatalError,
    SpendCapExceededError: AccountFatalError,
    BrokerError: RuntimeError,
    BrokerModeUnsupportedError: BrokerError,
    ClaudeAPINotAllowed: PermissionError,
    RetentionPolicyError: PermissionError,
}

# Pairs that must stay unrelated: catching one must never catch the other.
_DISJOINT = ((FatalModelError, AccountFatalError),)


def check_exception_contract() -> None:
    """Raise ``AssertionError`` naming every break in the exception contract:
    a class missing from the public seam, a changed base, or two run-level
    classes that became related. Call it from the consumer's own suite."""
    import llm_utils

    problems = []
    for exc, base in EXCEPTION_CONTRACT.items():
        exported = getattr(llm_utils, exc.__name__, None)
        if exported is not exc:
            problems.append(f"llm_utils.{exc.__name__} is not exported from the seam")
        if not (inspect.isclass(exc) and issubclass(exc, base)):
            problems.append(f"{exc.__name__} no longer subclasses {base.__name__}")
    for a, b in _DISJOINT:
        if issubclass(a, b) or issubclass(b, a):
            problems.append(f"{a.__name__} and {b.__name__} became related")
    if problems:
        raise AssertionError("llm_utils exception contract broken: " + "; ".join(problems))
