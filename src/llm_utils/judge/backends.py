"""Backends: whatever answers a question set about one state.

A backend has a ``name`` and ``evaluate(state, qset, *, personal_data=False)
-> Evaluation`` (the TypeSafe ``Evaluation``: answers keyed by question name
as ``NoulAnswer`` / ``ChoiceAnswer`` / ``ScoreAnswer``). An optional ``kind``
(``"jev"`` or ``"llm"``) picks its default concurrency in the eval. A backend
that cannot tell what a call cost returns an ``Evaluation`` whose
``cost_known`` attribute is False (``JudgeEvaluation``); the eval then reports
the cost as unknown instead of zero.

- ``JevBackend`` asks TypeSafe Jev (``TypeSafeService.evaluate``). Personal
  data is refused on both Jev routes (``RetentionPolicyError``), and the
  backend passes ``personal_data`` through so the refusal holds.
- ``LLMBackend`` asks any text model the same questions in ONE prompt and
  reads back probabilities: yes for a noul question, one per option for a
  choice question, one per level for a score question. With a service that
  implements ``chat_structured`` it uses structured output; otherwise it
  calls a caller-supplied ``complete(prompt) -> str`` (or ``-> (str,
  cost_usd)``) and reads the first JSON object of the reply. That is how a
  consumer runs a model through its own runner: this package never imports a
  runner. Personal data is the caller's policy for LLM backends: the
  ``personal_data`` flag is accepted and not acted on here.
"""
from __future__ import annotations

import inspect
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, Mapping, Optional, Protocol, Tuple, Union, runtime_checkable,
)

from ..llm_services.typesafe_service import (
    Answer, ChoiceAnswer, Evaluation, JSONContent, NoulAnswer, ScoreAnswer,
)
from .questions import QuestionSet

CompleteFn = Callable[[str], Union[str, Tuple[str, Optional[float]]]]


@dataclass(frozen=True)
class JudgeEvaluation(Evaluation):
    """An ``Evaluation`` that also says whether ``cost_usd`` is the real cost
    (False: unknown, stored as 0.0)."""
    cost_known: bool = True


@runtime_checkable
class Backend(Protocol):
    """What the eval and ``Combiner`` need from a backend."""
    name: str

    def evaluate(self, state: JSONContent, qset: QuestionSet, *,
                 personal_data: bool = False) -> Evaluation: ...


# ---------------------------------------------------------------------------
# Jev
# ---------------------------------------------------------------------------

class JevBackend:
    """TypeSafe Jev through a ``TypeSafeService`` (either route)."""

    kind = "jev"

    def __init__(self, service: Any, *, name: Optional[str] = None):
        self.service = service
        self.name = name or service.model.model_id

    @classmethod
    def default(cls, *, name: Optional[str] = None, **service_kwargs) -> "JevBackend":
        """Jev 1.13 through OpenRouter (``LLMModel.OR_JEV_1_13``,
        ``OPENROUTER_API_KEY``), built by ``LLMServiceFactory``."""
        from ..llm_model import LLMModel
        from ..llm_service_factory import LLMServiceFactory
        return cls(LLMServiceFactory.create(LLMModel.OR_JEV_1_13, **service_kwargs), name=name)

    def evaluate(self, state: JSONContent, qset: QuestionSet, *,
                 personal_data: bool = False) -> Evaluation:
        return self.service.evaluate(state, qset.questions, personal_data=personal_data)


# ---------------------------------------------------------------------------
# Any text model
# ---------------------------------------------------------------------------

def render_prompt(state: JSONContent, qset: QuestionSet) -> str:
    """The one prompt an LLM backend sends for one state."""
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    lines = [
        f"The text below is: {qset.state.strip()}",
        "",
        "<text>",
        text,
        "</text>",
        "",
        "Answer each question from the text alone. Give probabilities, numbers "
        "from 0 to 1. When the text does not say, choose the option \"unknown\" "
        "where a question offers it.",
        "",
        "Questions:",
    ]
    example: Dict[str, Any] = {}
    for name, q in qset.questions.items():
        kind = q["type"]
        if kind == "noul":
            lines.append(f"- {name} (yes or no): {q['instructions']}")
            lines.append("  Give the probability that the answer is yes.")
            example[name] = 0.5
        elif kind == "choice":
            lines.append(f"- {name} (pick one option): {q['instructions']}")
            for option, desc in q["criteria"].items():
                lines.append(f"  - {option}" + (f": {desc}" if desc else ""))
            lines.append("  Give a probability for each option; they sum to 1.")
            example[name] = {option: 0.0 for option in q["criteria"]}
        else:
            lines.append(f"- {name} (rate on ordered levels): {q['instructions']}")
            for i, level in enumerate(q["criteria"]):
                lines.append(f"  - {i}: {level}")
            lines.append("  Give a probability for each level number; they sum to 1.")
            example[name] = {str(i): 0.0 for i in range(len(q["criteria"]))}
    lines += ["", "Reply with only a JSON object of this shape (your numbers in place "
              "of the placeholders):", json.dumps(example, ensure_ascii=False)]
    return "\n".join(lines)


def first_json_object(text: str) -> dict:
    """The first JSON object in ``text`` (models often wrap it in prose or a
    code fence). Raises ``ValueError`` when there is none."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    raise ValueError("no JSON object in the reply")


def _prob(name: str, value: Any) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"question {name!r}: {value!r} is not a probability") from None
    if math.isnan(p):
        raise ValueError(f"question {name!r}: probability is NaN")
    return min(1.0, max(0.0, p))


def _distribution(name: str, value: Any, keys: Tuple[str, ...],
                  aliases: Mapping[str, str]) -> Dict[str, float]:
    """Clamp and renormalize one per-option (or per-level) distribution."""
    if not isinstance(value, Mapping):
        raise ValueError(f"question {name!r}: expected an object of probabilities")
    raw: Dict[str, float] = {k: 0.0 for k in keys}
    for key, p in value.items():
        k = aliases.get(str(key), str(key))
        if k in raw:
            raw[k] = _prob(name, p)
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError(f"question {name!r}: all probabilities are zero")
    return {k: p / total for k, p in raw.items()}


def answers_from_reply(reply: Mapping[str, Any], qset: QuestionSet) -> Dict[str, Answer]:
    """Typed answers from a reply object ``{question: probability or
    {option or level: probability}}``. A missing question raises
    ``ValueError``. Choice answers take the most likely option and its
    probability as confidence; score answers take the expected level index
    (the scale Jev reports) and the top level's probability."""
    out: Dict[str, Answer] = {}
    for name, q in qset.questions.items():
        if name not in reply:
            raise ValueError(f"the reply has no answer to question {name!r}")
        value = reply[name]
        kind = q["type"]
        if kind == "noul":
            out[name] = NoulAnswer(_prob(name, value))
        elif kind == "choice":
            options = tuple(q["criteria"])
            probs = _distribution(name, value, options, {})
            choice = max(options, key=lambda o: probs[o])     # first option wins a tie
            out[name] = ChoiceAnswer(choice, probs[choice], probs)
        else:
            levels = q["criteria"]
            keys = tuple(str(i) for i in range(len(levels)))
            aliases = {str(level): str(i) for i, level in enumerate(levels)}
            probs = _distribution(name, value, keys, aliases)
            score = sum(i * probs[k] for i, k in enumerate(keys))
            legend = {str(i): level for i, level in enumerate(levels)}
            out[name] = ScoreAnswer(score, max(probs.values()), probs, legend)
    return out


def _structured_schema(qset: QuestionSet):
    """A pydantic model for structured output: one field per question, keyed
    by alias so any question, option or level name works as a JSON key."""
    from pydantic import ConfigDict, Field, create_model

    config = ConfigDict(extra="forbid")
    fields: Dict[str, Any] = {}
    for i, (name, q) in enumerate(qset.questions.items()):
        kind = q["type"]
        if kind == "noul":
            annotation: Any = float
        else:
            keys = (list(q["criteria"]) if kind == "choice"
                    else [str(j) for j in range(len(q["criteria"]))])
            annotation = create_model(
                f"Q{i}", __config__=config,
                **{f"k{j}": (float, Field(alias=key)) for j, key in enumerate(keys)})
        fields[f"q{i}"] = (annotation, Field(alias=name, description=q["instructions"]))
    return create_model("JudgeAnswers", __config__=config, **fields)


class _UsageTap:
    """Per-thread tokens and cost a service records during one call.

    Services record usage in the calling thread, at their one choke point
    ``_record_usage``; the tap wraps that method on this service instance
    (recording still happens as before) so concurrent rows each see their
    own cost. A call that recorded nothing reads as unknown cost."""

    def __init__(self, service: Any):
        self._local = threading.local()
        original = service._record_usage

        def tapped(input_tokens, output_tokens, cost, is_test):
            original(input_tokens, output_tokens, cost, is_test)
            acc = getattr(self._local, "acc", None)
            if acc is not None:
                acc[0] += input_tokens
                acc[1] += output_tokens
                acc[2] += cost
                acc[3] += 1

        service._record_usage = tapped

    def start(self) -> None:
        self._local.acc = [0, 0, 0.0, 0]

    def stop(self) -> Tuple[int, int, Optional[float]]:
        acc = self._local.acc
        self._local.acc = None
        return acc[0], acc[1], (acc[2] if acc[3] else None)


class LLMBackend:
    """Any text model answering a question set in one prompt per state."""

    kind = "llm"

    def __init__(self, complete: Optional[CompleteFn] = None, *, name: str,
                 model: Optional[str] = None, service: Any = None):
        if (complete is None) == (service is None):
            raise ValueError("give exactly one of complete= or service= "
                             "(LLMBackend.from_service builds the second)")
        self.name = name
        self.model = model or (service.model.model_id if service is not None else name)
        self._complete = complete
        self._service = service
        self._structured = False
        self._tap: Optional[_UsageTap] = None
        self._schemas: Dict[str, Any] = {}
        self._schema_lock = threading.Lock()
        if service is not None:
            from ..base_llm_service import BaseLLMService
            # The base method only raises NotImplementedError; services that
            # implement structured output override it (spend-guard wrapped).
            impl = getattr(type(service), "chat_structured", None)
            self._structured = (impl is not None and inspect.unwrap(impl)
                                is not BaseLLMService.chat_structured)
            if hasattr(service, "_record_usage"):
                self._tap = _UsageTap(service)

    @classmethod
    def from_service(cls, service: Any, *, name: Optional[str] = None) -> "LLMBackend":
        """Wrap an llm_utils service: structured output when the service
        implements ``chat_structured`` (OpenAI-family and Claude services),
        else ``service.chat`` and the JSON reader."""
        return cls(name=name or service.model.model_id, service=service)

    def _schema(self, qset: QuestionSet):
        with self._schema_lock:
            key = qset.answers_hash
            if key not in self._schemas:
                self._schemas[key] = _structured_schema(qset)
            return self._schemas[key]

    def _ask_service(self, prompt: str, qset: QuestionSet) -> Mapping[str, Any]:
        if self._structured:
            parsed = self._service.chat_structured(prompt, self._schema(qset))
            if parsed is None:
                raise ValueError("structured output failed validation (no parsed answer)")
            return parsed.model_dump(by_alias=True)
        from ..base_llm_service import is_mechanism_error, strip_mechanism_error
        reply = self._service.chat(prompt)
        if is_mechanism_error(reply):
            raise RuntimeError(f"{self.name}: {strip_mechanism_error(reply)}")
        return first_json_object(reply)

    def evaluate(self, state: JSONContent, qset: QuestionSet, *,
                 personal_data: bool = False) -> Evaluation:
        prompt = render_prompt(state, qset)
        started = time.monotonic()
        in_tok = out_tok = 0
        cost: Optional[float] = None
        if self._service is not None:
            if self._tap is not None:
                self._tap.start()
            try:
                reply = self._ask_service(prompt, qset)
            finally:
                if self._tap is not None:
                    in_tok, out_tok, cost = self._tap.stop()
        else:
            out = self._complete(prompt)
            text, cost = out if isinstance(out, tuple) else (out, None)
            reply = first_json_object(text)
        answers = answers_from_reply(reply, qset)
        return JudgeEvaluation(self.model, answers, in_tok, out_tok,
                               0.0 if cost is None else float(cost),
                               time.monotonic() - started, cost_known=cost is not None)
