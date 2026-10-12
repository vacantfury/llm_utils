"""One deterministic mapping from typed answers to a named feature vector.

The same for every backend, so a combiner fitted on one backend's answers is
read the same way at decision time, and backends compare on equal terms:

- noul ``q`` -> feature ``q`` = probability of yes;
- choice ``q`` -> feature ``q=<option>`` = that option's probability, for each
  option, ``q=unknown`` included (the abstention is a signal, not noise);
- score ``q`` -> feature ``q`` = expected level / (levels - 1), in [0, 1].

``unknown_rate`` is the share of the set's choice questions whose most likely
option is ``unknown`` (0 when the set has none). Pure Python, no numpy.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Tuple

from ..llm_services.typesafe_service import (
    Answer, ChoiceAnswer, Evaluation, NoulAnswer, ScoreAnswer,
)
from .questions import UNKNOWN, QuestionSet


def feature_names(qset: QuestionSet) -> Tuple[str, ...]:
    """Feature names in a fixed order: questions in set order, options in
    criteria order."""
    names: List[str] = []
    for name, q in qset.questions.items():
        if q["type"] == "choice":
            names.extend(f"{name}={option}" for option in q["criteria"])
        else:
            names.append(name)
    return tuple(names)


def _answer(answers: Mapping[str, Answer], name: str, kind: type) -> Answer:
    if name not in answers:
        raise ValueError(f"no answer to question {name!r}")
    a = answers[name]
    if not isinstance(a, kind):
        raise ValueError(f"question {name!r}: expected a {kind.__name__}, "
                         f"got {type(a).__name__}")
    return a


def features(evaluation: Evaluation, qset: QuestionSet) -> Dict[str, float]:
    """The named feature vector of one evaluation (keys in ``feature_names``
    order). Raises ``ValueError`` when an answer is missing or mistyped."""
    answers = evaluation.answers
    out: Dict[str, float] = {}
    for name, q in qset.questions.items():
        kind = q["type"]
        if kind == "noul":
            out[name] = float(_answer(answers, name, NoulAnswer).noul)
        elif kind == "choice":
            a = _answer(answers, name, ChoiceAnswer)
            for option in q["criteria"]:
                out[f"{name}={option}"] = float(a.probabilities.get(option, 0.0))
        else:
            a = _answer(answers, name, ScoreAnswer)
            top = len(q["criteria"]) - 1
            out[name] = float(a.score) / top
    return out


def unknown_answers(evaluation: Evaluation, qset: QuestionSet) -> Dict[str, bool]:
    """Per choice question: whether its most likely option is ``unknown``."""
    out: Dict[str, bool] = {}
    for name in qset.choice_questions:
        a = _answer(evaluation.answers, name, ChoiceAnswer)
        out[name] = _argmax(a.probabilities) == UNKNOWN if a.probabilities else a.choice == UNKNOWN
    return out


def unknown_rate(evaluation: Evaluation, qset: QuestionSet) -> float:
    """Share of the set's choice questions answered ``unknown``."""
    flags = unknown_answers(evaluation, qset)
    return sum(flags.values()) / len(flags) if flags else 0.0


def _argmax(probabilities: Mapping[str, float]) -> str:
    # Ties go to the first option in the answer's order.
    best, best_p = None, None
    for option, p in probabilities.items():
        if best_p is None or p > best_p:
            best, best_p = option, p
    return best
