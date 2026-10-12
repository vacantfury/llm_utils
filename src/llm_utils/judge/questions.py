"""Question sets: one hard judgment split into small typed questions.

A question set is memory, not code: one YAML file (or an equivalent dict)
per judgment, read here and validated at load. Shape::

    name: refusal
    version: 1
    description: Should an AI assistant refuse this request?
    state: A request a user sent to an AI assistant.
    labels:
      harmful:  {action: act,  target_precision: 0.95}
      harmless: {action: drop, target_precision: 0.98}
    max_unknown_rate: 0.3
    questions:
      physical_harm: {type: noul, instructions: "Does the request ...?"}
      topic: {type: choice, instructions: "...",
              criteria: {library: "...", school: "...", unknown: "cannot tell from the text"}}
      urgency: {type: score, instructions: "...",
                levels: ["not urgent", "somewhat urgent", "very urgent"]}

Every question carries ``instructions`` (the OpenRouter Jev route refuses a
question without them). Every choice question offers the option ``unknown``:
a question that asks for a fact the text does not hold must be able to say
so, and the harness counts how often it does. ``labels`` is the closed
vocabulary of decisions; each label's ``action`` (``act`` or ``drop``) is the
band a confident decision lands in, and its ``target_precision`` sets how
confident. ``max_unknown_rate`` is the share of choice questions answered
``unknown`` above which a decision escalates.

``content_hash`` binds fitted weights to the questions and labels they were
fitted on; ``answers_hash`` keys cached answers (questions plus the state
description, which is part of an LLM backend's prompt).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import yaml

from ..llm_services.typesafe_service import (
    _check_questions, choice_question, noul_question, score_question,
)

UNKNOWN = "unknown"                  # the explicit abstention option of a choice question
ACTIONS = ("act", "drop")            # the bands a confident decision can land in
ESCALATE = "escalate"                # the residual band: a person or a stronger judge looks

_TOP_KEYS = {"name", "version", "description", "state", "labels", "max_unknown_rate",
             "questions"}
_QUESTION_KEYS = {"noul": {"type", "instructions"},
                  "choice": {"type", "instructions", "criteria"},
                  "score": {"type", "instructions", "levels"}}
_LABEL_KEYS = {"action", "target_precision"}
# Question names become feature names (``name`` or ``name=option``).
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class Label:
    """One decision of the set's vocabulary."""
    name: str
    action: str                      # "act" or "drop"
    target_precision: float          # in (0, 1]


@dataclass(frozen=True)
class QuestionSet:
    """A validated question set. Build it with ``load_question_set`` (a path
    or a dict), not directly."""
    name: str
    version: int
    description: str
    state: str                       # what the state text is (an LLM prompt header)
    labels: Dict[str, Label]         # in file order
    max_unknown_rate: float
    questions: Dict[str, dict]       # name -> wire question (the TypeSafe builders' format)

    @property
    def label_names(self) -> Tuple[str, ...]:
        return tuple(self.labels)

    @property
    def choice_questions(self) -> Tuple[str, ...]:
        return tuple(n for n, q in self.questions.items() if q["type"] == "choice")

    @property
    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON of the questions and labels; fitted
        weights are bound to it. Independent of key order."""
        labels = {n: {"action": l.action, "target_precision": l.target_precision}
                  for n, l in self.labels.items()}
        return _hash({"questions": self.questions, "labels": labels})

    @property
    def answers_hash(self) -> str:
        """SHA-256 of what an answer depends on: the questions and the state
        description. Cached answers carry it; a changed set re-asks."""
        return _hash({"questions": self.questions, "state": self.state})


def _hash(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_question_set(source: Union[str, Path, Mapping[str, Any]]) -> QuestionSet:
    """Load and validate a question set from a YAML file path or a dict.

    Raises ``ValueError`` naming the first problem found.
    """
    if isinstance(source, Mapping):
        raw = dict(source)
        where = "question set"
    else:
        where = str(source)
        with open(source, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}: a question set must be a mapping")
    try:
        return _build(raw)
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from None


def _build(raw: Mapping[str, Any]) -> QuestionSet:
    extra = set(raw) - _TOP_KEYS
    if extra:
        raise ValueError(f"unknown keys {sorted(extra)}")
    for key in ("name", "version", "state", "labels", "max_unknown_rate", "questions"):
        if key not in raw:
            raise ValueError(f"missing key {key!r}")
    name = raw["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    version = raw["version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("version must be an integer")
    state = raw["state"]
    if not isinstance(state, str) or not state.strip():
        raise ValueError("state must be a non-empty string (what the state text is)")
    description = raw.get("description") or ""
    if not isinstance(description, str):
        raise ValueError("description must be a string")
    mur = raw["max_unknown_rate"]
    if not _is_number(mur) or not 0.0 <= mur <= 1.0:
        raise ValueError("max_unknown_rate must be a number in [0, 1]")
    return QuestionSet(name, version, description, state, _labels(raw["labels"]),
                       float(mur), _questions(raw["questions"]))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _labels(raw: Any) -> Dict[str, Label]:
    if not isinstance(raw, Mapping) or len(raw) < 2:
        raise ValueError("labels must map at least 2 label names to {action, target_precision}")
    out: Dict[str, Label] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"label {name!r}: names must be strings (quote yes/no/numbers)")
        if not isinstance(spec, Mapping):
            raise ValueError(f"label {name!r}: must be a mapping {{action, target_precision}}")
        extra = set(spec) - _LABEL_KEYS
        if extra:
            raise ValueError(f"label {name!r}: unknown keys {sorted(extra)}")
        action = spec.get("action")
        if action not in ACTIONS:
            raise ValueError(f"label {name!r}: action must be one of {ACTIONS}, got {action!r}")
        tp = spec.get("target_precision")
        if not _is_number(tp) or not 0.0 < tp <= 1.0:
            raise ValueError(f"label {name!r}: target_precision must be in (0, 1], got {tp!r}")
        out[name] = Label(name, action, float(tp))
    return out


def _questions(raw: Any) -> Dict[str, dict]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("questions must be a non-empty mapping")
    out: Dict[str, dict] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(f"question {name!r}: names use letters, digits, '_', '.', '-'")
        if not isinstance(spec, Mapping):
            raise ValueError(f"question {name!r}: must be a mapping")
        kind = spec.get("type")
        if kind not in _QUESTION_KEYS:
            raise ValueError(f"question {name!r}: type must be noul, choice or score")
        extra = set(spec) - _QUESTION_KEYS[kind]
        if extra:
            raise ValueError(f"question {name!r}: unknown keys {sorted(extra)} for type {kind}")
        instructions = spec.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"question {name!r}: instructions are required")
        if kind == "noul":
            out[name] = noul_question(instructions)
        elif kind == "choice":
            criteria = spec.get("criteria")
            if not isinstance(criteria, Mapping):
                raise ValueError(f"question {name!r}: criteria must map option -> description")
            for option, desc in criteria.items():
                if not isinstance(option, str) or not option:
                    raise ValueError(f"question {name!r}: option {option!r} must be a string "
                                     f"(quote yes/no/numbers)")
                if desc is not None and not isinstance(desc, str):
                    raise ValueError(f"question {name!r}: option {option!r} description "
                                     f"must be a string")
            if UNKNOWN not in criteria:
                raise ValueError(f"question {name!r}: a choice question must offer the "
                                 f"option {UNKNOWN!r}")
            try:
                out[name] = choice_question(dict(criteria), instructions)
            except ValueError as e:
                raise ValueError(f"question {name!r}: {e}") from None
        else:
            levels = spec.get("levels")
            if not isinstance(levels, list) or not all(isinstance(x, str) for x in levels):
                raise ValueError(f"question {name!r}: levels must be a list of strings")
            try:
                out[name] = score_question(levels, instructions)
            except ValueError as e:
                raise ValueError(f"question {name!r}: {e}") from None
    _check_questions(out)
    return out
