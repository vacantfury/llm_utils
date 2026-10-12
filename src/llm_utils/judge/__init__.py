"""Split-question judgments: one hard judgment asked as small typed questions.

A question set (YAML) splits a judgment into about ten questions, each
answerable from the text alone; a backend (TypeSafe Jev, or any text model)
answers them with probabilities; a combiner fitted on labeled rows turns the
answers into one decision with a band (``act`` / ``drop`` / ``escalate``);
``eval_backends`` measures which backend does it best for a given set.

Import-light: nothing here imports scikit-learn or numpy until a fit or an
eval runs (the ``llm_utils[judge]`` extra); applying a saved combiner needs
neither.

    from llm_utils.judge import load_question_set, JevBackend, Combiner

    qset = load_question_set("refusal.yaml")
    jev = JevBackend.default()                      # OpenRouter, OPENROUTER_API_KEY
    combiner = Combiner.load("refusal.combiner.yaml")
    decision = combiner.decide(jev.evaluate(text, qset), qset)
    decision.band                                   # "act", "drop" or "escalate"
"""
from .backends import Backend, JevBackend, JudgeEvaluation, LLMBackend
from .combiner import Combiner, Decision
from .evaluate import EvalReport, eval_backends, fit_combiner, load_rows
from .features import feature_names, features, unknown_rate
from .questions import QuestionSet, load_question_set

__all__ = [
    "QuestionSet", "load_question_set",
    "Backend", "JevBackend", "LLMBackend", "JudgeEvaluation",
    "Combiner", "Decision",
    "eval_backends", "EvalReport", "fit_combiner", "load_rows",
    "features", "feature_names", "unknown_rate",
]
