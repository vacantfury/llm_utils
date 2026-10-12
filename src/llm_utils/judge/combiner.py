"""The combiner: typed answers -> one decision with a band.

Fitting (needs the ``judge`` extra: scikit-learn) learns a logistic
regression over the feature vector (``features.py``; multinomial when the set
has more than two labels) and one probability threshold per label. Each
threshold comes from OUT-OF-FOLD probabilities (``StratifiedKFold`` +
``cross_val_predict``): it is the lowest cut at which that label's precision,
among rows whose top label is that label and whose probability clears the
cut, reaches the label's ``target_precision``. When no cut reaches it the
threshold is None: the label is never taken automatically.

Deciding is pure Python (a dot product and a softmax), so a deployment that
only applies a fitted combiner needs no extra. The top label L with
probability p lands in L's band (``act`` or ``drop``) when p clears L's
threshold and the share of choice questions answered ``unknown`` is within
the set's ``max_unknown_rate``; otherwise it escalates, and ``reasons`` says
why. A fitted combiner is saved as YAML with a ``basis`` block (what it was
fitted on, with which settings, and how well it did out of fold) and refuses
a question set whose content hash differs from the one it was fitted on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from .. import config
from ..llm_services.typesafe_service import Evaluation
from .features import feature_names, features, unknown_rate
from .questions import ESCALATE, QuestionSet

FORMAT = 1                          # version of the saved YAML layout
_SETTING_KEYS = ("cv_folds", "eval_splits", "fit_fraction", "ece_bins", "logistic_c",
                 "logistic_max_iter", "seed", "workers")


def judge_settings(**overrides: Any) -> Dict[str, Any]:
    """The ``judge`` config block with per-call overrides (None = keep)."""
    out = config.get("judge")
    for key, value in overrides.items():
        if key not in _SETTING_KEYS:
            raise ValueError(f"unknown judge setting {key!r}")
        if value is not None:
            out[key] = value
    return out


@dataclass(frozen=True)
class Decision:
    """One decision: the top label, its probability, the band it lands in
    (``act`` / ``drop`` / ``escalate``), every label's probability, the
    row's unknown rate, and why it escalated (empty when it did not)."""
    label: str
    probability: float
    band: str
    probabilities: Dict[str, float]
    unknown_rate: float
    reasons: Tuple[str, ...] = ()


class Combiner:
    """A fitted combiner. Build it with ``Combiner.fit`` or ``Combiner.load``."""

    def __init__(self, *, labels: Sequence[str], features: Sequence[str],
                 coefficients: Mapping[str, Sequence[float]], intercepts: Mapping[str, float],
                 thresholds: Mapping[str, Optional[float]], basis: Mapping[str, Any]):
        self.labels: Tuple[str, ...] = tuple(labels)
        self.features: Tuple[str, ...] = tuple(features)
        self.coefficients = {l: [float(w) for w in coefficients[l]] for l in self.labels}
        self.intercepts = {l: float(intercepts[l]) for l in self.labels}
        self.thresholds = {l: (None if thresholds.get(l) is None else float(thresholds[l]))
                           for l in self.labels}
        self.basis: Dict[str, Any] = dict(basis)
        for l in self.labels:
            if len(self.coefficients[l]) != len(self.features):
                raise ValueError(f"label {l!r}: {len(self.coefficients[l])} coefficients "
                                 f"for {len(self.features)} features")

    # -- fitting (scikit-learn) ----------------------------------------------

    @classmethod
    def fit(cls, qset: QuestionSet, evaluations: Sequence[Evaluation],
            labels: Sequence[str], *, backend: str, model: Optional[str] = None,
            **settings: Any) -> "Combiner":
        """Fit on labeled evaluations (one label per evaluation). ``settings``
        override the ``judge`` config block (``cv_folds``, ``logistic_c``,
        ``logistic_max_iter``, ``seed``)."""
        if len(evaluations) != len(labels):
            raise ValueError(f"{len(evaluations)} evaluations but {len(labels)} labels")
        names = feature_names(qset)
        rows = []
        for ev in evaluations:
            f = features(ev, qset)
            rows.append([f[n] for n in names])
        return cls._fit_matrix(qset, rows, list(labels), backend=backend, model=model,
                               settings=judge_settings(**settings))

    @classmethod
    def _fit_matrix(cls, qset: QuestionSet, rows: Sequence[Sequence[float]],
                    labels: Sequence[str], *, backend: str, model: Optional[str],
                    settings: Mapping[str, Any]) -> "Combiner":
        import numpy as np
        from sklearn.base import clone
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        from ._metrics import scores

        names = qset.label_names
        unknown = sorted(set(labels) - set(names))
        if unknown:
            raise ValueError(f"labels {unknown} are not in the set's vocabulary {list(names)}")
        counts = {l: sum(1 for x in labels if x == l) for l in names}
        absent = [l for l, n in counts.items() if n == 0]
        if absent:
            raise ValueError(f"no rows labeled {absent}: every label needs rows to fit")
        folds = min(int(settings["cv_folds"]), min(counts.values()))
        if folds < 2:
            raise ValueError("every label needs at least 2 rows for the out-of-fold thresholds")

        X = np.asarray(rows, dtype=float)
        y = np.asarray([names.index(l) for l in labels])
        lr = LogisticRegression(C=float(settings["logistic_c"]),
                                max_iter=int(settings["logistic_max_iter"]))
        lr.fit(X, y)
        coefficients, intercepts = _unpack(lr, names)

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(settings["seed"]))
        oof = cross_val_predict(clone(lr), X, y, cv=cv, method="predict_proba")
        thresholds: Dict[str, Optional[float]] = {}
        threshold_basis: Dict[str, Any] = {}
        for i, l in enumerate(names):
            target = qset.labels[l].target_precision
            thresholds[l], threshold_basis[l] = _threshold(oof, y, i, target)

        basis = {
            "question_set": {"name": qset.name, "version": qset.version,
                             "content_hash": qset.content_hash},
            "backend": {"name": backend, "model": model},
            "rows": len(labels),
            "label_counts": counts,
            "fitted": datetime.now(timezone.utc).date().isoformat(),
            "folds": folds,
            "seed": int(settings["seed"]),
            "logistic_c": float(settings["logistic_c"]),
            "logistic_max_iter": int(settings["logistic_max_iter"]),
            "out_of_fold": scores(y, oof, len(names), int(settings["ece_bins"])),
            "target_precision": {l: qset.labels[l].target_precision for l in names},
            "thresholds": threshold_basis,
        }
        return cls(labels=names, features=feature_names(qset), coefficients=coefficients,
                   intercepts=intercepts, thresholds=thresholds, basis=basis)

    # -- deciding (pure Python) ------------------------------------------------

    def _check(self, qset: QuestionSet) -> None:
        fitted = self.basis.get("question_set", {}).get("content_hash")
        if qset.content_hash != fitted:
            raise ValueError(
                f"question set {qset.name!r} v{qset.version} (hash {qset.content_hash[:12]}) "
                f"is not the one this combiner was fitted on (hash {str(fitted)[:12]}); "
                f"refit it")

    def probabilities(self, vector: Sequence[float]) -> Dict[str, float]:
        """Label probabilities for one feature vector (in ``features`` order)."""
        z = [self.intercepts[l] + sum(w * x for w, x in zip(self.coefficients[l], vector))
             for l in self.labels]
        top = max(z)
        e = [math.exp(v - top) for v in z]
        total = sum(e)
        return {l: v / total for l, v in zip(self.labels, e)}

    def decide(self, evaluation: Evaluation, qset: QuestionSet) -> Decision:
        """The decision for one evaluation. Refuses (``ValueError``) a
        question set other than the one this combiner was fitted on."""
        self._check(qset)
        f = features(evaluation, qset)
        return self._decide_vector([f[name] for name in self.features],
                                   unknown_rate(evaluation, qset), qset)

    def _decide_vector(self, vector: Sequence[float], unknown: float,
                       qset: QuestionSet) -> Decision:
        probs = self.probabilities(vector)
        label = max(self.labels, key=lambda l: probs[l])     # first label wins a tie
        p = probs[label]
        reasons: List[str] = []
        threshold = self.thresholds[label]
        if threshold is None:
            reasons.append(f"no threshold for {label!r}: out of fold its precision never "
                           f"reached {qset.labels[label].target_precision}")
        elif p < threshold:
            reasons.append(f"probability {p:.3f} below the {label!r} threshold {threshold:.3f}")
        if unknown > qset.max_unknown_rate:
            reasons.append(f"unknown rate {unknown:.2f} above the ceiling "
                           f"{qset.max_unknown_rate:.2f}")
        band = ESCALATE if reasons else qset.labels[label].action
        return Decision(label, p, band, probs, unknown, tuple(reasons))

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": FORMAT,
            "labels": list(self.labels),
            "features": list(self.features),
            "coefficients": {l: list(self.coefficients[l]) for l in self.labels},
            "intercepts": dict(self.intercepts),
            "thresholds": dict(self.thresholds),
            "basis": self.basis,
        }

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Combiner":
        if data.get("format") != FORMAT:
            raise ValueError(f"combiner format {data.get('format')!r} is not {FORMAT}")
        return cls(labels=data["labels"], features=data["features"],
                   coefficients=data["coefficients"], intercepts=data["intercepts"],
                   thresholds=data["thresholds"], basis=data.get("basis") or {})

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Combiner":
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, Mapping):
            raise ValueError(f"{path}: not a saved combiner")
        return cls.from_dict(data)


def _unpack(lr: Any, names: Sequence[str]) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """Per-label coefficients and intercepts from a fitted LogisticRegression.
    A binary fit has one row for the second label; the first label gets
    zeros, so a softmax over both equals scikit-learn's sigmoid."""
    classes = [names[int(c)] for c in lr.classes_]
    coef, intercept = lr.coef_, lr.intercept_
    if len(classes) == 2 and coef.shape[0] == 1:
        zero = [0.0] * coef.shape[1]
        return ({classes[0]: zero, classes[1]: [float(w) for w in coef[0]]},
                {classes[0]: 0.0, classes[1]: float(intercept[0])})
    return ({c: [float(w) for w in coef[k]] for k, c in enumerate(classes)},
            {c: float(intercept[k]) for k, c in enumerate(classes)})


def _threshold(oof: Any, y: Any, index: int, target: float) -> Tuple[Optional[float], Dict]:
    """Lowest cut at which label ``index`` reaches ``target`` precision among
    rows whose out-of-fold top label it is, with the basis of the answer."""
    import numpy as np

    top = oof.argmax(axis=1) == index
    p = oof[top, index]
    hit = (y[top] == index)
    if p.size == 0:
        return None, {"threshold": None, "support": 0,
                      "reason": "never the top label out of fold"}
    order = np.argsort(-p, kind="stable")
    p, hit = p[order], hit[order]
    precision = np.cumsum(hit) / np.arange(1, p.size + 1)
    # A cut admits every row with p >= cut, so only the last row of a run of
    # equal probabilities is a valid cut position.
    last_of_tie = np.append(p[1:] < p[:-1], True)
    ok = np.flatnonzero(last_of_tie & (precision >= target))
    if ok.size == 0:
        return None, {"threshold": None, "support": int(p.size),
                      "best_precision": float(precision[last_of_tie].max()),
                      "reason": f"out-of-fold precision never reached {target}"}
    i = int(ok[-1])
    return float(p[i]), {"threshold": float(p[i]), "precision": float(precision[i]),
                         "rows_taken": i + 1, "coverage": (i + 1) / int(y.size),
                         "support": int(p.size)}
