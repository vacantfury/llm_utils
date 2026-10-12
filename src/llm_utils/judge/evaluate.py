"""The eval verb: which backend answers a question set best, measured.

``eval_backends(qset, rows, backends, cache_dir=...)`` answers every labeled
row with every backend, then scores each backend's answers the way they
would be used: fit a combiner on part of the rows, decide the rest.

Answering: a thread pool per backend (its size from the ``judge.workers``
config by the backend's ``kind``, overridable per call). Every answered row
is appended as one JSON line to ``cache_dir/<backend name>.jsonl`` the
moment it arrives; a rerun skips rows already answered for the same
questions (the set's ``answers_hash``) and retries rows that failed. A
``RetentionPolicyError`` counts the row as refused, not as a crash. An
account-level failure (bad key, no credit, the run spend cap) stops that
backend's remaining rows and is reported.

Scoring: ``judge.eval_splits`` repeated stratified fit/test splits of all
rows (``judge.fit_fraction`` to fit, ``judge.seed``), the SAME splits for
every backend, each restricted to the rows that backend answered. Per
backend: accuracy, AUC, Brier score, expected calibration error, band
coverage on the test part and accuracy within the act/drop bands, unknown
rates overall and per question, each question's feature means, spreads and
mean fitted weights (a question whose answers never vary, or are mostly
unknown, shows here), latency, cost and answer counts. Per backend pair:
mean accuracy and AUC differences on the test rows both answered, and the
share of splits each wins.

``fit_combiner`` fits the deployable combiner on every cached row of one
backend. Fitting and scoring need the ``judge`` extra (scikit-learn).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..exceptions import AccountFatalError, RetentionPolicyError
from ..llm_services.typesafe_service import ChoiceAnswer, NoulAnswer, _parse_answer
from .backends import Backend, JudgeEvaluation
from .combiner import Combiner, judge_settings
from .features import feature_names, features, unknown_answers, unknown_rate
from .questions import ESCALATE, QuestionSet

_ERROR_CHARS = 300                  # stored error messages are cut to this length


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def load_rows(path: Union[str, Path]) -> List[dict]:
    """Labeled rows from a JSONL file, one ``{"id", "state", "label"}`` object
    per line (blank lines skipped)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: not JSON ({e.msg})") from None
    return rows


def _check_rows(rows: Sequence[Mapping[str, Any]], qset: QuestionSet) -> List[dict]:
    out, seen = [], set()
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping) or not {"id", "state", "label"} <= set(row):
            raise ValueError(f"row {i}: needs id, state and label")
        rid = str(row["id"])
        if rid in seen:
            raise ValueError(f"row id {rid!r} appears twice")
        seen.add(rid)
        if row["label"] not in qset.labels:
            raise ValueError(f"row {rid!r}: label {row['label']!r} is not in the set's "
                             f"vocabulary {list(qset.labels)}")
        out.append({"id": rid, "state": row["state"], "label": row["label"]})
    if not out:
        raise ValueError("no rows")
    return out


# ---------------------------------------------------------------------------
# The answer cache
# ---------------------------------------------------------------------------

def cache_path(cache_dir: Union[str, Path], backend_name: str) -> Path:
    """Where one backend's answered rows live."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", backend_name).strip("_") or "backend"
    return Path(cache_dir) / f"{safe}.jsonl"


def _cache_key(qset: QuestionSet, backend: Any) -> str:
    """What a cached answer must match to be reused: the questions as asked
    (``answers_hash``) plus the backend's own key (an LLM backend's prompt
    version), so a changed question or prompt asks again."""
    extra = getattr(backend, "cache_key", "")
    return f"{qset.answers_hash}:{extra}" if extra else qset.answers_hash


def _state_hash(state: Any) -> str:
    """A row's text, hashed: new text under an old id asks again."""
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True,
                                                           ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _record(rid: str, qset: QuestionSet, ev: Any) -> dict:
    answers = {}
    for name, a in ev.answers.items():
        if isinstance(a, NoulAnswer):
            answers[name] = {"type": "noul", "noul": a.noul}
        elif isinstance(a, ChoiceAnswer):
            answers[name] = {"type": "choice", "choice": a.choice, "confidence": a.confidence,
                             "probabilities": dict(a.probabilities)}
        else:
            answers[name] = {"type": "score", "score": a.score, "confidence": a.confidence,
                             "probabilities": dict(a.probabilities), "legend": dict(a.legend)}
    return {"id": rid, "status": "ok", "answers_hash": qset.answers_hash, "model": ev.model,
            "answers": answers, "input_tokens": ev.input_tokens,
            "output_tokens": ev.output_tokens, "cost_usd": ev.cost_usd,
            "cost_known": bool(getattr(ev, "cost_known", True)), "latency_s": ev.latency_s}


def _evaluation(rec: Mapping[str, Any]) -> JudgeEvaluation:
    answers = {name: _parse_answer(name, raw) for name, raw in rec["answers"].items()}
    return JudgeEvaluation(rec.get("model", ""), answers, int(rec.get("input_tokens") or 0),
                           int(rec.get("output_tokens") or 0), float(rec.get("cost_usd") or 0.0),
                           float(rec.get("latency_s") or 0.0),
                           cost_known=bool(rec.get("cost_known", True)))


def read_cache(cache_dir: Union[str, Path], backend_name: str,
               qset: QuestionSet, *, key: Optional[str] = None) -> Dict[str, dict]:
    """The latest record per row id for this question set's answers (other
    sets' records and unreadable lines are ignored). ``key`` (the answering
    backend's cache key) narrows it to answers asked exactly that way."""
    path = cache_path(cache_dir, backend_name)
    out: Dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                       # a line cut short by a crash
            if not isinstance(rec, dict):
                continue
            stored = str(rec.get("answers_hash"))
            if (stored == key if key is not None else
                    stored == qset.answers_hash or stored.startswith(qset.answers_hash + ":")):
                out[str(rec.get("id"))] = rec
    return out


def _end_partial_line(path: Path) -> None:
    """A crash can leave the last line cut short; end it, so the next record
    starts a line of its own instead of being lost with it."""
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as fh:
            fh.seek(-1, 2)
            last = fh.read(1)
        if last != b"\n":
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n")


def _workers(backend: Any, override: Any, settings: Mapping[str, Any]) -> int:
    if isinstance(override, Mapping) and backend.name in override:
        return int(override[backend.name])
    if isinstance(override, int) and not isinstance(override, bool):
        return override
    table = settings["workers"]
    return int(table.get(getattr(backend, "kind", "llm"), table["llm"]))


def _answer_backend(qset: QuestionSet, rows: List[dict], backend: Backend,
                    cache_dir: Path, personal_data: bool, workers: int) -> Dict[str, Any]:
    key = _cache_key(qset, backend)
    cached = read_cache(cache_dir, backend.name, qset, key=key)
    # Only records of these rows, asked about their current text, count.
    cached = {r["id"]: cached[r["id"]] for r in rows if r["id"] in cached
              and cached[r["id"]].get("state_hash") == _state_hash(r["state"])}
    todo = [r for r in rows if cached.get(r["id"], {}).get("status") != "ok"]
    path = cache_path(cache_dir, backend.name)
    aborted: Optional[str] = None
    calls = 0

    def one(row: dict) -> dict:
        try:
            ev = backend.evaluate(row["state"], qset, personal_data=personal_data)
            features(ev, qset)                 # a missing or mistyped answer fails the row
            return {**_record(row["id"], qset, ev), "answers_hash": key,
                    "state_hash": _state_hash(row["state"])}
        except RetentionPolicyError as e:
            status, err = "refused", e
        except AccountFatalError:
            raise
        except Exception as e:  # noqa: BLE001 (one row's failure is recorded, then retried)
            status, err = "error", e
        return {"id": row["id"], "status": status, "answers_hash": key,
                "state_hash": _state_hash(row["state"]),
                "error_class": type(err).__name__, "error": str(err)[:_ERROR_CHARS]}

    if todo:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _end_partial_line(path)
        with open(path, "a", encoding="utf-8") as fh, \
                ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(one, r): r for r in todo}
            for fut in as_completed(futures):
                if fut.cancelled():
                    continue
                calls += 1
                try:
                    rec = fut.result()
                except AccountFatalError as e:
                    if aborted is None:
                        aborted = f"{type(e).__name__}: {str(e)[:_ERROR_CHARS]}"
                        for other in futures:
                            other.cancel()
                    continue
                # Results are written from this thread only, one line each.
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                cached[rec["id"]] = rec
    return {"records": cached, "calls": calls, "aborted": aborted}


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _clean(obj: Any) -> Any:
    """NaN and infinities become None so the report is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


@dataclass
class EvalReport:
    """What ``eval_backends`` measured. ``backends`` maps each backend name
    to its scores; ``pairs`` holds the paired comparisons."""
    question_set: Dict[str, Any]
    rows: int
    label_counts: Dict[str, int]
    settings: Dict[str, Any]
    backends: Dict[str, Dict[str, Any]]
    pairs: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return _clean({"question_set": self.question_set, "rows": self.rows,
                       "label_counts": self.label_counts, "settings": self.settings,
                       "backends": self.backends, "pairs": self.pairs})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        qs = self.question_set
        lines = [f"**{qs['name']}** v{qs['version']}: {self.rows} rows "
                 f"({', '.join(f'{k} {v}' for k, v in self.label_counts.items())}), "
                 f"{self.settings['eval_splits']} splits fitting on "
                 f"{self.settings['fit_fraction']:.0%}", "",
                 "| backend | answered | errors | refused | accuracy | AUC | Brier | ECE "
                 "| act / drop / escalate | in-band accuracy | unknown | latency p50 / p95 "
                 "| cost |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, b in self.backends.items():
            m, bands = b.get("metrics") or {}, b.get("bands") or {}
            lat = b["latency_s"]
            lines.append(
                f"| {name} | {b['answered']} | {b['errors']} | {b['refused']} "
                f"| {_fmt(m.get('accuracy'))} | {_fmt(m.get('auc'))} | {_fmt(m.get('brier'))} "
                f"| {_fmt(m.get('ece'))} "
                f"| {_pct(bands.get('act'))} / {_pct(bands.get('drop'))} / "
                f"{_pct(bands.get('escalate'))} | {_fmt(bands.get('accuracy_in_band'))} "
                f"| {_pct(b['unknown_rate'])} | {_fmt(lat.get('median'), 2)} / "
                f"{_fmt(lat.get('p95'), 2)} s | {b['cost']['text']} |")
        if self.pairs:
            lines += ["", "| pair | accuracy difference | AUC difference "
                      "| first wins on AUC | second wins on AUC |", "|---|---|---|---|---|"]
            for p in self.pairs:
                lines.append(
                    f"| {p['first']} vs {p['second']} | {_fmt(p['accuracy_diff'], 3, True)} "
                    f"| {_fmt(p['auc_diff'], 3, True)} | {_pct(p['first_wins_auc'])} "
                    f"| {_pct(p['second_wins_auc'])} |")
        flags = [f"{name}: {flag}" for name, b in self.backends.items() for flag in b["flags"]]
        notes = [f"{name}: {b['aborted']}" for name, b in self.backends.items() if b["aborted"]]
        if flags or notes:
            lines += [""] + [f"- {x}" for x in notes + flags]
        return "\n".join(lines)


def _fmt(v: Any, digits: int = 3, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _pct(v: Any) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.0%}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _answered(qset: QuestionSet, records: Mapping[str, dict], rows: List[dict]) -> Dict[str, Any]:
    """Feature vectors, unknown flags and latencies of the rows answered."""
    names = feature_names(qset)
    vec, unk, flags, latency, costs = {}, {}, {}, [], []
    for r in rows:
        rec = records.get(r["id"])
        if not rec or rec.get("status") != "ok":
            continue
        ev = _evaluation(rec)
        f = features(ev, qset)
        vec[r["id"]] = [f[n] for n in names]
        unk[r["id"]] = unknown_rate(ev, qset)
        flags[r["id"]] = unknown_answers(ev, qset)
        latency.append(ev.latency_s)
        costs.append(ev.cost_usd if ev.cost_known else None)
    return {"vectors": vec, "unknown": unk, "flags": flags, "latency": latency, "costs": costs}


def _cost(costs: List[Optional[float]]) -> Dict[str, Any]:
    known = [c for c in costs if c is not None]
    unknown = len(costs) - len(known)
    if not costs:
        return {"usd": None, "unknown_rows": 0, "text": "n/a (no rows answered)"}
    if not known:
        return {"usd": None, "unknown_rows": unknown, "text": "unknown (no cost reported)"}
    text = f"${sum(known):.4f}"
    if unknown:
        text += f" (+{unknown} rows of unknown cost)"
    return {"usd": sum(known), "unknown_rows": unknown, "text": text}


def _mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    import numpy as np
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(finite)), "std": float(np.std(finite))}


def _score_backends(qset: QuestionSet, rows: List[dict], answered: Dict[str, Dict[str, Any]],
                    settings: Mapping[str, Any]) -> Dict[str, Any]:
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    from ._metrics import scores

    labels = qset.label_names
    y_all = np.asarray([labels.index(r["label"]) for r in rows])
    ids = [r["id"] for r in rows]
    label_of = {r["id"]: labels.index(r["label"]) for r in rows}
    splitter = StratifiedShuffleSplit(n_splits=int(settings["eval_splits"]),
                                      train_size=float(settings["fit_fraction"]),
                                      random_state=int(settings["seed"]))
    try:
        splits = list(splitter.split(np.zeros(len(ids)), y_all))
    except ValueError as e:
        raise ValueError(f"cannot split {len(ids)} rows by label: {e}") from None

    per_split: Dict[str, List[Optional[Dict[str, Any]]]] = {b: [] for b in answered}
    for fit_idx, test_idx in splits:
        for b, data in answered.items():
            vec = data["vectors"]
            fit_ids = [ids[i] for i in fit_idx if ids[i] in vec]
            test_ids = [ids[i] for i in test_idx if ids[i] in vec]
            try:
                comb = Combiner._fit_matrix(
                    qset, [vec[i] for i in fit_ids],
                    [labels[label_of[i]] for i in fit_ids],
                    backend=b, model=None, settings=settings)
            except ValueError:
                per_split[b].append(None)      # too few rows of a label in this split
                continue
            if not test_ids:
                per_split[b].append(None)
                continue
            decisions = [comb._decide_vector(vec[i], data["unknown"][i], qset) for i in test_ids]
            P = np.asarray([[d.probabilities[l] for l in labels] for d in decisions])
            y = np.asarray([label_of[i] for i in test_ids])
            bands = Counter(d.band for d in decisions)
            auto = [(labels.index(d.label) == label_of[i])
                    for d, i in zip(decisions, test_ids) if d.band != ESCALATE]
            per_split[b].append({
                "test_ids": test_ids, "P": P, "y": y,
                "metrics": scores(y, P, len(labels), int(settings["ece_bins"])),
                "bands": {k: bands.get(k, 0) / len(decisions) for k in ("act", "drop", ESCALATE)},
                "accuracy_in_band": (sum(auto) / len(auto)) if auto else float("nan"),
                "weights": {l: comb.coefficients[l] for l in labels},
            })
    return {"splits": per_split}


def _backend_summary(qset: QuestionSet, b: str, data: Dict[str, Any], answer: Dict[str, Any],
                     splits: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    import numpy as np

    labels = qset.label_names
    names = feature_names(qset)
    records = answer["records"]
    status = Counter(records[r["id"]].get("status") for r in answer["rows"] if r["id"] in records)
    # The labels of rows without an answer: a backend whose errors or refusals
    # fall on one label is scored on a skewed test set (compare pairs instead).
    unanswered = Counter(r["label"] for r in answer["rows"]
                         if records.get(r["id"], {}).get("status") != "ok")
    scored = [s for s in splits if s is not None]
    metrics = None
    bands = None
    if scored:
        metrics = {k: _mean_std([s["metrics"][k] for s in scored])["mean"]
                   for k in scored[0]["metrics"]}
        metrics["std"] = {k: _mean_std([s["metrics"][k] for s in scored])["std"]
                          for k in ("accuracy", "auc")}
        bands = {k: float(np.mean([s["bands"][k] for s in scored]))
                 for k in ("act", "drop", ESCALATE)}
        bands["accuracy_in_band"] = _mean_std([s["accuracy_in_band"] for s in scored])["mean"]

    # Per question: unknown share, and each feature's spread and mean weight.
    vec = data["vectors"]
    X = np.asarray(list(vec.values()), dtype=float) if vec else None
    # A two-label fit has one weight row (the second label's); the first is zeros.
    weight_labels = labels[1:] if len(labels) == 2 else labels
    questions: Dict[str, Any] = {}
    flags: List[str] = []
    for qname, q in qset.questions.items():
        entry: Dict[str, Any] = {"type": q["type"]}
        if q["type"] == "choice":
            share = (sum(f[qname] for f in data["flags"].values()) / len(data["flags"])
                     if data["flags"] else None)
            entry["unknown_rate"] = share
            if share is not None and share > qset.max_unknown_rate:
                flags.append(f"{qname} answered unknown in {share:.0%} of rows")
        feats = {}
        for j, fname in enumerate(names):
            if fname != qname and not fname.startswith(f"{qname}="):
                continue
            col = X[:, j] if X is not None else None
            feats[fname] = {
                "mean": float(col.mean()) if col is not None else None,
                "std": float(col.std()) if col is not None else None,
                "constant": bool(col.max() == col.min()) if col is not None else None,
                "weight": {l: float(np.mean([s["weights"][l][j] for s in scored]))
                           for l in weight_labels} if scored else None,
            }
        entry["features"] = feats
        if X is not None and all(v["constant"] for v in feats.values()):
            flags.append(f"{qname} never varies across rows")
        questions[qname] = entry

    if sum(unanswered.values()):
        flags.append("unanswered rows by label " + ", ".join(
            f"{l} {unanswered.get(l, 0)}" for l in labels)
            + ": this backend is scored on the rows it answered; compare pairs")
    latency = data["latency"]
    return {
        "answered": status.get("ok", 0),
        "errors": status.get("error", 0),
        "refused": status.get("refused", 0),
        "missing": len(answer["rows"]) - sum(status.values()),
        "unanswered_by_label": {l: unanswered.get(l, 0) for l in labels},
        "new_calls": answer["calls"],
        "aborted": answer["aborted"],
        "splits_scored": len(scored),
        "metrics": metrics,
        "bands": bands,
        "unknown_rate": (float(np.mean(list(data["unknown"].values())))
                         if data["unknown"] else None),
        "over_unknown_ceiling": (float(np.mean([u > qset.max_unknown_rate
                                                for u in data["unknown"].values()]))
                                 if data["unknown"] else None),
        "questions": questions,
        "latency_s": ({"median": float(np.median(latency)),
                       "p95": float(np.percentile(latency, 95))} if latency else {}),
        "cost": _cost(data["costs"]),
        "flags": flags,
    }


def _pair(first: str, second: str, splits_a: List[Optional[dict]],
          splits_b: List[Optional[dict]], n_labels: int) -> Dict[str, Any]:
    import numpy as np

    from ._metrics import auc

    acc_d, auc_d = [], []
    wins = {"first_acc": 0, "second_acc": 0, "first_auc": 0, "second_auc": 0}
    compared_acc = compared_auc = 0
    for a, b in zip(splits_a, splits_b):
        if a is None or b is None:
            continue
        common = sorted(set(a["test_ids"]) & set(b["test_ids"]))
        if not common:
            continue
        ia = {r: k for k, r in enumerate(a["test_ids"])}
        ib = {r: k for k, r in enumerate(b["test_ids"])}
        Pa = a["P"][[ia[r] for r in common]]
        Pb = b["P"][[ib[r] for r in common]]
        y = a["y"][[ia[r] for r in common]]
        da = float((Pa.argmax(axis=1) == y).mean()) - float((Pb.argmax(axis=1) == y).mean())
        acc_d.append(da)
        compared_acc += 1
        wins["first_acc"] += da > 0
        wins["second_acc"] += da < 0
        ua, ub = auc(y, Pa, n_labels), auc(y, Pb, n_labels)
        if math.isfinite(ua) and math.isfinite(ub):
            auc_d.append(ua - ub)
            compared_auc += 1
            wins["first_auc"] += ua > ub
            wins["second_auc"] += ub > ua
    share = lambda k, n: (wins[k] / n) if n else None  # noqa: E731
    return {
        "first": first, "second": second, "splits": compared_acc,
        "accuracy_diff": float(np.mean(acc_d)) if acc_d else None,
        "auc_diff": float(np.mean(auc_d)) if auc_d else None,
        "first_wins_accuracy": share("first_acc", compared_acc),
        "second_wins_accuracy": share("second_acc", compared_acc),
        "first_wins_auc": share("first_auc", compared_auc),
        "second_wins_auc": share("second_auc", compared_auc),
    }


# ---------------------------------------------------------------------------
# The verbs
# ---------------------------------------------------------------------------

def eval_backends(qset: QuestionSet, rows: Sequence[Mapping[str, Any]],
                  backends: Sequence[Backend], *, cache_dir: Union[str, Path],
                  personal_data: bool = False,
                  workers: Union[None, int, Mapping[str, int]] = None,
                  eval_splits: Optional[int] = None, fit_fraction: Optional[float] = None,
                  seed: Optional[int] = None) -> EvalReport:
    """Answer ``rows`` (``{"id", "state", "label"}``) with every backend
    (cached in ``cache_dir``), then score each backend and every pair.

    ``workers`` overrides the concurrency: one number for every backend, or a
    mapping by backend name. ``eval_splits``, ``fit_fraction`` and ``seed``
    override the ``judge`` config for this call. ``personal_data`` is passed
    to every backend (Jev refuses it; LLM backends leave it to the caller).
    """
    checked = _check_rows(rows, qset)
    names = [b.name for b in backends]
    if len(set(names)) != len(names) or not names:
        raise ValueError(f"backend names must be present and unique, got {names}")
    settings = judge_settings(eval_splits=eval_splits, fit_fraction=fit_fraction, seed=seed)
    cache = Path(cache_dir)
    answers, answered = {}, {}
    for backend in backends:
        a = _answer_backend(qset, checked, backend, cache, personal_data,
                            _workers(backend, workers, settings))
        a["rows"] = checked
        answers[backend.name] = a
        answered[backend.name] = _answered(qset, a["records"], checked)
    scored = _score_backends(qset, checked, answered, settings)["splits"]
    report_backends = {b: _backend_summary(qset, b, answered[b], answers[b], scored[b])
                       for b in names}
    pairs = [_pair(a, b, scored[a], scored[b], len(qset.labels))
             for a, b in itertools.combinations(names, 2)]
    counts = Counter(r["label"] for r in checked)
    return EvalReport(
        question_set={"name": qset.name, "version": qset.version,
                      "content_hash": qset.content_hash},
        rows=len(checked),
        label_counts={l: counts.get(l, 0) for l in qset.label_names},
        settings={k: settings[k] for k in ("eval_splits", "fit_fraction", "seed", "cv_folds",
                                           "logistic_c", "logistic_max_iter", "ece_bins")},
        backends=report_backends, pairs=pairs)


def fit_combiner(qset: QuestionSet, rows: Sequence[Mapping[str, Any]], backend_name: str,
                 cache_dir: Union[str, Path], **settings: Any) -> Combiner:
    """The deployable combiner: fitted on every cached answered row of one
    backend (run ``eval_backends`` first to fill the cache). ``settings``
    override the ``judge`` config block."""
    checked = _check_rows(rows, qset)
    records = read_cache(cache_dir, backend_name, qset)
    evs, labels, models = [], [], Counter()
    for r in checked:
        rec = records.get(r["id"])
        if (rec and rec.get("status") == "ok"
                and rec.get("state_hash") == _state_hash(r["state"])):
            evs.append(_evaluation(rec))
            labels.append(r["label"])
            models[rec.get("model")] += 1
    if not evs:
        raise ValueError(f"no answered rows cached for backend {backend_name!r} in {cache_dir}")
    model = models.most_common(1)[0][0]
    return Combiner.fit(qset, evs, labels, backend=backend_name, model=model, **settings)
