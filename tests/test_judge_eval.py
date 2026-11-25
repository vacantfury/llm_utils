"""Guards cached judge evaluations, backend comparisons and deployable combiners.

Backends are offline fakes; fitting requires scikit-learn."""

import json
import random
import threading

import pytest

from llm_utils import (
    ChoiceAnswer, CreditsExhaustedError, Evaluation, NoulAnswer, RetentionPolicyError,
    ScoreAnswer,
)
from llm_utils.judge import (
    Combiner, EvalReport, JudgeEvaluation, eval_backends, fit_combiner, load_question_set,
    load_rows,
)
from llm_utils.judge.evaluate import cache_path

pytest.importorskip("sklearn")

QSET = load_question_set({
    "name": "refusal",
    "version": 1,
    "state": "A request a user sent to an AI assistant.",
    "labels": {"harmful": {"action": "act", "target_precision": 0.9},
               "harmless": {"action": "drop", "target_precision": 0.9}},
    "max_unknown_rate": 0.5,
    "questions": {
        "physical_harm": {"type": "noul", "instructions": "Does it ask for help to hurt someone?"},
        "topic": {"type": "choice", "instructions": "What is it about?",
                  "criteria": {"weapons": None, "cooking": None, "unknown": "cannot tell"}},
        "urgency": {"type": "score", "instructions": "How urgent?",
                    "levels": ["not urgent", "somewhat urgent", "very urgent"]},
    },
})

ROWS = [{"id": f"r{i}", "state": f"request {i} ({'harmful' if i % 2 else 'harmless'})",
         "label": "harmful" if i % 2 else "harmless"} for i in range(120)]

FAST = {"eval_splits": 12}           # fewer splits than the config default, for speed


def _answers(harm, topic, level):
    probs = dict(zip(("weapons", "cooking", "unknown"), topic))
    choice = max(probs, key=probs.get)
    return {"physical_harm": NoulAnswer(harm),
            "topic": ChoiceAnswer(choice, probs[choice], probs),
            "urgency": ScoreAnswer(level, 1.0, {"0": 0.0, "1": 1.0, "2": 0.0})}


class Fake:
    """Counts calls; can fail chosen states once, or always raise."""

    def __init__(self, name, kind="jev", fail=(), raise_always=None):
        self.name, self.kind = name, kind
        self.fail, self.raise_always = set(fail), raise_always
        self.calls = 0
        self._lock = threading.Lock()

    def evaluate(self, state, qset, *, personal_data=False):
        with self._lock:
            self.calls += 1
        if self.raise_always is not None:
            raise self.raise_always
        if state in self.fail:
            raise RuntimeError("HTTP 503 from the fake")
        return self.answer(state, random.Random(state))


class Informative(Fake):
    def answer(self, state, rng):
        harmful = "(harmful)" in state
        harm = min(1.0, max(0.0, (0.7 if harmful else 0.3) + rng.uniform(-0.35, 0.35)))
        t = [rng.random() + 0.01 for _ in range(3)]
        topic = tuple(x / sum(t) for x in t)
        return Evaluation("fake-jev", _answers(harm, topic, rng.uniform(0, 2)),
                          100, 0, 0.000004, rng.uniform(0.2, 0.5))


class Noise(Fake):
    def answer(self, state, rng):
        # Uninformative, its topic always "unknown", and no cost reported.
        return JudgeEvaluation("fake-llm", _answers(rng.random(), (0.1, 0.1, 0.8),
                                                    rng.uniform(0, 2)),
                               0, 0, 0.0, rng.uniform(1.0, 2.0), cost_known=False)


def test_informative_backend_wins_and_the_report_shows_why(tmp_path):
    good, noise = Informative("good"), Noise("noise", kind="llm")
    report = eval_backends(QSET, ROWS, [good, noise], cache_dir=tmp_path, **FAST)
    assert isinstance(report, EvalReport)
    g, n = report.backends["good"], report.backends["noise"]
    assert g["answered"] == n["answered"] == 120 and good.calls == noise.calls == 120
    assert g["splits_scored"] == n["splits_scored"] == 12
    assert g["metrics"]["accuracy"] > n["metrics"]["accuracy"] + 0.15
    assert g["metrics"]["auc"] > 0.85 > 0.65 > n["metrics"]["auc"]
    for key in ("brier", "ece", "log_loss"):
        assert g["metrics"][key] is not None
    assert sum(g["bands"][k] for k in ("act", "drop", "escalate")) == pytest.approx(1.0)
    # The noise backend never clears the unknown ceiling: everything escalates.
    assert n["bands"]["escalate"] == 1.0 and n["over_unknown_ceiling"] == 1.0

    pair = report.pairs[0]
    assert (pair["first"], pair["second"], pair["splits"]) == ("good", "noise", 12)
    assert pair["accuracy_diff"] > 0.15 and pair["auc_diff"] > 0.2
    assert pair["first_wins_auc"] == 1.0 and pair["second_wins_auc"] == 0.0

    # Per-question visibility: the noise backend's topic is all unknown and flat.
    assert n["questions"]["topic"]["unknown_rate"] == 1.0
    assert any("topic answered unknown in 100%" in f for f in n["flags"])
    assert any("topic never varies" in f for f in n["flags"])
    assert g["questions"]["physical_harm"]["features"]["physical_harm"]["weight"]["harmless"] < 0
    assert g["latency_s"]["p95"] >= g["latency_s"]["median"] > 0

    assert g["cost"]["usd"] == pytest.approx(120 * 0.000004)
    assert n["cost"]["usd"] is None and "unknown" in n["cost"]["text"]

    data = json.loads(report.to_json())                     # strict JSON
    assert data["question_set"]["content_hash"] == QSET.content_hash
    md = report.to_markdown()
    assert "| good |" in md and "| good vs noise |" in md and "unknown" in md


def test_rerun_resumes_from_the_cache(tmp_path):
    eval_backends(QSET, ROWS, [Informative("good")], cache_dir=tmp_path, **FAST)
    again = Informative("good")
    report = eval_backends(QSET, ROWS, [again], cache_dir=tmp_path, **FAST)
    assert again.calls == 0 and report.backends["good"]["new_calls"] == 0
    assert report.backends["good"]["answered"] == 120
    lines = cache_path(tmp_path, "good").read_text().splitlines()
    assert len(lines) == 120 and json.loads(lines[0])["answers_hash"] == QSET.answers_hash


def test_a_changed_question_set_reasks(tmp_path):
    eval_backends(QSET, ROWS, [Informative("good")], cache_dir=tmp_path, **FAST)
    raw = {"name": QSET.name, "version": 2, "state": "A message to an assistant.",
           "labels": {l: {"action": v.action, "target_precision": v.target_precision}
                      for l, v in QSET.labels.items()},
           "max_unknown_rate": QSET.max_unknown_rate,
           "questions": {"physical_harm": {"type": "noul", "instructions": "Harm?"},
                         "topic": {"type": "choice", "instructions": "About?",
                                   "criteria": {"weapons": None, "unknown": None}},
                         "urgency": {"type": "score", "instructions": "Urgent?",
                                     "levels": ["no", "somewhat", "very"]}}}
    again = Informative("good")
    eval_backends(load_question_set(raw), ROWS, [again], cache_dir=tmp_path, **FAST)
    assert again.calls == 120


def test_an_error_row_is_recorded_and_retried_next_run(tmp_path):
    flaky = Informative("good", fail={ROWS[3]["state"]})
    report = eval_backends(QSET, ROWS, [flaky], cache_dir=tmp_path, **FAST)
    assert report.backends["good"]["errors"] == 1 and report.backends["good"]["answered"] == 119
    lines = [json.loads(x) for x in cache_path(tmp_path, "good").read_text().splitlines()]
    failed = [x for x in lines if x["id"] == ROWS[3]["id"]]
    assert [(x["status"], x["error_class"]) for x in failed] == [("error", "RuntimeError")]
    assert all("state" not in x for x in lines)             # the cache keeps no state text
    healed = Informative("good")
    report = eval_backends(QSET, ROWS, [healed], cache_dir=tmp_path, **FAST)
    assert healed.calls == 1
    assert report.backends["good"]["errors"] == 0 and report.backends["good"]["answered"] == 120


def test_a_refusing_backend_is_counted_not_crashed(tmp_path):
    refuser = Fake("jev-refuses", raise_always=RetentionPolicyError("no zero-retention route"))
    report = eval_backends(QSET, ROWS, [Informative("good"), refuser], cache_dir=tmp_path,
                           personal_data=True, **FAST)
    r = report.backends["jev-refuses"]
    assert r["refused"] == 120 and r["answered"] == 0 and r["metrics"] is None
    assert r["splits_scored"] == 0 and r["cost"]["usd"] is None
    assert report.backends["good"]["metrics"]["accuracy"] > 0.7
    assert report.pairs[0]["splits"] == 0 and report.pairs[0]["auc_diff"] is None
    assert "n/a" in report.to_markdown()


def test_an_account_failure_stops_the_backend(tmp_path):
    broke = Fake("broke", raise_always=CreditsExhaustedError("no credit"))
    report = eval_backends(QSET, ROWS, [broke], cache_dir=tmp_path, workers=1, **FAST)
    b = report.backends["broke"]
    assert "CreditsExhaustedError" in b["aborted"]
    assert broke.calls < len(ROWS) and b["missing"] == len(ROWS)


def test_workers_come_from_config_by_kind_and_per_call(tmp_path, monkeypatch):
    from llm_utils.judge import evaluate as ev_mod
    seen = {}
    real = ev_mod.ThreadPoolExecutor

    class Spy(real):
        def __init__(self, max_workers=None, **kw):
            seen.setdefault("workers", []).append(max_workers)
            super().__init__(max_workers=max_workers, **kw)

    monkeypatch.setattr(ev_mod, "ThreadPoolExecutor", Spy)
    eval_backends(QSET, ROWS[:40], [Informative("a"), Noise("b", kind="llm")],
                  cache_dir=tmp_path, eval_splits=2)
    assert seen["workers"] == [8, 16]
    eval_backends(QSET, ROWS[:40], [Informative("c")], cache_dir=tmp_path, eval_splits=2,
                  workers={"c": 3})
    assert seen["workers"][-1] == 3


def test_rows_are_checked(tmp_path):
    with pytest.raises(ValueError, match="vocabulary"):
        eval_backends(QSET, [{"id": 1, "state": "x", "label": "spam"}], [Informative("a")],
                      cache_dir=tmp_path)
    with pytest.raises(ValueError, match="twice"):
        eval_backends(QSET, [ROWS[0], ROWS[0]], [Informative("a")], cache_dir=tmp_path)
    with pytest.raises(ValueError, match="unique"):
        eval_backends(QSET, ROWS, [Informative("a"), Informative("a")], cache_dir=tmp_path)


def test_load_rows_and_fit_the_deployable_combiner(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n\n")
    rows = load_rows(path)
    assert rows == ROWS
    eval_backends(QSET, rows, [Informative("good")], cache_dir=tmp_path / "cache", **FAST)
    comb = fit_combiner(QSET, rows, "good", tmp_path / "cache")
    assert isinstance(comb, Combiner)
    assert comb.basis["rows"] == 120 and comb.basis["backend"] == {"name": "good",
                                                                    "model": "fake-jev"}
    assert comb.basis["out_of_fold"]["auc"] > 0.85
    with pytest.raises(ValueError, match="no answered rows"):
        fit_combiner(QSET, rows, "never-ran", tmp_path / "cache")


# -- review fixes (2026-10-12) ------------------------------------------------

def test_new_text_under_an_old_id_is_asked_again(tmp_path):
    jev = Informative("jev")
    eval_backends(QSET, ROWS[:40], [jev], cache_dir=tmp_path, **FAST)
    assert jev.calls == 40
    changed = [dict(r, state=r["state"] + " (edited)") if i < 5 else r
               for i, r in enumerate(ROWS[:40])]
    eval_backends(QSET, changed, [jev], cache_dir=tmp_path, **FAST)
    assert jev.calls == 45                       # only the five edited rows


def test_counts_cover_only_the_rows_given(tmp_path):
    jev = Informative("jev")
    eval_backends(QSET, ROWS, [jev], cache_dir=tmp_path, **FAST)
    report = eval_backends(QSET, ROWS[:40], [jev], cache_dir=tmp_path, **FAST)
    b = report.backends["jev"]
    assert (b["answered"], b["missing"], b["new_calls"]) == (40, 0, 0)


def test_a_changed_backend_cache_key_asks_again(tmp_path):
    jev = Informative("llm")
    eval_backends(QSET, ROWS[:40], [jev], cache_dir=tmp_path, **FAST)
    jev.cache_key = "prompt-v2"
    eval_backends(QSET, ROWS[:40], [jev], cache_dir=tmp_path, **FAST)
    assert jev.calls == 80
    # fit_combiner reads the newest answers of this question set either way.
    assert fit_combiner(QSET, ROWS[:40], "llm", tmp_path).basis["rows"] == 40


def test_a_cut_short_last_line_does_not_swallow_the_next_record(tmp_path):
    jev = Informative("jev", fail={ROWS[0]["state"]})
    eval_backends(QSET, ROWS[:40], [jev], cache_dir=tmp_path, **FAST)
    path = cache_path(tmp_path, "jev")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"id": "r0", "status": "ok", "answ')          # a crash mid-write
    jev.fail = set()
    eval_backends(QSET, ROWS[:40], [jev], cache_dir=tmp_path, **FAST)
    calls = jev.calls
    report = eval_backends(QSET, ROWS[:40], [jev], cache_dir=tmp_path, **FAST)
    assert jev.calls == calls                    # the retried row's record was kept
    assert report.backends["jev"]["answered"] == 40


def test_unanswered_rows_are_counted_by_label_and_flagged(tmp_path):
    harmful = [r["state"] for r in ROWS if r["label"] == "harmful"][:6]
    jev = Informative("jev", fail=harmful)
    report = eval_backends(QSET, ROWS, [jev], cache_dir=tmp_path, **FAST)
    b = report.backends["jev"]
    assert b["unanswered_by_label"] == {"harmful": 6, "harmless": 0}
    assert any("unanswered rows by label" in f for f in b["flags"])


def test_default_split_indices_match_original_stratified_shuffle():
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    from llm_utils.judge.combiner import judge_settings
    from llm_utils.judge.evaluate import _split_indices

    settings = judge_settings(eval_splits=3)
    y = np.asarray([QSET.label_names.index(r["label"]) for r in ROWS])
    expected = list(StratifiedShuffleSplit(
        n_splits=3, train_size=settings["fit_fraction"], random_state=settings["seed"]
    ).split(np.zeros(len(ROWS)), y))
    actual = _split_indices(ROWS, y, settings)
    for (fit, test), (old_fit, old_test) in zip(actual, expected):
        np.testing.assert_array_equal(fit, old_fit)
        np.testing.assert_array_equal(test, old_test)
    assert len(actual) == len(expected)


def test_grouped_splits_never_share_groups_and_are_reproducible():
    import numpy as np

    from llm_utils.judge.combiner import judge_settings
    from llm_utils.judge.evaluate import _check_rows, _split_indices

    # Unequal groups, noncontiguous members, and distinct integer/string ids.
    groups = [1, "1", "b", "c", "d", "d", "d"]
    rows = _check_rows([dict(r, cluster=groups[i % len(groups)])
                        for i, r in enumerate(ROWS)], QSET)
    settings = judge_settings(split_strategy="grouped", group_key="cluster", eval_splits=4)
    splits = _split_indices(rows, [r["label"] for r in rows], settings)
    again = _split_indices(rows, [r["label"] for r in rows], settings)
    assert len(splits) == 4
    for (fit, test), (fit_again, test_again) in zip(splits, again):
        fit_groups = {rows[i]["cluster"] for i in fit}
        test_groups = {rows[i]["cluster"] for i in test}
        assert fit_groups.isdisjoint(test_groups)
        assert len(fit_groups) == 2  # floor(5 unique groups * 0.5)
        assert set(fit) | set(test) == set(range(len(rows)))
        np.testing.assert_array_equal(fit, fit_again)
        np.testing.assert_array_equal(test, test_again)


def test_time_ordered_split_sorts_rows_and_keeps_boundary_ties_together():
    from llm_utils.judge.combiner import judge_settings
    from llm_utils.judge.evaluate import _check_rows, _split_indices

    times = [30, 10, 20, 20, 40, 20]
    rows = _check_rows([dict(r, observed=t) for r, t in zip(ROWS, times)], QSET)
    settings = judge_settings(split_strategy="time_ordered", time_key="observed")
    splits = _split_indices(rows, [r["label"] for r in rows], settings)
    assert len(splits) == 1
    fit, test = splits[0]
    assert max(rows[i]["observed"] for i in fit) < min(rows[i]["observed"] for i in test)
    assert set(fit) == {1}
    assert set(test) == {0, 2, 3, 4, 5}


def test_time_ordered_split_compares_instants_across_timezone_offsets():
    from llm_utils.judge.combiner import judge_settings
    from llm_utils.judge.evaluate import _split_indices

    times = ["2026-01-01T00:00:00-08:00", "2026-01-01T02:00:00Z",
             "2026-01-01T01:00:00Z", "2026-01-01T04:00:00+00:00"]
    rows = [dict(r, timestamp=t) for r, t in zip(ROWS, times)]
    fit, test = _split_indices(rows, [r["label"] for r in rows],
                               judge_settings(split_strategy="time_ordered"))[0]
    assert list(fit) == [2, 1]
    assert list(test) == [3, 0]


@pytest.mark.parametrize("strategy,metadata,match", [
    ("grouped", {}, "group"),
    ("grouped", {"group": []}, "group"),
    ("grouped", {"group": "only-one"}, "cannot split"),
    ("time_ordered", {}, "timestamp"),
    ("time_ordered", {"timestamp": float("nan")}, "finite"),
    ("time_ordered", {"timestamp": float("inf")}, "finite"),
    ("time_ordered", {"timestamp": "2026-01-01"}, "timezone"),
    ("time_ordered", {"timestamp": "not-a-date"}, "invalid"),
    ("time_ordered", {"timestamp": True}, "invalid"),
    ("time_ordered", {"timestamp": 1}, "no fit rows"),
    ("unknown", {}, "unknown split_strategy"),
])
def test_invalid_splits_fail_before_backend_calls(tmp_path, strategy, metadata, match):
    backend = Informative("good")
    with pytest.raises(ValueError, match=match):
        eval_backends(QSET, [dict(r, **metadata) for r in ROWS], [backend],
                      cache_dir=tmp_path, split_strategy=strategy)
    assert backend.calls == 0
    assert not cache_path(tmp_path, backend.name).exists()


@pytest.mark.parametrize("strategy,split_count", [("grouped", 2), ("time_ordered", 1)])
def test_split_metadata_reaches_scoring_and_recall_reaches_reports(tmp_path, strategy, split_count):
    rows = [dict(r, cluster=i // 4, observed=i // 2) for i, r in enumerate(ROWS)]
    backend = Informative("good")
    report = eval_backends(QSET, list(reversed(rows)), [backend], cache_dir=tmp_path,
                           split_strategy=strategy, group_key="cluster", time_key="observed",
                           eval_splits=2)
    assert report.settings["split_strategy"] == strategy
    assert report.settings["group_key"] == "cluster"
    assert report.settings["time_key"] == "observed"
    assert report.settings["eval_splits"] == split_count
    result = report.backends["good"]
    assert result["splits_scored"] == split_count
    recall = result["metrics"]["recall_per_class"]
    assert set(recall) == set(QSET.label_names)
    assert all(0 <= value <= 1 for value in recall.values())
    assert json.loads(report.to_json())["backends"]["good"]["metrics"]["recall_per_class"] == recall
    assert "| backend | label | recall |" in report.to_markdown()
    assert "| good | harmful |" in report.to_markdown()
    for line in cache_path(tmp_path, "good").read_text().splitlines():
        assert not {"cluster", "observed"} & json.loads(line).keys()


def test_default_evaluation_results_match_original_random_split(tmp_path, monkeypatch):
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    from llm_utils.judge import evaluate

    default = eval_backends(QSET, ROWS, [Informative("good")], cache_dir=tmp_path, eval_splits=2)

    def original_splits(rows, y, settings):
        return list(StratifiedShuffleSplit(
            n_splits=settings["eval_splits"], train_size=settings["fit_fraction"],
            random_state=settings["seed"]).split(np.zeros(len(rows)), y))

    monkeypatch.setattr(evaluate, "_split_indices", original_splits)
    baseline = eval_backends(QSET, ROWS, [Informative("good")], cache_dir=tmp_path, eval_splits=2)
    assert default.backends["good"]["metrics"] == baseline.backends["good"]["metrics"]
    assert default.backends["good"]["bands"] == baseline.backends["good"]["bands"]
    assert default.backends["good"]["questions"] == baseline.backends["good"]["questions"]


def test_report_averages_recall_by_label_and_ignores_unscorable_splits():
    from llm_utils.judge.evaluate import _backend_summary

    splits = [None] + [
        {"metrics": {"accuracy": 0.5, "auc": 0.5, "recall_per_class": recall},
         "bands": {"act": 0.0, "drop": 0.0, "escalate": 1.0},
         "accuracy_in_band": float("nan")}
        for recall in ([0.2, 1.0], [0.6, 0.0])
    ]
    data = {"vectors": {}, "unknown": {}, "flags": {}, "latency": [], "costs": []}
    answer = {"records": {}, "rows": [], "calls": 0, "aborted": None}
    # With no feature vectors, _backend_summary still reports weights from
    # scored splits; give each label its complete synthetic weight vector.
    from llm_utils.judge.features import feature_names

    for split in splits[1:]:
        split["weights"] = {label: [0.0] * len(feature_names(QSET)) for label in QSET.label_names}
    report = _backend_summary(QSET, "fake", data, answer, splits)
    assert report["splits_scored"] == 2
    assert report["metrics"]["recall_per_class"] == pytest.approx({"harmful": 0.4, "harmless": 0.5})


def test_split_strategy_and_field_names_can_come_from_yaml_config(tmp_path):
    from llm_utils import config

    config.configure(judge={"split_strategy": "grouped", "group_key": "cluster", "eval_splits": 2})
    rows = [dict(r, cluster=i // 4) for i, r in enumerate(ROWS)]
    report = eval_backends(QSET, rows, [Informative("good")], cache_dir=tmp_path)
    assert report.settings["split_strategy"] == "grouped"
    assert report.settings["group_key"] == "cluster"
    assert report.backends["good"]["splits_scored"] == 2
    # Per-call random overrides do not require the configured group field.
    report = eval_backends(QSET, ROWS, [Informative("good")], cache_dir=tmp_path,
                           split_strategy="random")
    assert report.settings["split_strategy"] == "random"
    assert report.backends["good"]["splits_scored"] == 2
