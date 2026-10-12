"""Split-question judgments (t23): question sets, LLM backends, features and
the combiner. Offline and keyless; fitting needs the judge extra
(scikit-learn), so those tests skip without it."""

import json
import random
from types import SimpleNamespace

import pytest

from llm_utils import LLMModel, ChoiceAnswer, Evaluation, NoulAnswer, ScoreAnswer
from llm_utils.base_llm_service import BaseLLMService
from llm_utils.judge import (
    Combiner, Decision, JevBackend, LLMBackend, feature_names, features, load_question_set,
    unknown_rate,
)
from llm_utils.judge.backends import _structured_schema, first_json_object, render_prompt
from llm_utils.testing import FakeService

SET = {
    "name": "refusal",
    "version": 1,
    "description": "Should an AI assistant refuse this request?",
    "state": "A request a user sent to an AI assistant.",
    "labels": {
        "harmful": {"action": "act", "target_precision": 0.9},
        "harmless": {"action": "drop", "target_precision": 0.9},
    },
    "max_unknown_rate": 0.5,
    "questions": {
        "physical_harm": {"type": "noul",
                          "instructions": "Does the request ask for help to hurt someone?"},
        "topic": {"type": "choice", "instructions": "What is the request about?",
                  "criteria": {"weapons": "arms or explosives", "cooking": "food",
                               "unknown": "cannot tell from the text"}},
        "urgency": {"type": "score", "instructions": "How urgent does the request sound?",
                    "levels": ["not urgent", "somewhat urgent", "very urgent"]},
    },
}


def qset(**changes):
    raw = {**SET, **changes}
    return load_question_set(raw)


def ev(harm, topic=(0.2, 0.7, 0.1), score=0.5, model="fake"):
    probs = dict(zip(("weapons", "cooking", "unknown"), topic))
    choice = max(probs, key=probs.get)
    return Evaluation(model, {
        "physical_harm": NoulAnswer(harm),
        "topic": ChoiceAnswer(choice, probs[choice], probs),
        "urgency": ScoreAnswer(score, 0.5, {"0": 0.5, "1": 0.5, "2": 0.0}),
    }, 10, 0, 0.0001, 0.3)


# ---------------------------------------------------------------------------
# Question sets
# ---------------------------------------------------------------------------

class TestQuestionSet:
    def test_valid_set_loads_from_a_dict_and_a_file(self, tmp_path):
        import yaml
        q = qset()
        assert q.name == "refusal" and q.label_names == ("harmful", "harmless")
        assert q.questions["topic"] == {
            "type": "choice", "instructions": "What is the request about?",
            "criteria": {"weapons": "arms or explosives", "cooking": "food",
                         "unknown": "cannot tell from the text"}}
        assert q.questions["urgency"]["criteria"] == ["not urgent", "somewhat urgent",
                                                      "very urgent"]
        assert q.choice_questions == ("topic",)
        path = tmp_path / "refusal.yaml"
        path.write_text(yaml.safe_dump(SET))
        assert load_question_set(path) == q

    @pytest.mark.parametrize("change,match", [
        ({"questions": {"a": {"type": "noul"}}}, "instructions are required"),
        ({"questions": {"a": {"type": "choice", "instructions": "x?",
                              "criteria": {"b": None, "c": None}}}}, "option 'unknown'"),
        ({"questions": {"a": {"type": "choice", "instructions": "x?",
                              "criteria": {"unknown": None}}}}, "2..255 options"),
        ({"questions": {"a": {"type": "score", "instructions": "x?", "levels": ["only"]}}},
         "2..10 levels"),
        ({"questions": {"a": {"type": "score", "instructions": "x?",
                              "levels": [str(i) for i in range(11)]}}}, "2..10 levels"),
        ({"questions": {"a": {"type": "open", "instructions": "x?"}}}, "type must be"),
        ({"questions": {"a": {"type": "noul", "instructions": "x?", "levels": []}}},
         "unknown keys"),
        ({"labels": {"only": {"action": "act", "target_precision": 0.9}}}, "at least 2"),
        ({"labels": {"a": {"action": "maybe", "target_precision": 0.9},
                     "b": {"action": "drop", "target_precision": 0.9}}}, "action must be"),
        ({"labels": {"a": {"action": "act", "target_precision": 0},
                      "b": {"action": "drop", "target_precision": 0.9}}}, r"\(0, 1\]"),
        ({"labels": {"a": {"action": "act", "target_precision": 1.2},
                      "b": {"action": "drop", "target_precision": 0.9}}}, r"\(0, 1\]"),
        ({"max_unknown_rate": 1.5}, r"\[0, 1\]"),
        ({"max_unknown_rate": -0.1}, r"\[0, 1\]"),
        ({"surprise": 1}, "unknown keys"),
    ])
    def test_each_validation_error(self, change, match):
        with pytest.raises(ValueError, match=match):
            qset(**change)

    def test_missing_key(self):
        raw = dict(SET)
        del raw["state"]
        with pytest.raises(ValueError, match="'state'"):
            load_question_set(raw)

    def test_content_hash_ignores_key_order_and_sees_an_instruction_edit(self):
        reordered = {
            **SET,
            "questions": dict(reversed(list(SET["questions"].items()))),
            "labels": dict(reversed(list(SET["labels"].items()))),
        }
        assert load_question_set(reordered).content_hash == qset().content_hash
        edited = {**SET["questions"],
                  "physical_harm": {"type": "noul", "instructions": "Could it hurt anyone?"}}
        assert qset(questions=edited).content_hash != qset().content_hash
        # The unknown ceiling is policy outside the hash; the state header
        # changes the cached answers' key but not the fitted weights' binding.
        assert qset(max_unknown_rate=0.1).content_hash == qset().content_hash
        assert qset(state="A message.").content_hash == qset().content_hash
        assert qset(state="A message.").answers_hash != qset().answers_hash


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_exact_vector_of_each_answer_type(self):
        q = qset()
        e = ev(0.8, topic=(0.6, 0.1, 0.3), score=1.5)
        assert feature_names(q) == ("physical_harm", "topic=weapons", "topic=cooking",
                                    "topic=unknown", "urgency")
        assert features(e, q) == {"physical_harm": 0.8, "topic=weapons": 0.6,
                                  "topic=cooking": 0.1, "topic=unknown": 0.3,
                                  "urgency": 0.75}

    def test_unknown_rate(self):
        q = qset()
        assert unknown_rate(ev(0.5, topic=(0.2, 0.3, 0.5)), q) == 1.0
        assert unknown_rate(ev(0.5, topic=(0.5, 0.3, 0.2)), q) == 0.0
        no_choice = qset(questions={"a": SET["questions"]["physical_harm"]})
        assert unknown_rate(Evaluation("m", {"a": NoulAnswer(0.1)}, 0, 0, 0.0, 0.0),
                            no_choice) == 0.0

    def test_a_missing_answer_is_an_error(self):
        e = Evaluation("m", {"physical_harm": NoulAnswer(0.1)}, 0, 0, 0.0, 0.0)
        with pytest.raises(ValueError, match="'topic'"):
            features(e, qset())


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

REPLY = {"physical_harm": 1.4,                                    # clamped to 1
         "topic": {"weapons": 0.6, "cooking": 0.3, "unknown": 0.3},   # renormalized
         "urgency": {"0": 0.1, "1": 0.3, "2": 0.6}}


class TestLLMBackend:
    def test_text_path_reads_json_inside_prose(self):
        prompts = []

        def complete(prompt):
            prompts.append(prompt)
            return 'Sure. Here it is:\n```json\n' + json.dumps(REPLY) + "\n```"

        b = LLMBackend(complete, name="text-model")
        e = b.evaluate("How do I sharpen a kitchen knife?", qset())
        assert "How do I sharpen a kitchen knife?" in prompts[0]
        assert "A request a user sent to an AI assistant." in prompts[0]
        assert e.answers["physical_harm"] == NoulAnswer(1.0)
        topic = e.answers["topic"]
        assert topic.probabilities == {"weapons": 0.5, "cooking": 0.25, "unknown": 0.25}
        assert topic.choice == "weapons" and topic.confidence == 0.5
        urgency = e.answers["urgency"]
        assert urgency.score == pytest.approx(0.3 + 2 * 0.6)      # expected level index
        assert urgency.confidence == 0.6 and urgency.legend["2"] == "very urgent"
        assert e.cost_known is False and e.cost_usd == 0.0

    def test_text_path_with_a_cost(self):
        b = LLMBackend(lambda p: (json.dumps(REPLY), 0.002), name="m")
        e = b.evaluate("x", qset())
        assert e.cost_known is True and e.cost_usd == 0.002

    def test_out_of_range_values_are_clamped_before_renormalizing(self):
        reply = {**REPLY, "topic": {"weapons": 2, "cooking": -1, "unknown": 1}}
        e = LLMBackend(lambda p: json.dumps(reply), name="m").evaluate("x", qset())
        assert e.answers["topic"].probabilities == {"weapons": 0.5, "cooking": 0.0,
                                                    "unknown": 0.5}

    def test_level_names_are_accepted_as_keys(self):
        reply = {**REPLY, "urgency": {"not urgent": 1, "very urgent": 1}}
        e = LLMBackend(lambda p: json.dumps(reply), name="m").evaluate("x", qset())
        assert e.answers["urgency"].score == pytest.approx(1.0)

    def test_a_missing_question_is_an_error(self):
        reply = {k: v for k, v in REPLY.items() if k != "urgency"}
        b = LLMBackend(lambda p: json.dumps(reply), name="m")
        with pytest.raises(ValueError, match="'urgency'"):
            b.evaluate("x", qset())

    def test_an_all_zero_distribution_is_an_error(self):
        reply = {**REPLY, "topic": {"weapons": 0, "cooking": 0, "unknown": 0}}
        b = LLMBackend(lambda p: json.dumps(reply), name="m")
        with pytest.raises(ValueError, match="all probabilities are zero"):
            b.evaluate("x", qset())

    def test_no_json_is_an_error(self):
        with pytest.raises(ValueError, match="no JSON object"):
            first_json_object("I cannot answer that.")
        assert first_json_object("a {broken then {\"a\": 1} done") == {"a": 1}

    def test_structured_path(self):
        q = qset()

        class Structured:
            model = SimpleNamespace(model_id="fake-structured")

            def __init__(self):
                self.schemas = []

            def chat_structured(self, prompt, schema, system_message=None, **kw):
                self.schemas.append(schema)
                return schema.model_validate(REPLY)

        svc = Structured()
        b = LLMBackend.from_service(svc)
        e = b.evaluate("x", q)
        assert b.name == e.model == "fake-structured"
        assert e.answers["topic"].probabilities["weapons"] == 0.5
        schema = svc.schemas[0].model_json_schema()
        assert set(schema["properties"]) == {"physical_harm", "topic", "urgency"}
        b.evaluate("y", q)
        assert svc.schemas[0] is svc.schemas[1]                  # built once per set

    def test_structured_path_records_the_cost_of_each_call(self):
        q = qset()
        schema = _structured_schema(q)
        fake = FakeService(LLMModel.GPT_6_LUNA,
                           responses=lambda text, cid: schema.model_validate(REPLY))
        b = LLMBackend.from_service(fake, name="luna")
        e = b.evaluate("x", q)
        assert fake.calls[0].method == "chat_structured"
        assert e.cost_known and e.cost_usd == pytest.approx(fake.total_usage.cost)
        assert e.cost_usd > 0 and e.input_tokens > 0

    def test_service_without_structured_output_uses_chat(self):
        class ChatOnly(FakeService):
            chat_structured = BaseLLMService.chat_structured

        fake = ChatOnly(LLMModel.GPT_6_LUNA, responses=json.dumps(REPLY))
        e = LLMBackend.from_service(fake).evaluate("x", qset())
        assert fake.calls[0].method == "batch_chat"
        assert e.answers["physical_harm"].noul == 1.0 and e.cost_known

    def test_prompt_lists_every_question_option_and_level(self):
        p = render_prompt({"text": "hello"}, qset())
        for needle in ("physical_harm", "weapons", "cannot tell from the text",
                       "2: very urgent", '"text": "hello"'):
            assert needle in p


class TestJevBackend:
    def test_passes_personal_data_through(self):
        seen = {}

        class Svc:
            model = SimpleNamespace(model_id="typesafe/jev-1.13-20260917")

            def evaluate(self, state, questions, *, personal_data=False):
                seen.update(state=state, questions=questions, personal_data=personal_data)
                return ev(0.1)

        b = JevBackend(Svc())
        q = qset()
        b.evaluate("hello", q, personal_data=True)
        assert seen == {"state": "hello", "questions": q.questions, "personal_data": True}
        assert b.name == "typesafe/jev-1.13-20260917" and b.kind == "jev"

    def test_default_is_the_openrouter_row(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        b = JevBackend.default()
        assert b.service.model is LLMModel.OR_JEV_1_13
        assert b.service.route.path == "/decisions"


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------

def _binary_rows(n=80, seed=1):
    rng = random.Random(seed)
    evs, labels = [], []
    for i in range(n):
        harmful = i % 2 == 0
        harm = rng.uniform(0.6, 1.0) if harmful else rng.uniform(0.0, 0.4)
        t = [rng.random() + 0.01 for _ in range(3)]
        s = sum(t)
        evs.append(ev(harm, topic=tuple(x / s for x in t), score=rng.uniform(0, 2)))
        labels.append("harmful" if harmful else "harmless")
    return evs, labels


class TestCombiner:
    @pytest.fixture(autouse=True)
    def _needs_sklearn(self):
        pytest.importorskip("sklearn")

    def _sklearn_twin(self, q, evs, labels):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        names = feature_names(q)
        X = np.asarray([[features(e, q)[n] for n in names] for e in evs])
        y = np.asarray([q.label_names.index(l) for l in labels])
        return LogisticRegression(C=1.0, max_iter=1000).fit(X, y), X

    def test_fit_and_pure_python_probabilities_equal_sklearn(self):
        q = qset()
        evs, labels = _binary_rows()
        comb = Combiner.fit(q, evs, labels, backend="fake", model="fake")
        lr, X = self._sklearn_twin(q, evs, labels)
        expected = lr.predict_proba(X)
        for row, p in zip(X, expected):
            got = comb.probabilities(list(row))
            for k, l in enumerate(q.label_names):
                assert got[l] == pytest.approx(p[k], abs=1e-9)
        assert comb.basis["out_of_fold"]["accuracy"] > 0.95
        assert comb.thresholds["harmful"] is not None

    def test_decide_bands_a_clear_case(self):
        q = qset()
        evs, labels = _binary_rows()
        comb = Combiner.fit(q, evs, labels, backend="fake")
        d = comb.decide(ev(0.97, topic=(0.6, 0.3, 0.1)), q)
        assert isinstance(d, Decision)
        assert d.label == "harmful" and d.band == "act" and d.reasons == ()
        d = comb.decide(ev(0.02, topic=(0.2, 0.7, 0.1)), q)
        assert d.label == "harmless" and d.band == "drop"

    def test_save_load_round_trip(self, tmp_path):
        q = qset()
        evs, labels = _binary_rows()
        comb = Combiner.fit(q, evs, labels, backend="fake", model="fake-1")
        path = tmp_path / "refusal.combiner.yaml"
        comb.save(path)
        back = Combiner.load(path)
        assert back.to_dict() == comb.to_dict()
        assert back.basis["question_set"]["content_hash"] == q.content_hash
        assert back.basis["backend"] == {"name": "fake", "model": "fake-1"}
        for key in ("rows", "label_counts", "fitted", "folds", "seed", "out_of_fold",
                    "target_precision", "thresholds"):
            assert key in back.basis
        e = ev(0.7)
        assert back.decide(e, q) == comb.decide(e, q)

    def test_a_different_question_set_is_refused(self):
        q = qset()
        evs, labels = _binary_rows()
        comb = Combiner.fit(q, evs, labels, backend="fake")
        edited = qset(questions={**SET["questions"], "physical_harm": {
            "type": "noul", "instructions": "Could it hurt anyone?"}})
        with pytest.raises(ValueError, match="refit"):
            comb.decide(ev(0.9), edited)

    def test_unreachable_target_never_auto_decides(self):
        q = qset()
        # Every row looks the same, 60/40: no cut reaches 0.9 precision.
        evs = [ev(0.5) for _ in range(50)]
        labels = ["harmful"] * 30 + ["harmless"] * 20
        comb = Combiner.fit(q, evs, labels, backend="fake")
        assert comb.thresholds == {"harmful": None, "harmless": None}
        assert "never" in comb.basis["thresholds"]["harmful"]["reason"]
        d = comb.decide(ev(0.5), q)
        assert d.band == "escalate" and "no threshold" in d.reasons[0]

    def test_unknown_ceiling_escalates(self):
        q = qset()
        evs, labels = _binary_rows()
        comb = Combiner.fit(q, evs, labels, backend="fake")
        d = comb.decide(ev(0.99, topic=(0.1, 0.1, 0.8)), q)
        assert d.unknown_rate == 1.0 and d.band == "escalate"
        assert any("unknown rate" in r for r in d.reasons)
        relaxed = qset(max_unknown_rate=1.0)                     # same hash, looser policy
        assert comb.decide(ev(0.99, topic=(0.1, 0.1, 0.8)), relaxed).band == "act"

    def test_labels_outside_the_vocabulary_are_refused(self):
        evs, labels = _binary_rows(10)
        with pytest.raises(ValueError, match="vocabulary"):
            Combiner.fit(qset(), evs, ["spam"] + labels[1:], backend="fake")

    def test_multiclass(self):
        q = qset(labels={"weapons": {"action": "act", "target_precision": 0.8},
                         "cooking": {"action": "drop", "target_precision": 0.8},
                         "other": {"action": "drop", "target_precision": 0.8}})
        rng = random.Random(3)
        evs, labels = [], []
        for i in range(90):
            k = i % 3
            t = [0.05, 0.05, 0.05]
            t[k] = 0.85 + rng.uniform(-0.05, 0.05)
            s = sum(t)
            evs.append(ev(rng.random(), topic=tuple(x / s for x in t)))
            labels.append(("weapons", "cooking", "other")[k])
        comb = Combiner.fit(q, evs, labels, backend="fake")
        assert len(comb.labels) == 3 and all(any(c) for c in comb.coefficients.values())
        lr, X = self._sklearn_twin(q, evs, labels)
        for row, p in zip(X[:10], lr.predict_proba(X[:10])):
            got = comb.probabilities(list(row))
            for k, l in enumerate(q.label_names):
                assert got[l] == pytest.approx(p[k], abs=1e-9)
        d = comb.decide(ev(0.5, topic=(0.05, 0.9, 0.05)), q)
        assert d.label == "cooking" and set(d.probabilities) == {"weapons", "cooking", "other"}
        assert comb.basis["out_of_fold"]["auc"] > 0.9


# -- review fixes (2026-10-12) ------------------------------------------------

def test_digit_string_levels_keep_their_index():
    from llm_utils.judge.backends import answers_from_reply
    qset = load_question_set({
        "name": "t", "version": 1, "state": "s", "max_unknown_rate": 0.5,
        "labels": {"a": {"action": "act", "target_precision": 0.9},
                   "b": {"action": "drop", "target_precision": 0.9}},
        "questions": {"sev": {"type": "score", "instructions": "rate",
                              "levels": ["1", "2", "3", "4", "5"]}}})
    top = answers_from_reply({"sev": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1.0}}, qset)["sev"]
    bottom = answers_from_reply({"sev": {"0": 1.0, "1": 0, "2": 0, "3": 0, "4": 0}}, qset)["sev"]
    assert (top.score, bottom.score) == (4.0, 0.0)


def test_backends_on_one_service_share_one_usage_tap():
    from llm_utils.judge import LLMBackend

    class Svc:
        model = type("M", (), {"model_id": "fake"})()

        def _record_usage(self, i, o, c, t):
            pass

        def chat(self, prompt):
            self._record_usage(10, 5, 0.001, False)
            return '{"x": 0.5}'

    svc = Svc()
    for _ in range(1500):
        backend = LLMBackend.from_service(svc, name="fake")
    qset = load_question_set({
        "name": "t", "version": 1, "state": "s", "max_unknown_rate": 0.5,
        "labels": {"a": {"action": "act", "target_precision": 0.9},
                   "b": {"action": "drop", "target_precision": 0.9}},
        "questions": {"x": {"type": "noul", "instructions": "yes?"}}})
    ev = backend.evaluate("text", qset)
    assert (ev.input_tokens, ev.output_tokens, ev.cost_usd) == (10, 5, 0.001)
