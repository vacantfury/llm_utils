"""TypeSafe Jev typed evaluation (t14). Offline: the HTTP layer is a fake;
dummy key; no network."""

import httpx2
import pytest

from llm_utils import (
    BaseLLMService, ChoiceAnswer, CreditsExhaustedError, Evaluation, FatalModelError,
    InvalidCredentialError, LLMModel, LLMServiceFactory, NoulAnswer, Provider,
    RetentionPolicyError, ScoreAnswer, SpendCapExceededError, TypeSafeService,
    choice_question, noul_question, route_status, score_question, spend_status,
)
from llm_utils import spend
from llm_utils.llm_services import typesafe_service as ts

JEV = LLMModel.JEV_1_13_0

OK_BODY = {
    "model": "jev-1.13.0",
    "answers": {
        "billing": {"type": "noul", "noul": 0.97},
        "topic": {"type": "choice", "choice": "billing", "confidence": 0.9,
                  "probabilities": {"billing": 0.9, "tech": 0.05, "other": 0.05}},
        "urgency": {"type": "score", "score": 1.7, "confidence": 0.8,
                    "legend": {"0": "later", "1": "this week", "2": "today"},
                    "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}},
    },
    "usage": {"input_tokens": 1_000_000, "output_tokens": 12},
}

QUESTIONS = {
    "billing": noul_question("Is this about billing?"),
    "topic": choice_question(["billing", "tech", "other"], "What is it about?"),
    "urgency": score_question(["later", "this week", "today"], "How urgent?"),
}


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    def post(self, url, json=None, headers=None):
        self.requests.append((url, json, headers))
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        code, body, hdrs = a if len(a) == 3 else (*a, {})
        return httpx2.Response(code, json=body, headers=hdrs,
                               request=httpx2.Request("POST", url))

    def close(self):
        pass


@pytest.fixture
def svc(monkeypatch):
    monkeypatch.setattr(ts.time, "sleep", lambda s: None)
    s = TypeSafeService(JEV, api_key="test-key")

    def use(*answers):
        s._client = FakeClient(answers)
        return s
    return use


class TestRegistry:
    def test_row_and_provider(self):
        assert JEV.provider is Provider.TYPESAFE and JEV.jurisdiction == "us"
        assert JEV.output_price == 0.0 and JEV.input_price == 0.042

    def test_factory_builds_it(self):
        assert isinstance(LLMServiceFactory.create(JEV, api_key="k"), TypeSafeService)

    def test_missing_key(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            TypeSafeService(JEV)

    def test_broker_mode_refused(self, monkeypatch):
        monkeypatch.setenv("LLM_UTILS_BROKER_URL", "http://127.0.0.1:9")
        from llm_utils import BrokerModeUnsupportedError
        with pytest.raises(BrokerModeUnsupportedError):
            TypeSafeService(JEV, api_key="k")
        assert not route_status(JEV).usable

    def test_route_status_reads_the_key(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        assert not route_status(JEV).usable
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        assert route_status(JEV).usable


class TestQuestions:
    def test_builders(self):
        assert noul_question("q?", true="yes", false="no") == {
            "type": "noul", "instructions": "q?", "criteria": {"true": "yes", "false": "no"}}
        assert choice_question({"a": "first", "b": None})["criteria"] == {"a": "first", "b": None}

    @pytest.mark.parametrize("n", [1, 256])
    def test_choice_limits(self, n):
        with pytest.raises(ValueError):
            choice_question([str(i) for i in range(n)])

    @pytest.mark.parametrize("n", [1, 11])
    def test_score_limits(self, n):
        with pytest.raises(ValueError):
            score_question([str(i) for i in range(n)])

    def test_raw_dicts_are_checked_too(self, svc):
        s = svc()
        with pytest.raises(ValueError, match="type"):
            s.evaluate("x", {"q": {"type": "open"}})
        with pytest.raises(ValueError):
            s.evaluate("x", {})
        assert s._client.requests == []


class TestEvaluate:
    def test_request_and_typed_answers(self, svc):
        s = svc((200, OK_BODY))
        ev = s.evaluate({"ticket": "charged twice"}, QUESTIONS)
        url, body, headers = s._client.requests[0]
        assert url == "https://api.typesafe.ai/v1/systemone"
        assert headers["Authorization"] == "Bearer test-key"
        assert body["model"] == "jev-1.13.0" and body["state"] == {"ticket": "charged twice"}
        assert set(body["questions"]) == {"billing", "topic", "urgency"}
        assert isinstance(ev, Evaluation)
        assert ev.answers["billing"] == NoulAnswer(0.97)
        assert isinstance(ev.answers["topic"], ChoiceAnswer)
        assert ev.answers["topic"].probabilities["billing"] == 0.9
        assert isinstance(ev.answers["urgency"], ScoreAnswer) and ev.answers["urgency"].score == 1.7

    def test_cost_is_input_only_and_recorded(self, svc):
        rows = []
        BaseLLMService.set_usage_hook(lambda model, i, o, c, **kw: rows.append((model, i, o, c)))
        try:
            ev = svc((200, OK_BODY)).evaluate("x", QUESTIONS)
        finally:
            BaseLLMService.clear_usage_hook()
        assert ev.cost_usd == pytest.approx(0.042)
        assert rows == [(JEV, 1_000_000, 12, pytest.approx(0.042))]

    def test_retries_429_and_529_then_succeeds(self, svc):
        s = svc((429, {}, {"retry-after": "0"}), (529, {}), (200, OK_BODY))
        assert s.evaluate("x", QUESTIONS).model == "jev-1.13.0"
        assert len(s._client.requests) == 3

    def test_retries_a_dropped_connection(self, svc):
        s = svc(httpx2.ConnectError("reset"), (200, OK_BODY))
        s.evaluate("x", QUESTIONS)
        assert len(s._client.requests) == 2

    def test_gives_up_after_max_retries(self, svc):
        s = svc(*[(529, {})] * 4)
        with pytest.raises(ts.TypeSafeHTTPError, match="529"):
            s.evaluate("x", QUESTIONS)
        assert len(s._client.requests) == 4

    @pytest.mark.parametrize("code,exc", [
        (401, InvalidCredentialError), (402, CreditsExhaustedError),
        (404, FatalModelError), (422, ValueError)])
    def test_error_mapping(self, svc, code, exc):
        with pytest.raises(exc):
            svc((code, {"detail": [{"msg": "bad"}]})).evaluate("x", QUESTIONS)

    def test_personal_data_needs_a_zero_retention_route(self, svc):
        s = svc((200, OK_BODY))
        with pytest.raises(RetentionPolicyError):
            s.evaluate("about a person", QUESTIONS, personal_data=True)
        assert s._client.requests == []
        s.zero_retention = True
        s.evaluate("about a person", QUESTIONS, personal_data=True)

    def test_no_chat_surface(self, svc):
        with pytest.raises(NotImplementedError, match="evaluate"):
            svc().chat("hi")

    def test_spend_cap_applies(self, svc, monkeypatch):
        monkeypatch.setenv(spend.MAX_USD_ENV, "0")
        s = svc((200, OK_BODY))
        with pytest.raises(SpendCapExceededError):
            s.evaluate("x" * 4000, QUESTIONS)
        assert s._client.requests == []
        assert spend_status().reserved == 0

    def test_async(self, svc):
        import asyncio
        ev = asyncio.run(svc((200, OK_BODY)).aevaluate("x", QUESTIONS))
        assert ev.answers["billing"].noul == 0.97
