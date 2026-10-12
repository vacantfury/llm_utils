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

    def test_a_long_retry_after_is_not_waited_out(self, svc):
        s = svc((429, {}, {"retry-after": "3600"}))
        with pytest.raises(ts.TypeSafeHTTPError, match="429"):
            s.evaluate("x", QUESTIONS)
        assert len(s._client.requests) == 1

    def test_a_read_timeout_after_sending_is_not_resent(self, svc):
        s = svc(httpx2.ReadTimeout("slow"), (200, OK_BODY))
        with pytest.raises(httpx2.ReadTimeout):
            s.evaluate("x", QUESTIONS)
        assert len(s._client.requests) == 1

    def test_usage_recorded_even_when_the_answer_is_unreadable(self, svc):
        rows = []
        bad = {**OK_BODY, "answers": {"billing": {"type": "mystery"}}}
        BaseLLMService.set_usage_hook(lambda model, i, o, c, **kw: rows.append(c))
        try:
            with pytest.raises(ValueError, match="unknown type"):
                svc((200, bad)).evaluate("x", QUESTIONS)
        finally:
            BaseLLMService.clear_usage_hook()
        assert rows == [pytest.approx(0.042)]

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


# ---------------------------------------------------------------------------
# The OpenRouter route (t23): probe facts of 2026-10-11
# ---------------------------------------------------------------------------

OR_JEV = LLMModel.OR_JEV_1_13

# The success body OpenRouter returned to the live probe, verbatim in shape.
OR_BODY = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "closure": {"type": "noul", "noul": 0.99},
        "topic": {"type": "choice", "choice": "library",
                  "probabilities": {"library": 1, "school": 0, "unknown": 0}, "confidence": 1},
        "urgency": {"type": "score", "score": 0.65,
                    "legend": {"0": "not urgent", "1": "somewhat urgent", "2": "very urgent"},
                    "probabilities": {"0": 0.35, "1": 0.65, "2": 0}, "confidence": 0.47},
    },
    "usage": {"input_tokens": 415, "output_tokens": 69, "cost": 1.743e-05},
    "id": "gen-dec-test",
    "provider": "TypeSafe",
}

OR_QUESTIONS = {
    "closure": noul_question("Is the place closed?"),
    "topic": choice_question({"library": None, "school": None, "unknown": "cannot tell"},
                             "Which place is it about?"),
    "urgency": score_question(["not urgent", "somewhat urgent", "very urgent"], "How urgent?"),
}


@pytest.fixture
def or_svc(monkeypatch):
    monkeypatch.setattr(ts.time, "sleep", lambda s: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    s = TypeSafeService(OR_JEV)

    def use(*answers):
        s._client = FakeClient(answers)
        return s
    return use


class TestOpenRouterRoute:
    def test_row_is_a_route_twin_of_the_direct_row(self):
        assert OR_JEV.provider is Provider.OPENROUTER and OR_JEV.family == "jev"
        assert OR_JEV.model_id == "typesafe/jev-1.13-20260917"
        assert OR_JEV.input_price == 0.042 and OR_JEV.output_price == 0.0
        assert OR_JEV.weights == JEV.weights == "jev-1.13"
        assert OR_JEV.route_twins() == (JEV,) and JEV.route_twins() == (OR_JEV,)

    def test_factory_builds_the_typed_service_not_the_chat_one(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        s = LLMServiceFactory.create(OR_JEV)
        assert type(s) is TypeSafeService
        assert LLMServiceFactory.service_class_for(OR_JEV) is TypeSafeService
        assert LLMServiceFactory.service_class_for(LLMModel.OR_GLM_5).__name__ == "OpenRouterService"

    def test_key_comes_from_openrouter(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("TYPESAFE_API_KEY", "direct-key")
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            TypeSafeService(OR_JEV)

    def test_request_url_auth_and_pinned_model(self, or_svc):
        s = or_svc((200, OR_BODY))
        s.evaluate("The library on Main Street is closed today.", OR_QUESTIONS)
        url, body, headers = s._client.requests[0]
        assert url == "https://openrouter.ai/api/alpha/decisions"
        assert headers["Authorization"] == "Bearer or-test-key"
        assert headers["Content-Type"] == "application/json"
        assert body["model"] == "typesafe/jev-1.13-20260917"
        assert body["questions"]["topic"]["instructions"] == "Which place is it about?"

    def test_parses_the_probe_body_with_the_billed_cost(self, or_svc):
        rows = []
        BaseLLMService.set_usage_hook(lambda model, i, o, c, **kw: rows.append((model, i, o, c)))
        try:
            ev = or_svc((200, OR_BODY)).evaluate("x", OR_QUESTIONS)
        finally:
            BaseLLMService.clear_usage_hook()
        assert ev.model == "typesafe/jev-1.13-20260917"
        assert ev.answers["closure"] == NoulAnswer(0.99)
        assert ev.answers["topic"] == ChoiceAnswer(
            "library", 1.0, {"library": 1.0, "school": 0.0, "unknown": 0.0})
        u = ev.answers["urgency"]
        assert u.score == 0.65 and u.confidence == 0.47 and u.probabilities["1"] == 0.65
        assert u.legend["2"] == "very urgent"
        assert (ev.input_tokens, ev.output_tokens) == (415, 69)
        assert ev.cost_usd == 1.743e-05                   # billed, not recomputed
        assert rows == [(OR_JEV, 415, 69, 1.743e-05)]

    def test_cost_falls_back_to_tokens_without_a_billed_cost(self, or_svc):
        body = {**OR_BODY, "usage": {"input_tokens": 1_000_000, "output_tokens": 5}}
        assert or_svc((200, body)).evaluate("x", OR_QUESTIONS).cost_usd == pytest.approx(0.042)

    def test_a_question_without_instructions_is_refused_before_sending(self, or_svc):
        s = or_svc((200, OR_BODY))
        bare = {**OR_QUESTIONS, "topic": choice_question(["library", "school", "unknown"])}
        with pytest.raises(ValueError, match="'topic'.*instructions"):
            s.evaluate("x", bare)
        assert s._client.requests == []

    def test_400_is_a_value_error_without_the_user_id(self, or_svc):
        body = {"error": {"message": "[{\"code\": \"invalid_union\", \"message\": "
                                     "\"expected string, received undefined\"}]",
                          "code": 400},
                "user_id": "user_SECRET123"}
        s = or_svc((400, body))
        with pytest.raises(ValueError, match="invalid_union") as err:
            s.evaluate("x", OR_QUESTIONS)
        assert "user_SECRET123" not in str(err.value)
        assert len(s._client.requests) == 1               # validation is not retried

    def test_400_on_the_direct_route_stays_an_http_error(self, svc):
        with pytest.raises(ts.TypeSafeHTTPError, match="400"):
            svc((400, {"detail": "bad"})).evaluate("x", QUESTIONS)

    def test_an_unrecognized_json_error_body_is_not_echoed(self, or_svc):
        s = or_svc(*[(500, {"user_id": "user_SECRET123"})] * 4)
        with pytest.raises(ts.TypeSafeHTTPError) as err:
            s.evaluate("x", OR_QUESTIONS)
        assert "user_SECRET123" not in str(err.value)

    @pytest.mark.parametrize("code,exc", [
        (401, InvalidCredentialError), (402, CreditsExhaustedError), (404, FatalModelError)])
    def test_error_mapping_matches_the_direct_route(self, or_svc, code, exc):
        with pytest.raises(exc, match="OpenRouter"):
            or_svc((code, {"error": {"message": "no", "code": code}})).evaluate("x", OR_QUESTIONS)

    def test_retries_429_then_succeeds(self, or_svc):
        s = or_svc((429, {}, {"retry-after": "0"}), (502, {}), (200, OR_BODY))
        assert s.evaluate("x", OR_QUESTIONS).answers["closure"].noul == 0.99
        assert len(s._client.requests) == 3

    def test_personal_data_refused(self, or_svc):
        s = or_svc((200, OR_BODY))
        with pytest.raises(RetentionPolicyError, match="OpenRouter"):
            s.evaluate("about a person", OR_QUESTIONS, personal_data=True)
        assert s._client.requests == []

    def test_route_status_reads_the_openrouter_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        s = route_status(OR_JEV)
        assert not s.usable and "OPENROUTER_API_KEY" in s.reason
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        assert route_status(OR_JEV).usable

    def test_usable_routes_of_the_logical_model(self, monkeypatch):
        from llm_utils import usable_routes, routes_for
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        assert routes_for("jev-1.13") == {"typesafe", "openrouter"}
        assert usable_routes("jev-1.13") == (OR_JEV,)

    def test_broker_mode_does_not_report_the_openrouter_row_usable(self, monkeypatch):
        monkeypatch.setenv("LLM_UTILS_BROKER_URL", "http://127.0.0.1:9")
        s = route_status(OR_JEV)
        assert not s.usable and "not a broker route" in s.reason
        assert route_status(LLMModel.OR_GLM_5).usable     # the chat rows still are
