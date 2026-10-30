"""Mainland-China endpoints are opt-in per process (LLM_UTILS_ALLOW_PRC_ENDPOINTS=1); US routes are not covered."""
import pytest

from llm_utils import LLMModel, LLMServiceFactory, PRCEndpointNotAllowed, is_prc_route
from llm_utils.prc_endpoint_policy import ALLOW_ENV, require_prc_allowed


@pytest.fixture
def not_allowed(monkeypatch):
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    for key in ("DEEPSEEK_API_KEY", "ZAI_API_KEY", "MOONSHOT_API_KEY"):
        monkeypatch.setenv(key, "test-key")


def test_prc_routes_are_recognized():
    assert is_prc_route(LLMModel.from_string("deepseek-v4-flash"))
    assert is_prc_route(LLMModel.from_string("glm-5"))
    assert is_prc_route(LLMModel.from_string("kimi-k2.6"))
    assert not is_prc_route(LLMModel.from_string("deepseek/deepseek-v4-flash"))   # OpenRouter, US


def test_request_is_refused_before_anything_is_sent(not_allowed):
    service = LLMServiceFactory.create("deepseek-v4-flash")   # building is allowed (balance checks)

    class NoNetwork:
        def __getattr__(self, name):
            raise AssertionError("a request reached the provider client")

    service.async_client = NoNetwork()   # if the guard failed, the call would hit this, never the network
    with pytest.raises(PRCEndpointNotAllowed, match=ALLOW_ENV):
        service.chat("hello")
    with pytest.raises(PRCEndpointNotAllowed):
        service.batch_chat([[{"role": "user", "content": "hello"}]])


def test_us_routes_are_not_covered(not_allowed):
    require_prc_allowed(LLMModel.from_string("deepseek/deepseek-v4-flash"))   # no raise


def test_opt_in_allows(monkeypatch):
    monkeypatch.setenv(ALLOW_ENV, "1")
    require_prc_allowed(LLMModel.from_string("deepseek-v4-flash"))   # no raise


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_only_exact_one_opts_in(monkeypatch, value):
    monkeypatch.setenv(ALLOW_ENV, value)
    with pytest.raises(PRCEndpointNotAllowed):
        require_prc_allowed(LLMModel.from_string("glm-5"))
