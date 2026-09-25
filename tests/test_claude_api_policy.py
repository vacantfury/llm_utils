"""Anthropic API calls are opt-in per process (LLM_UTILS_ALLOW_CLAUDE_API=1); other Claude routes are not covered."""
import pytest

from llm_utils import ClaudeAPINotAllowed, LLMModel, LLMServiceFactory, is_claude_model
from llm_utils.claude_api_policy import ALLOW_ENV
from llm_utils.llm_services.claude_service import ClaudeService


@pytest.fixture
def not_allowed(monkeypatch):
    monkeypatch.delenv(ALLOW_ENV, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")


def test_claude_models_are_recognized_on_every_route():
    assert is_claude_model(LLMModel.CLAUDE_OPUS_5)
    assert is_claude_model(LLMModel.BEDROCK_CLAUDE_HAIKU_4_5)
    assert not is_claude_model(LLMModel.from_string("gemini-3.5-flash-lite"))


def test_factory_refuses_claude_without_opt_in(not_allowed):
    with pytest.raises(ClaudeAPINotAllowed, match=ALLOW_ENV):
        LLMServiceFactory.create(LLMModel.CLAUDE_OPUS_5)
    with pytest.raises(ClaudeAPINotAllowed):
        LLMServiceFactory.create("claude-haiku-4-5-20251001")


def test_other_claude_routes_are_not_covered(not_allowed):
    """Bedrock and router-served Claude bill their own accounts: no opt-in needed."""
    from llm_utils.claude_api_policy import require_claude_api_allowed

    require_claude_api_allowed(LLMModel.BEDROCK_CLAUDE_HAIKU_4_5)   # no raise


def test_direct_construction_refuses_claude_without_opt_in(not_allowed):
    with pytest.raises(ClaudeAPINotAllowed):
        ClaudeService(LLMModel.CLAUDE_OPUS_5)


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_only_the_exact_opt_in_value_counts(not_allowed, monkeypatch, value):
    monkeypatch.setenv(ALLOW_ENV, value)
    with pytest.raises(ClaudeAPINotAllowed):
        LLMServiceFactory.create(LLMModel.CLAUDE_OPUS_5)


def test_opt_in_builds_the_claude_service(not_allowed, monkeypatch):
    monkeypatch.setenv(ALLOW_ENV, "1")
    assert isinstance(LLMServiceFactory.create(LLMModel.CLAUDE_OPUS_5), ClaudeService)


def test_non_claude_models_need_no_opt_in(not_allowed, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-not-real")
    LLMServiceFactory.create("gemini-3.5-flash-lite")
