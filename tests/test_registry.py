"""Guards consumer model lookup, realtime billing and installed-version provenance.

The Bedrock temperature-quirk regression remains covered; provider calls are faked."""

import collections
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm_utils import LLMModel
from llm_utils.base_llm_service import BaseLLMService
from llm_utils.llm_services import OpenAIService


def test_from_string_resolves_unique_ids_and_rejects_ambiguous():
    """Every uniquely-registered id must resolve; an id shared across serving
    routes (same weights local AND cluster) must raise instead of silently
    picking a provider."""
    by_id = collections.Counter(m.model_id for m in LLMModel)
    for m in LLMModel:
        if by_id[m.model_id] == 1:
            assert LLMModel.from_string(m.model_id) is m
        else:
            with pytest.raises(ValueError, match="Ambiguous"):
                LLMModel.from_string(m.model_id)
            # The enum NAME still resolves each route unambiguously.
            assert LLMModel.from_string(m.name) is m


@pytest.mark.parametrize("model_id,expected_cost", [
    ("gpt-6-luna", 0.002),
])
def test_gpt6_realtime_usage_cost(monkeypatch, model_id, expected_cost):
    """Bill 10K input + 2K output tokens through the real accounting path."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10_000, completion_tokens=2_000),
    )
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(
        "llm_utils.llm_services.openai_service.AsyncOpenAI", lambda **kw: client)
    # Keep synthetic usage out of any installed consumer's external ledger.
    monkeypatch.setattr(BaseLLMService, "_usage_hook", lambda *a, **kw: None)
    service = OpenAIService(
        LLMModel.from_string(model_id), api_key="test-key-not-real",
        use_batch_api=False,
    )
    assert service.chat("synthetic prompt", is_test=True) == "ok"
    assert service.total_usage.inference_count == 1
    assert service.total_usage.input_tokens == 10_000
    assert service.total_usage.output_tokens == 2_000
    assert service.total_usage.cost == pytest.approx(expected_cost)
    create.assert_awaited_once()
    params = create.call_args.kwargs
    assert params["model"] == model_id
    assert params["max_completion_tokens"] == service.max_tokens
    assert "max_tokens" not in params
    assert "temperature" not in params


def test_bedrock_claude_twins_share_temperature_quirk():
    """The same model rejects a custom temperature regardless of serving
    route — a Bedrock row must carry the quirk its direct-API twin has
    (the gap that 400'd every Bedrock Claude 5.x batch row)."""
    from llm_utils import ModelQuirk
    pairs = [
        (LLMModel.CLAUDE_SONNET_5, LLMModel.BEDROCK_CLAUDE_SONNET_5),
        (LLMModel.CLAUDE_OPUS_4_8, LLMModel.BEDROCK_CLAUDE_OPUS_4_8),
        (LLMModel.CLAUDE_FABLE_5, LLMModel.BEDROCK_CLAUDE_FABLE_5),
        (LLMModel.CLAUDE_FABLE_5_1, LLMModel.BEDROCK_CLAUDE_FABLE_5_1),
    ]
    for direct, bedrock in pairs:
        assert direct.has_quirk(ModelQuirk.NO_CUSTOM_TEMPERATURE), direct.name
        assert bedrock.has_quirk(ModelQuirk.NO_CUSTOM_TEMPERATURE), bedrock.name


class TestVersionProvenance:
    """`__version__` must equal the INSTALLED distribution version.

    A hardcoded literal here is a second source of truth beside
    pyproject.toml and it drifted unnoticed across two releases (v6.0.0 and
    v6.1.0 both shipped reading "5.2.0"). Consumers print this string into
    experiment provenance records, so a stale value silently misattributes
    which code produced a result.
    """

    def test_version_matches_installed_distribution(self):
        import importlib.metadata as md
        import llm_utils
        assert llm_utils.__version__ == md.version("llm_utils")
