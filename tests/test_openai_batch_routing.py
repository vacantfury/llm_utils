"""Native-batch auto-routing: cost estimator, tri-state override,
compatible-endpoint gating, cross-provider symmetry. Pure decision logic —
no network."""

import pytest

from llm_utils import LLMModel
from llm_utils.base_llm_service import (
    _EST_CHARS_PER_TOKEN,
    _EST_IMAGE_TOKENS,
)
from llm_utils.llm_services import (
    ClaudeService,
    DeepSeekService,
    GoogleService,
    OpenAIService,
)

KEY = {"api_key": "test-key-not-real"}


def _convs(n_convs: int, chars_each: int):
    """n single-message text conversations in seam format."""
    return [(f"c{i}", [("x" * chars_each, None)]) for i in range(n_convs)]


class TestEstimator:
    def test_text_only_math(self):
        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
        convs = _convs(2, 400)  # 2 × 100 est. input tokens
        est = svc._estimate_cost_usd(convs, max_tokens=1000)
        in_tok = 2 * (400 // _EST_CHARS_PER_TOKEN)
        out_tok = 2 * 1000
        expected = (
            in_tok * svc.model.input_price + out_tok * svc.model.output_price
        ) / 1_000_000
        assert est == pytest.approx(expected)

    def test_image_counts_flat_tokens(self):
        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
        convs = [("c0", [("x" * 40, "fake_image.png")])]
        est = svc._estimate_cost_usd(convs, max_tokens=0)
        in_tok = 40 // _EST_CHARS_PER_TOKEN + _EST_IMAGE_TOKENS
        assert est == pytest.approx(in_tok * svc.model.input_price / 1_000_000)


class TestRouting:
    def test_small_job_stays_realtime(self):
        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
        assert svc._route_to_native_batch(_convs(2, 100), max_tokens=100) is False

    def test_big_job_routes_to_batch(self):
        svc = OpenAIService(LLMModel.GPT_5, **KEY)
        assert svc._route_to_native_batch(_convs(2000, 4000), max_tokens=4096) is True

    def test_threshold_boundary_is_inclusive(self):
        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
        convs = _convs(1, 400)
        est = svc._estimate_cost_usd(convs, max_tokens=500)
        svc.batch_threshold_usd = est  # est >= threshold → batch
        assert svc._route_to_native_batch(convs, max_tokens=500) is True

    def test_force_true_and_false_override_estimate(self):
        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY, use_batch_api=True)
        assert svc._route_to_native_batch(_convs(1, 4), max_tokens=1) is True
        svc = OpenAIService(LLMModel.GPT_5, **KEY, use_batch_api=False)
        assert svc._route_to_native_batch(_convs(2000, 4000), max_tokens=4096) is False

    def test_custom_threshold_kwarg(self):
        svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY, batch_threshold_usd=0.0)
        assert svc._route_to_native_batch(_convs(1, 4), max_tokens=1) is True


class TestProviderSymmetry:
    """Claude and Google get the same auto-routing seam as OpenAI."""

    def test_claude_routes_by_threshold(self):
        svc = ClaudeService(LLMModel.CLAUDE_SONNET_5, **KEY)
        assert svc._supports_native_batch() is True
        assert svc._route_to_native_batch(_convs(2, 100), max_tokens=100) is False
        assert svc._route_to_native_batch(
            _convs(2000, 4000), max_tokens=4096) is True

    def test_google_routes_by_threshold(self):
        svc = GoogleService(LLMModel.GEMINI_2_5_FLASH_LITE, **KEY)
        assert svc._supports_native_batch() is True
        assert svc._route_to_native_batch(_convs(2, 100), max_tokens=100) is False
        assert svc._route_to_native_batch(
            _convs(20000, 4000), max_tokens=4096) is True

    def test_claude_thinking_models_get_output_headroom(self):
        from llm_utils.llm_services.claude_service import _THINKING_HEADROOM
        thinking = ClaudeService(LLMModel.CLAUDE_OPUS_5, **KEY)
        plain = ClaudeService(LLMModel.CLAUDE_SONNET_5, **KEY)
        assert thinking._output_budget(4096) == 4096 + _THINKING_HEADROOM
        assert plain._output_budget(4096) == 4096


class TestCompatibleEndpointGating:
    def test_compatible_endpoint_never_batches(self):
        svc = DeepSeekService(LLMModel.DEEPSEEK_V4_FLASH, **KEY)
        assert svc.BASE_URL is not None
        assert svc._supports_native_batch() is False
        assert svc._route_to_native_batch(_convs(2000, 4000), max_tokens=4096) is False

    def test_forced_true_still_gated_off(self):
        svc = DeepSeekService(
            LLMModel.DEEPSEEK_V4_FLASH, **KEY, use_batch_api=True)
        assert svc._route_to_native_batch(_convs(1, 4), max_tokens=1) is False

    def test_batch_trio_raises_on_compatible_endpoint(self):
        svc = DeepSeekService(LLMModel.DEEPSEEK_V4_FLASH, **KEY)
        with pytest.raises(NotImplementedError):
            svc.submit_batch_chat([("a", [("hi", None)])])
        with pytest.raises(NotImplementedError):
            svc.batch_chat_status("batch_x")
        with pytest.raises(NotImplementedError):
            svc.harvest_batch_chat("batch_x")

    def test_services_without_batch_api_get_clean_trio_error(self):
        # Base-class stubs: a duck-typed caller gets NotImplementedError,
        # never AttributeError.
        svc = DeepSeekService(LLMModel.DEEPSEEK_V4_FLASH, **KEY)
        for name in ("submit_batch_chat", "batch_chat_status",
                     "harvest_batch_chat"):
            assert hasattr(svc, name)
