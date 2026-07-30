"""Usage-hook seam: fires on every recorded call, exceptions swallowed,
label pass-through. No network, no keys."""

import pytest

from llm_utils import LLMModel
from llm_utils.base_llm_service import BaseLLMService
from llm_utils.llm_services import OpenAIService


@pytest.fixture()
def service():
    svc = OpenAIService(LLMModel.GPT_5_NANO, api_key="test-key-not-real")
    yield svc
    BaseLLMService.clear_usage_hook()


def test_hook_receives_model_tokens_cost_label(service):
    seen = []
    BaseLLMService.set_usage_hook(
        lambda model, i, o, c, *, is_test, label: seen.append(
            (model, i, o, c, is_test, label)))
    service.usage_label = "audit-test"
    service._record_usage(100, 50, 0.5, is_test=True)
    assert seen == [(LLMModel.GPT_5_NANO, 100, 50, 0.5, True, "audit-test")]


def test_hook_exception_never_kills_the_call(service):
    def bad_hook(*a, **k):
        raise RuntimeError("ledger down")

    BaseLLMService.set_usage_hook(bad_hook)
    service._record_usage(1, 1, 0.0, is_test=False)  # must not raise
    assert service.total_usage.inference_count == 1


def test_is_test_splits_accumulators(service):
    service._record_usage(10, 10, 0.1, is_test=True)
    service._record_usage(20, 20, 0.2, is_test=False)
    assert service.total_usage.inference_count == 2
    assert service.algorithm_usage.inference_count == 1
    assert abs(service.algorithm_usage.cost - 0.2) < 1e-12


def test_clear_usage_hook(service):
    seen = []
    BaseLLMService.set_usage_hook(lambda *a, **k: seen.append(1))
    BaseLLMService.clear_usage_hook()
    service._record_usage(1, 1, 0.0, is_test=False)
    assert seen == []
