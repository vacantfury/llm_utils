"""Factory: provider dispatch, config-loader merge precedence, label wiring,
SLURM manager guard. No network, no keys."""

import pytest

from llm_utils import LLMModel, LLMServiceFactory, Provider
from llm_utils.llm_services import OpenAIService


@pytest.fixture(autouse=True)
def _clean_factory_state():
    yield
    LLMServiceFactory.clear_config_loader()
    LLMServiceFactory.clear_server_manager()


def test_create_by_enum_and_by_string():
    svc = LLMServiceFactory.create(LLMModel.GPT_5_NANO, api_key="test-key")
    assert isinstance(svc, OpenAIService)
    svc2 = LLMServiceFactory.create(
        LLMModel.GPT_5_NANO.model_id, api_key="test-key")
    assert svc2.model is LLMModel.GPT_5_NANO


def test_every_provider_has_a_registered_service():
    for provider in Provider:
        assert LLMServiceFactory.is_provider_supported(provider), provider


def test_caller_kwargs_beat_loader_defaults():
    LLMServiceFactory.set_config_loader(
        lambda model: {"temperature": 0.7, "max_tokens": 111, "model": "junk"})
    svc = LLMServiceFactory.create(
        LLMModel.GPT_5_NANO, api_key="test-key", temperature=0.2)
    assert svc.temperature == 0.2      # caller wins
    assert svc.max_tokens == 111       # loader default survives
    # loader's stray 'model' key must not collide with the positional arg
    assert svc.model is LLMModel.GPT_5_NANO


def test_label_sets_usage_label():
    svc = LLMServiceFactory.create(
        LLMModel.GPT_5_NANO, api_key="test-key", label="ledger-tag")
    assert svc.usage_label == "ledger-tag"


def test_cluster_model_without_manager_raises():
    cluster_models = [m for m in LLMModel
                      if m.provider is Provider.SLURM_CLUSTER]
    if not cluster_models:
        pytest.skip("no SLURM_CLUSTER models registered")
    with pytest.raises(RuntimeError, match="No ClusterModelServerManager"):
        LLMServiceFactory.create(cluster_models[0])
