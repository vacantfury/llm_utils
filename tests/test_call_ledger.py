"""Default call-ledger record: fires only when no hook is installed and the
optional agent_manager package is importable. A fake agent_manager module is
injected; no network, no keys, no real ledger."""

import sys
import types

import pytest

from llm_utils import LLMModel, LLMServiceFactory, call_ledger
from llm_utils.base_llm_service import BaseLLMService
from llm_utils.llm_services import OpenAIService


@pytest.fixture()
def fake_ledger(monkeypatch):
    rows = []
    pkg = types.ModuleType("agent_manager")
    ledger = types.ModuleType("agent_manager.ledger")
    ledger.record = lambda **fields: rows.append(fields) or fields
    pkg.ledger = ledger
    monkeypatch.setitem(sys.modules, "agent_manager", pkg)
    monkeypatch.setitem(sys.modules, "agent_manager.ledger", ledger)
    call_ledger.census_id.cache_clear()
    call_ledger.repo_of.cache_clear()
    yield rows
    call_ledger.census_id.cache_clear()
    call_ledger.repo_of.cache_clear()


@pytest.fixture()
def service():
    svc = OpenAIService(LLMModel.GPT_5_NANO, api_key="test-key-not-real")
    BaseLLMService.clear_usage_hook()
    yield svc
    BaseLLMService.clear_usage_hook()


def test_default_records_one_row_with_explicit_launch_point(fake_ledger, service):
    service.launch_point = "demo.site"
    service.usage_label = "audit"
    service._record_usage(100, 50, 0.25, is_test=False)
    assert fake_ledger == [{
        "launch_point": "demo.site", "surface": "llm_api", "host": "api",
        "model": LLMModel.GPT_5_NANO.model_id, "repo": None, "purpose": "audit",
        "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.25, "status": "ok",
    }]


def test_factory_launch_point_kwarg(fake_ledger):
    svc = LLMServiceFactory.create(LLMModel.GPT_5_NANO, api_key="test-key-not-real",
                                   launch_point="demo.factory")
    assert svc.launch_point == "demo.factory"
    svc._record_usage(1, 1, 0.0, is_test=False)
    assert fake_ledger[0]["launch_point"] == "demo.factory"


def test_unlabeled_falls_back_to_calling_module(fake_ledger, service):
    # no census module in the fake package -> resolution fails -> module name
    service._record_usage(1, 2, 0.0, is_test=False)
    assert len(fake_ledger) == 1
    assert fake_ledger[0]["launch_point"] == __name__


def test_installed_hook_suppresses_default(fake_ledger, service):
    seen = []
    BaseLLMService.set_usage_hook(lambda *a, **k: seen.append(1))
    service._record_usage(1, 1, 0.0, is_test=False)
    assert seen == [1] and fake_ledger == []


def test_absent_agent_manager_is_inert(monkeypatch, service):
    monkeypatch.setitem(sys.modules, "agent_manager", None)  # import raises
    service._record_usage(1, 1, 0.0, is_test=False)  # must not raise
    assert service.total_usage.inference_count == 1
    assert call_ledger.available() is False


def test_recorder_failure_never_raises(monkeypatch, service):
    pkg = types.ModuleType("agent_manager")
    ledger = types.ModuleType("agent_manager.ledger")

    def boom(**_):
        raise RuntimeError("disk full")

    ledger.record = boom
    pkg.ledger = ledger
    monkeypatch.setitem(sys.modules, "agent_manager", pkg)
    monkeypatch.setitem(sys.modules, "agent_manager.ledger", ledger)
    service._record_usage(1, 1, 0.0, is_test=False)
    assert service.total_usage.inference_count == 1


def test_chainable_from_a_hook(fake_ledger, service):
    service.launch_point = "demo.chained"
    BaseLLMService.set_usage_hook(
        lambda model, i, o, c, *, is_test, label: call_ledger.record_call(
            service, i, o, c, is_test=is_test, label=label))
    service._record_usage(3, 4, 0.5, is_test=False)
    assert [r["launch_point"] for r in fake_ledger] == ["demo.chained"]


def test_census_resolution_by_file(monkeypatch, tmp_path, fake_ledger, service):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    target = repo / "pkg" / "site.py"
    target.write_text("")
    other = repo / "pkg" / "other.py"
    other.write_text("")
    census = types.ModuleType("agent_manager.census")
    rows = {str(target): "repo.site", str(other): "repo.dir"}
    census.resolve = lambda p: {"repo": "repo", "launch_point": rows.get(str(p))}
    monkeypatch.setitem(sys.modules, "agent_manager.census", census)
    sys.modules["agent_manager"].census = census
    assert call_ledger.census_id(str(target)) == "repo.site"
    assert call_ledger.census_id(str(other)) == "repo.dir"
