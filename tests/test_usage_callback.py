"""Default usage record: one dict per call to the process-wide callback a caller
registers (``llm_utils.set_usage_callback``), only when no usage hook is
installed. Provider clients are stubbed; no network, no keys, no real ledger."""

from types import SimpleNamespace

import pytest

from llm_utils import LLMModel, LLMServiceFactory, set_usage_callback, usage_callback
from llm_utils.base_llm_service import BaseLLMService
from llm_utils.llm_services import OpenAIService


def _stub_transport(svc, in_tok=100, out_tok=50):
    async def create(**_params):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"),
                                     finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok))
    svc.async_client.chat.completions.create = create
    return svc


@pytest.fixture(autouse=True)
def clean_state():
    BaseLLMService.clear_usage_hook()
    set_usage_callback(None)
    yield
    BaseLLMService.clear_usage_hook()
    set_usage_callback(None)


@pytest.fixture()
def records():
    out = []
    set_usage_callback(out.append)
    return out


@pytest.mark.parametrize("how, launch_point, source", [
    ("attribute", "demo.site", "explicit"),
    ("factory", "demo.factory", "explicit"),
    ("unlabeled", __name__, "module"),
])
def test_a_stubbed_call_reports_one_record(records, how, launch_point, source):
    if how == "factory":
        svc = LLMServiceFactory.create(LLMModel.GPT_5_NANO, api_key="[redacted]",
                                       launch_point="demo.factory")
    else:
        svc = OpenAIService(LLMModel.GPT_5_NANO, api_key="[redacted]")
        if how == "attribute":
            svc.launch_point = "demo.site"
    svc.usage_label = "audit"
    assert _stub_transport(svc).chat("hi") == "ok"
    assert records == [{
        "launch_point": launch_point, "launch_point_source": source,
        "caller_file": __file__, "surface": "llm_api", "host": "api",
        "model": LLMModel.GPT_5_NANO.model_id, "repo": None, "purpose": "audit",
        "input_tokens": 100, "output_tokens": 50,
        "cost_usd": pytest.approx(svc.total_usage.cost), "is_test": False,
        "status": "ok", "error_class": None,
    }]


def test_no_callback_reports_nothing_and_the_call_works(monkeypatch):
    monkeypatch.setattr(usage_callback, "caller",
                        lambda: pytest.fail("built a record with no callback"))
    svc = _stub_transport(OpenAIService(LLMModel.GPT_5_NANO, api_key="[redacted]"))
    assert svc.chat("hi") == "ok"
    assert svc.total_usage.inference_count == 1
    assert usage_callback.report(svc, 1, 1, 0.0) is None


def test_installed_hook_suppresses_the_callback(records):
    seen = []
    BaseLLMService.set_usage_hook(lambda *a, **k: seen.append(1))
    svc = OpenAIService(LLMModel.GPT_5_NANO, api_key="[redacted]")
    svc._record_usage(1, 1, 0.0, is_test=False)
    svc._record_failure(TimeoutError())
    assert seen == [1, 1] and records == []


def test_failing_callback_never_raises_and_warns_once(caplog):
    def boom(_record):
        raise RuntimeError("disk full")

    set_usage_callback(boom)
    svc = OpenAIService(LLMModel.GPT_5_NANO, api_key="[redacted]")
    with caplog.at_level("WARNING"):
        svc._record_usage(1, 1, 0.0, is_test=False)
        svc._record_failure(RuntimeError("x"))
    assert svc.total_usage.inference_count == 1
    assert sum("usage callback failed" in r.getMessage() for r in caplog.records) == 1


def test_chainable_from_a_hook(records):
    svc = OpenAIService(LLMModel.GPT_5_NANO, api_key="[redacted]")
    svc.launch_point = "demo.chained"
    BaseLLMService.set_usage_hook(
        lambda model, i, o, c, *, is_test, label: usage_callback.report(
            svc, i, o, c, is_test=is_test, label=label))
    svc._record_usage(3, 4, 0.5, is_test=True)
    assert [(r["launch_point"], r["is_test"]) for r in records] == [("demo.chained", True)]


def test_a_non_callable_is_refused():
    with pytest.raises(TypeError):
        set_usage_callback("not a function")
