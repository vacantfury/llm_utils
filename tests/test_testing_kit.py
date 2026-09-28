"""llm_utils.testing: FakeService, use_fake_service, exception contract."""

import pytest

import llm_utils
from llm_utils import (
    BaseLLMService, CreditsExhaustedError, LLMModel, LLMServiceFactory, is_mechanism_error,
)
from llm_utils import testing
from llm_utils.testing import FakeService, check_exception_contract, use_fake_service


class TestFakeService:
    def test_string_and_default(self):
        assert FakeService().chat("hi") == "fake response"
        assert FakeService(responses="same").chat("x") == "same"

    def test_list_in_order_then_exhausted(self):
        fake = FakeService(responses=["a", "b"])
        assert [fake.chat("1"), fake.chat("2")] == ["a", "b"]
        with pytest.raises(AssertionError, match="no scripted response"):
            fake.chat("3")

    def test_dict_by_id_then_text(self):
        fake = FakeService(responses={"q1": "by id", "hello": "by text"}, default="d")
        out = fake.batch_chat([("q1", [("zzz", None)]), ("q2", [("hello", None)]),
                               ("q3", [("?", None)])], system_message="sys", temperature=0.3)
        assert out == [("q1", "by id"), ("q2", "by text"), ("q3", "d")]
        assert [c.conversation_id for c in fake.calls] == ["q1", "q2", "q3"]
        assert fake.calls[0].system_message == "sys" and fake.calls[0].kwargs["temperature"] == 0.3

    def test_callable(self):
        fake = FakeService(responses=lambda text, cid: text.upper())
        assert fake.chat("abc") == "ABC"

    def test_transport_error_becomes_a_mechanism_error(self):
        fake = FakeService(responses=[TimeoutError("slow")])
        assert is_mechanism_error(fake.chat("x"))

    def test_account_fatal_is_raised(self):
        fake = FakeService(responses=[CreditsExhaustedError("empty")])
        with pytest.raises(CreditsExhaustedError):
            fake.chat("x")

    def test_structured_returns_the_object(self):
        obj = object()
        assert FakeService(responses=[obj]).chat_structured("p", dict) is obj

    def test_usage_is_priced_from_the_registry(self):
        fake = FakeService(LLMModel.GPT_4O, responses="y" * 40)
        fake.chat("x" * 400)
        assert fake.total_usage.inference_count == 1
        assert fake.total_usage.cost > 0

    def test_usage_hook_called_but_no_default_ledger(self, monkeypatch):
        seen = []
        import llm_utils.call_ledger as ledger
        monkeypatch.setattr(ledger, "record_call",
                            lambda *a, **k: pytest.fail("wrote the default ledger"))
        FakeService(LLMModel.GPT_4O).chat("no hook installed")
        BaseLLMService.set_usage_hook(lambda model, i, o, c, **kw: seen.append((model, c)))
        try:
            FakeService(LLMModel.GPT_4O).chat("hook installed")
        finally:
            BaseLLMService.clear_usage_hook()
        assert seen and seen[0][0] is LLMModel.GPT_4O


class TestUseFakeService:
    def test_patches_and_restores(self):
        original = LLMServiceFactory.__dict__["create"]
        fake = FakeService(responses="ok")
        with use_fake_service(fake):
            svc = LLMServiceFactory.create("gpt-4o", label="judge")
            assert svc is fake and svc.model is LLMModel.GPT_4O and svc.usage_label == "judge"
            assert svc.chat("x") == "ok"
        assert LLMServiceFactory.__dict__["create"] is original

    def test_restores_after_an_error(self):
        original = LLMServiceFactory.__dict__["create"]
        with pytest.raises(RuntimeError):
            with use_fake_service(FakeService()):
                raise RuntimeError("boom")
        assert LLMServiceFactory.__dict__["create"] is original

    def test_builder_form(self):
        with use_fake_service(lambda model, **kw: FakeService(model, responses=model.model_id)):
            assert LLMServiceFactory.create(LLMModel.GPT_4O).chat("x") == "gpt-4o"


class TestExceptionContract:
    def test_holds(self):
        check_exception_contract()

    def test_detects_a_flattened_hierarchy(self, monkeypatch):
        broken = dict(testing.EXCEPTION_CONTRACT)
        broken[llm_utils.FatalModelError] = llm_utils.AccountFatalError
        monkeypatch.setattr(testing, "EXCEPTION_CONTRACT", broken)
        with pytest.raises(AssertionError, match="FatalModelError no longer subclasses"):
            check_exception_contract()

    def test_detects_a_missing_export(self, monkeypatch):
        monkeypatch.delattr(llm_utils, "SpendCapExceededError")
        with pytest.raises(AssertionError, match="SpendCapExceededError is not exported"):
            check_exception_contract()
