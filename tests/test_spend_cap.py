"""Run spend cap (t12): admission before sending, typed refusal, config
precedence, API spend only; plus the expected_output_tokens estimate hint (t2).
Offline: FakeService and constructed services with dummy keys."""

import pytest

from llm_utils import (
    AccountFatalError, LLMModel, SpendCapExceededError, config, max_usd_per_run,
    reset_run_spend, spend_status,
)
from llm_utils import spend
from llm_utils.llm_services import OpenAIService
from llm_utils.testing import FakeService

PAID = LLMModel.GPT_4O          # $2.50 / $10.00 per 1M tokens
SELF = LLMModel.LLAMA3_8B_CLUSTER


def _est(svc, text="x" * 400, n=1, **kw):
    return svc._estimate_cost_usd([(str(i), [(text, None)]) for i in range(n)],
                                  kw.get("max_tokens", svc.max_tokens),
                                  kw.get("expected_output_tokens"))


class TestCapSource:
    def test_packaged_default(self, monkeypatch):
        monkeypatch.delenv(spend.MAX_USD_ENV)
        assert max_usd_per_run() == 5.0

    @pytest.mark.parametrize("raw,cap", [("none", None), ("OFF", None), ("12.5", 12.5), ("0", 0.0)])
    def test_env_wins(self, monkeypatch, raw, cap):
        monkeypatch.setenv(spend.MAX_USD_ENV, raw)
        assert max_usd_per_run() == cap

    @pytest.mark.parametrize("raw", ["lots", "-1", "nan"])
    def test_bad_env_raises(self, monkeypatch, raw):
        monkeypatch.setenv(spend.MAX_USD_ENV, raw)
        with pytest.raises(ValueError):
            max_usd_per_run()

    def test_project_file_and_null(self, monkeypatch, tmp_path):
        monkeypatch.delenv(spend.MAX_USD_ENV)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "llm_utils.yaml").write_text("spend_cap:\n  max_usd_per_run: 40\n")
        config.reload_config()
        assert max_usd_per_run() == 40
        (tmp_path / "llm_utils.yaml").write_text("spend_cap:\n  max_usd_per_run: null\n")
        config.reload_config()
        assert max_usd_per_run() is None

    def test_configure(self, monkeypatch):
        monkeypatch.delenv(spend.MAX_USD_ENV)
        config.configure(spend_cap={"max_usd_per_run": 0.5})
        assert max_usd_per_run() == 0.5


class TestAdmission:
    def test_refused_before_sending(self, monkeypatch):
        monkeypatch.setenv(spend.MAX_USD_ENV, "0.000001")
        fake = FakeService(PAID, responses="ok")
        with pytest.raises(SpendCapExceededError) as err:
            fake.chat("hello there")
        assert fake.calls == []                       # nothing reached the "provider"
        assert isinstance(err.value, AccountFatalError)  # runners abort the run
        assert err.value.cap == 0.000001 and err.value.estimate > 0
        assert spend_status().reserved == 0

    def test_spend_accumulates_until_the_cap(self, monkeypatch):
        fake = FakeService(PAID, responses="y" * 400, max_tokens=100)
        per_call = _est(fake, "x" * 400)
        monkeypatch.setenv(spend.MAX_USD_ENV, str(per_call * 2.5))
        fake.chat("x" * 400)
        fake.chat("x" * 400)
        st = spend_status()
        assert st.spent > 0 and st.reserved == 0
        with pytest.raises(SpendCapExceededError):
            fake.chat("x" * 400)
        assert len(fake.calls) == 2

    def test_self_served_routes_are_never_capped(self, monkeypatch):
        monkeypatch.setenv(spend.MAX_USD_ENV, "0")
        fake = FakeService(SELF, responses="ok")
        assert fake.chat("hi") == "ok"

    def test_nested_call_is_admitted_once(self, monkeypatch):
        monkeypatch.setenv(spend.MAX_USD_ENV, "100")
        seen = []
        fake = FakeService(PAID, responses=lambda text, cid: seen.append(spend_status().reserved) or "ok")
        fake.chat("hello")                            # chat -> batch_chat, both guarded
        assert seen[0] == pytest.approx(_est(fake, "hello"))

    def test_real_service_refuses_without_touching_the_client(self, monkeypatch):
        monkeypatch.setenv(spend.MAX_USD_ENV, "0")
        svc = OpenAIService(PAID, api_key="test-key")

        class Boom:
            def __getattr__(self, name):
                raise AssertionError("the provider client was used")

        svc.async_client = Boom()
        with pytest.raises(SpendCapExceededError):
            svc.batch_chat([("a", [("hi", None)])])

    def test_submitted_batches_hold_their_estimate_until_harvested(self, monkeypatch):
        monkeypatch.setenv(spend.MAX_USD_ENV, "100")

        class BatchFake(FakeService):
            done = False

            def submit_batch_chat(self, conversations, system_message=None, **kwargs):
                return "batch-1"

            def harvest_batch_chat(self, batch_id, *, is_test=False):
                return [("a", "ok")] if self.done else None

        fake = BatchFake(PAID)
        assert fake.submit_batch_chat([("a", [("x" * 400, None)])]) == "batch-1"
        held = spend_status().committed
        assert held == pytest.approx(_est(fake, "x" * 400) * fake.BATCH_COST_DISCOUNT)
        assert fake.harvest_batch_chat("batch-1") is None
        assert spend_status().committed == held
        fake.done = True
        fake.harvest_batch_chat("batch-1")
        assert spend_status().committed == 0

    def test_reset(self):
        spend.record(3.0)
        assert spend_status().spent == 3.0
        reset_run_spend()
        assert spend_status().spent == 0


class TestOutputHint:
    """t2: the estimate's per-request output is expected_output_tokens when
    given, else max_tokens (the default, so existing routing is unchanged)."""

    def test_estimate_uses_the_hint(self):
        fake = FakeService(PAID, max_tokens=16384)
        full = _est(fake, n=10)
        hinted = _est(fake, n=10, expected_output_tokens=500)
        assert hinted < full / 20

    def test_openai_routing_default_unchanged_hint_moves_it(self):
        convs = [(str(i), [("x" * 4000, None)]) for i in range(40)]
        svc = OpenAIService(LLMModel.GPT_5_MINI, api_key="k", max_tokens=16384)
        assert svc._route_to_native_batch(convs, 16384, svc._output_hint({})) is True
        assert svc._route_to_native_batch(
            convs, 16384, svc._output_hint({"expected_output_tokens": 300})) is False
        hinted = OpenAIService(LLMModel.GPT_5_MINI, api_key="k", max_tokens=16384,
                               expected_output_tokens=300)
        assert hinted._route_to_native_batch(convs, 16384, hinted._output_hint({})) is False

    def test_bad_hint_raises(self):
        with pytest.raises(ValueError):
            FakeService(PAID)._output_hint({"expected_output_tokens": -1})

    def test_hint_tightens_the_cap_check(self, monkeypatch):
        fake = FakeService(PAID, responses="ok", max_tokens=16384)
        monkeypatch.setenv(spend.MAX_USD_ENV, str(_est(fake, "hi") / 2))
        with pytest.raises(SpendCapExceededError):
            fake.chat("hi")
        assert fake.chat("hi", expected_output_tokens=10) == "ok"
