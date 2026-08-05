"""Account-status seam: pure spend math + the balance-fetch contract.

Offline like the rest of the suite: provider HTTP is monkeypatched, services
are built with dummy keys.
"""
from datetime import datetime, timedelta

import pytest

from llm_utils.account_status import AccountStatus, burn_rate, days_to_empty


def _t(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 8, day, hour)


# ── burn_rate ─────────────────────────────────────────────────────────


def test_burn_rate_simple_decline():
    snaps = [(_t(1), 100.0), (_t(2), 90.0), (_t(3), 80.0)]
    assert burn_rate(snaps) == pytest.approx(10.0)


def test_burn_rate_ignores_topups():
    # 100→90 (spend 10), top-up to 190, 190→170 (spend 20): 30 over 3 days.
    snaps = [(_t(1), 100.0), (_t(2), 90.0), (_t(3), 190.0), (_t(4), 170.0)]
    assert burn_rate(snaps) == pytest.approx(10.0)


def test_burn_rate_unsorted_input():
    snaps = [(_t(3), 80.0), (_t(1), 100.0), (_t(2), 90.0)]
    assert burn_rate(snaps) == pytest.approx(10.0)


def test_burn_rate_insufficient_or_degenerate():
    assert burn_rate([]) is None
    assert burn_rate([(_t(1), 100.0)]) is None
    assert burn_rate([(_t(1), 100.0), (_t(1), 100.0)]) is None  # zero elapsed


def test_burn_rate_sub_day_resolution():
    snaps = [(_t(1, 0), 100.0), (_t(1, 12), 95.0)]  # 5 in half a day
    assert burn_rate(snaps) == pytest.approx(10.0)


# ── days_to_empty ─────────────────────────────────────────────────────


def test_days_to_empty_basic():
    assert days_to_empty(50.0, 10.0) == pytest.approx(5.0)


def test_days_to_empty_unknowns():
    assert days_to_empty(None, 10.0) is None
    assert days_to_empty(50.0, None) is None
    assert days_to_empty(50.0, 0.0) is None
    assert days_to_empty(50.0, -1.0) is None


# ── AccountStatus / base contract ─────────────────────────────────────


def test_account_status_to_dict_roundtrip():
    st = AccountStatus(provider="X", supported=True, balance=1.5, currency="USD")
    d = st.to_dict()
    assert d["provider"] == "X" and d["balance"] == 1.5 and d["supported"]


def test_unsupported_service_reports_unsupported():
    from llm_utils import LLMModel, GoogleService

    svc = GoogleService(LLMModel.GEMINI_2_5_FLASH, api_key="dummy")
    st = svc.get_account_status()
    assert st.supported is False
    assert st.balance is None


# ── provider balance fetches (HTTP faked; payloads from the providers'
#    documented samples, verified 2026-08-04) ──────────────────────────


def _fake_get(payload_by_url):
    calls = []

    def fake(url, headers=None, timeout=None):
        calls.append((url, headers))

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return payload_by_url[url]

        return _Resp()

    return fake, calls


def test_deepseek_balance(monkeypatch):
    from llm_utils import LLMModel, DeepSeekService
    from llm_utils.llm_services import openai_compatible_services as mod

    svc = DeepSeekService(LLMModel.DEEPSEEK_V4_FLASH, api_key="dummy")
    url = "https://api.deepseek.com/user/balance"
    fake, calls = _fake_get({url: {
        "is_available": True,
        "balance_infos": [{
            "currency": "CNY", "total_balance": "110.00",
            "granted_balance": "10.00", "topped_up_balance": "100.00",
        }],
    }})
    monkeypatch.setattr(mod.httpx2, "get", fake)
    st = svc.get_account_status()
    assert st.supported and st.provider == "DeepSeek"
    assert st.balance == pytest.approx(110.0)
    assert st.currency == "CNY"
    assert st.total_granted == pytest.approx(10.0)
    assert calls[0][1]["Authorization"] == "Bearer dummy"


def test_moonshot_balance(monkeypatch):
    from llm_utils import LLMModel, MoonshotService
    from llm_utils.llm_services import openai_compatible_services as mod

    svc = MoonshotService(LLMModel.KIMI_K3, api_key="dummy")
    url = "https://api.moonshot.ai/v1/users/me/balance"
    fake, _ = _fake_get({url: {
        "code": 0, "status": True, "scode": "0x0",
        "data": {"available_balance": 49.58894,
                 "voucher_balance": 46.58893, "cash_balance": 3.00001},
    }})
    monkeypatch.setattr(mod.httpx2, "get", fake)
    st = svc.get_account_status()
    assert st.supported and st.balance == pytest.approx(49.58894)
    assert st.currency == "USD"


def test_moonshot_balance_rejected(monkeypatch):
    from llm_utils import LLMModel, MoonshotService
    from llm_utils.llm_services import openai_compatible_services as mod

    svc = MoonshotService(LLMModel.KIMI_K3, api_key="dummy")
    url = "https://api.moonshot.ai/v1/users/me/balance"
    fake, _ = _fake_get({url: {"code": 1, "status": False}})
    monkeypatch.setattr(mod.httpx2, "get", fake)
    with pytest.raises(ValueError, match="balance query rejected"):
        svc.get_account_status()


def test_openrouter_key_fallback(monkeypatch):
    from llm_utils import LLMModel, OpenRouterService
    from llm_utils.llm_services import openai_compatible_services as mod

    monkeypatch.delenv("OPENROUTER_MANAGEMENT_KEY", raising=False)
    svc = OpenRouterService(LLMModel.OR_DEEPSEEK_V4_FLASH, api_key="dummy")
    url = "https://openrouter.ai/api/v1/key"
    fake, calls = _fake_get({url: {
        "data": {"label": "k", "limit": None, "limit_remaining": None,
                 "usage": 25.75, "is_free_tier": False},
    }})
    monkeypatch.setattr(mod.httpx2, "get", fake)
    st = svc.get_account_status()
    assert st.supported and st.balance is None
    assert st.total_used == pytest.approx(25.75)
    assert calls[0][1]["Authorization"] == "Bearer dummy"


def test_openrouter_management_credits(monkeypatch):
    from llm_utils import LLMModel, OpenRouterService
    from llm_utils.llm_services import openai_compatible_services as mod

    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "mgmt")
    svc = OpenRouterService(LLMModel.OR_DEEPSEEK_V4_FLASH, api_key="dummy")
    url = "https://openrouter.ai/api/v1/credits"
    fake, calls = _fake_get({url: {
        "data": {"total_credits": 100.5, "total_usage": 25.75},
    }})
    monkeypatch.setattr(mod.httpx2, "get", fake)
    st = svc.get_account_status()
    assert st.balance == pytest.approx(74.75)
    assert st.total_granted == pytest.approx(100.5)
    assert calls[0][1]["Authorization"] == "Bearer mgmt"


def test_capability_markers():
    from llm_utils import (
        ClaudeService, DeepSeekService, MoonshotService, OpenAIService,
        OpenRouterService, XAIService, ZAIService,
    )

    assert DeepSeekService.BALANCE_QUERY_VIA == "api_key"
    assert MoonshotService.BALANCE_QUERY_VIA == "api_key"
    assert OpenRouterService.BALANCE_QUERY_VIA == "api_key"
    assert XAIService.BALANCE_QUERY_VIA == "management_key"
    assert ZAIService.BALANCE_QUERY_VIA is None
    assert OpenAIService.BALANCE_QUERY_VIA is None
    assert ClaudeService.BALANCE_QUERY_VIA is None
