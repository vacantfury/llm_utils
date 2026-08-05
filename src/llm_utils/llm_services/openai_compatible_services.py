"""
OpenAI-compatible third-party services — DeepSeek, Z.AI (GLM), xAI (Grok),
Moonshot (Kimi), OpenRouter.

All expose OpenAI-compatible ``/v1/chat/completions`` endpoints, so they
subclass ``OpenAIService`` and only override the three class attributes; every
request/batch path is inherited unchanged.

PRIVACY (data jurisdiction): DeepSeek, Z.AI, and Moonshot are DIRECT MAINLAND
endpoints — their data is processed under PRC jurisdiction per the providers'
own policies. Consumers must route ONLY zero-personal-data bulk work through
them (LLM-judge calls / evals / sweeps over PUBLIC benchmark responses) and
NEVER personal data. xAI is US jurisdiction, same posture as the other US
frontier providers. OpenRouter is a US aggregator over hosted open weights —
with a zero-data-retention routing policy enabled on the account it is the
US-jurisdiction route to Chinese open-weight models. This package is transport
only; each consumer enforces its own routing policy on top.
"""
import os

import httpx2

from ..account_status import AccountStatus
from ..constants import (
    DEEPSEEK_API_URL, MOONSHOT_API_URL, OPENROUTER_API_URL, XAI_API_URL, ZAI_API_URL,
)
from .openai_service import OpenAIService

# Fail-safe default for the one-shot balance GETs below; callers needing a
# different budget pass `timeout=` to get_account_status via _fetch overrides.
_ACCOUNT_TIMEOUT_S = 30.0


def _bearer_get(url: str, api_key: str, timeout: float = _ACCOUNT_TIMEOUT_S) -> dict:
    """One authenticated GET, JSON out. Raises on HTTP/network errors —
    a monitor must see a failed balance check as failed, never as empty."""
    resp = httpx2.get(
        url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()


class DeepSeekService(OpenAIService):
    """DeepSeek platform (deepseek-v4-*). Mainland endpoint — no personal data."""

    API_KEY_ENV = "DEEPSEEK_API_KEY"
    BASE_URL = DEEPSEEK_API_URL
    SERVICE_NAME = "DeepSeek"
    BALANCE_QUERY_VIA = "api_key"

    def _fetch_account_status(self) -> AccountStatus:
        # GET /user/balance (NOT under /v1) — amounts are STRINGS, currency
        # can be CNY or USD (api-docs.deepseek.com/api/get-user-balance).
        payload = _bearer_get(f"{self.BASE_URL}/user/balance", self.api_key)
        infos = payload.get("balance_infos") or []
        first = infos[0] if infos else {}
        return AccountStatus(
            provider=self.SERVICE_NAME,
            supported=True,
            balance=float(first["total_balance"]) if "total_balance" in first else None,
            currency=first.get("currency"),
            total_granted=(
                float(first["granted_balance"]) if "granted_balance" in first else None
            ),
            raw=payload,
        )


class ZAIService(OpenAIService):
    """Z.AI open platform (GLM-*). Mainland endpoint — no personal data."""

    API_KEY_ENV = "ZAI_API_KEY"
    BASE_URL = ZAI_API_URL
    SERVICE_NAME = "Z.AI"
    # No balance endpoint in the published API reference (verified 2026-08-04
    # against docs.z.ai — billing is dashboard-only). Consumers estimate from
    # their usage ledger; CreditsExhaustedError ("insufficient balance", 1113)
    # is the reactive backstop.
    BALANCE_QUERY_VIA = None


class XAIService(OpenAIService):
    """xAI Grok family (grok-*). US jurisdiction, OpenAI-compatible."""

    API_KEY_ENV = "XAI_API_KEY"
    BASE_URL = XAI_API_URL
    SERVICE_NAME = "xAI"
    # Balance exists but only on the separate Management API host
    # (management-api.x.ai .../prepaid/balance) with a management key —
    # a different credential + team id. Not implemented; see repo TODO.
    BALANCE_QUERY_VIA = "management_key"


class MoonshotService(OpenAIService):
    """Moonshot Kimi family (kimi-*). Mainland endpoint — no personal data."""

    API_KEY_ENV = "MOONSHOT_API_KEY"
    BASE_URL = MOONSHOT_API_URL
    SERVICE_NAME = "Moonshot"
    BALANCE_QUERY_VIA = "api_key"

    def _fetch_account_status(self) -> AccountStatus:
        # GET /v1/users/me/balance (platform.kimi.ai/docs/api/balance).
        # available_balance = voucher + cash, USD; cash may go negative.
        payload = _bearer_get(f"{self.BASE_URL}/users/me/balance", self.api_key)
        if not payload.get("status", False):
            raise ValueError(
                f"{self.SERVICE_NAME} balance query rejected: {payload!r}"
            )
        data = payload.get("data") or {}
        return AccountStatus(
            provider=self.SERVICE_NAME,
            supported=True,
            balance=data.get("available_balance"),
            currency="USD",
            raw=payload,
        )


class OpenRouterService(OpenAIService):
    """OpenRouter aggregator (openrouter.ai) — hosted open-weight models.

    US jurisdiction. Data-retention behavior depends on the ACCOUNT's routing
    policy: with zero-data-retention routing enabled, requests reach only
    non-retaining hosts (see the ``Provider.OPENROUTER`` registry note).
    """

    API_KEY_ENV = "OPENROUTER_API_KEY"
    BASE_URL = OPENROUTER_API_URL
    SERVICE_NAME = "OpenRouter"
    BALANCE_QUERY_VIA = "api_key"
    # Optional upgrade: account-level remaining credits need a MANAGEMENT key
    # (GET /credits rejects inference keys). If this env var is set, the fetch
    # uses it; otherwise it falls back to GET /key (normal key), which only
    # reports this key's lifetime usage + its spend-cap remainder (null when
    # the key has no cap) — balance stays None then.
    MANAGEMENT_KEY_ENV = "OPENROUTER_MANAGEMENT_KEY"

    def _fetch_account_status(self) -> AccountStatus:
        mgmt_key = os.getenv(self.MANAGEMENT_KEY_ENV)
        if mgmt_key:
            # openrouter.ai/docs/api/api-reference/credits/get-remaining-credits
            payload = _bearer_get(f"{self.BASE_URL}/credits", mgmt_key)
            data = payload.get("data") or {}
            granted = data.get("total_credits")
            used = data.get("total_usage")
            balance = (
                granted - used if granted is not None and used is not None else None
            )
            return AccountStatus(
                provider=self.SERVICE_NAME,
                supported=True,
                balance=balance,
                currency="USD",
                total_granted=granted,
                total_used=used,
                raw=payload,
            )
        # Fallback: per-key view (openrouter.ai/docs/api-reference/limits).
        payload = _bearer_get(f"{self.BASE_URL}/key", self.api_key)
        data = payload.get("data") or {}
        return AccountStatus(
            provider=self.SERVICE_NAME,
            supported=True,
            balance=data.get("limit_remaining"),
            currency="USD",
            total_used=data.get("usage"),
            raw=payload,
        )
