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
from ..constants import (
    DEEPSEEK_API_URL, MOONSHOT_API_URL, OPENROUTER_API_URL, XAI_API_URL, ZAI_API_URL,
)
from .openai_service import OpenAIService


class DeepSeekService(OpenAIService):
    """DeepSeek platform (deepseek-v4-*). Mainland endpoint — no personal data."""

    API_KEY_ENV = "DEEPSEEK_API_KEY"
    BASE_URL = DEEPSEEK_API_URL
    SERVICE_NAME = "DeepSeek"


class ZAIService(OpenAIService):
    """Z.AI open platform (GLM-*). Mainland endpoint — no personal data."""

    API_KEY_ENV = "ZAI_API_KEY"
    BASE_URL = ZAI_API_URL
    SERVICE_NAME = "Z.AI"


class XAIService(OpenAIService):
    """xAI Grok family (grok-*). US jurisdiction, OpenAI-compatible."""

    API_KEY_ENV = "XAI_API_KEY"
    BASE_URL = XAI_API_URL
    SERVICE_NAME = "xAI"


class MoonshotService(OpenAIService):
    """Moonshot Kimi family (kimi-*). Mainland endpoint — no personal data."""

    API_KEY_ENV = "MOONSHOT_API_KEY"
    BASE_URL = MOONSHOT_API_URL
    SERVICE_NAME = "Moonshot"


class OpenRouterService(OpenAIService):
    """OpenRouter aggregator (openrouter.ai) — hosted open-weight models.

    US jurisdiction. Data-retention behavior depends on the ACCOUNT's routing
    policy: with zero-data-retention routing enabled, requests reach only
    non-retaining hosts (see the ``Provider.OPENROUTER`` registry note).
    """

    API_KEY_ENV = "OPENROUTER_API_KEY"
    BASE_URL = OPENROUTER_API_URL
    SERVICE_NAME = "OpenRouter"
