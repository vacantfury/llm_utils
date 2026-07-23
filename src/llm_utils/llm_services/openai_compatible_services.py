"""
OpenAI-compatible third-party services — DeepSeek, Z.AI (GLM), xAI (Grok),
Moonshot (Kimi).

All expose OpenAI-compatible ``/v1/chat/completions`` endpoints, so they
subclass ``OpenAIService`` and only override the API key + base URL; every
request/batch path is inherited unchanged.

PRIVACY (global data-jurisdiction rule): DeepSeek, Z.AI, and Moonshot are DIRECT
MAINLAND endpoints — their data is processed under PRC jurisdiction. Use them
ONLY for ZERO-personal-data bulk work — LLM-judge calls / ASR sweeps / attack
targets over PUBLIC benchmark responses. NEVER route personal data through them.
This public research repo runs exactly that zero-personal-data work, so the
mainland APIs are sanctioned here. xAI (Grok) is US jurisdiction, same posture as
the other US frontier providers.
"""
from ..constants import (
    DEEPSEEK_API_KEY, DEEPSEEK_API_URL,
    ZAI_API_KEY, ZAI_API_URL,
    XAI_API_KEY, XAI_API_URL,
    MOONSHOT_API_KEY, MOONSHOT_API_URL,
)
from .openai_service import OpenAIService


class DeepSeekService(OpenAIService):
    """DeepSeek platform (deepseek-v4-*). Mainland endpoint — no personal data."""

    API_KEY = DEEPSEEK_API_KEY
    BASE_URL = DEEPSEEK_API_URL
    SERVICE_NAME = "DeepSeek"


class ZAIService(OpenAIService):
    """Z.AI open platform (GLM-*). Mainland endpoint — no personal data."""

    API_KEY = ZAI_API_KEY
    BASE_URL = ZAI_API_URL
    SERVICE_NAME = "Z.AI"


class XAIService(OpenAIService):
    """xAI Grok family (grok-*). US jurisdiction, OpenAI-compatible."""

    API_KEY = XAI_API_KEY
    BASE_URL = XAI_API_URL
    SERVICE_NAME = "xAI"


class MoonshotService(OpenAIService):
    """Moonshot Kimi family (kimi-*). Mainland endpoint — no personal data."""

    API_KEY = MOONSHOT_API_KEY
    BASE_URL = MOONSHOT_API_URL
    SERVICE_NAME = "Moonshot"
