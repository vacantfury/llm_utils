"""Per-request wall-clock deadline on the OpenAI-compatible realtime path. An
endpoint that keeps a request open forever (OpenRouter's whitespace keepalive
resets the SDK's per-read timer) must end as a mechanism error, never a hang.
No network: the client's create/parse are replaced by coroutines that sleep."""

import asyncio
import time

from llm_utils import LLMModel, is_mechanism_error
from llm_utils.llm_services import OpenAIService, OpenRouterService
from llm_utils.llm_services.openai_service import _DEFAULT_CALL_TIMEOUT_S

KEY = {"api_key": "test-key-not-real"}


async def _never_answers(**_params):
    await asyncio.sleep(3600)


def test_default_deadline_is_finite():
    svc = OpenAIService(LLMModel.GPT_5_NANO, **KEY)
    assert svc.call_timeout == _DEFAULT_CALL_TIMEOUT_S


def test_hung_request_becomes_mechanism_error():
    svc = OpenRouterService(LLMModel.OR_DEEPSEEK_V4_FLASH, call_timeout=0.05, **KEY)
    svc.async_client.chat.completions.create = _never_answers
    started = time.monotonic()
    text = svc.chat("hello")
    assert is_mechanism_error(text)
    assert time.monotonic() - started < 5
