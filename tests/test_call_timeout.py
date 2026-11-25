"""Guards consumer chat calls against endpoints that hold requests open forever.

The provider client is replaced by a sleeping coroutine; no network is used."""

import asyncio
import time

from llm_utils import LLMModel, is_mechanism_error
from llm_utils.llm_services import OpenRouterService

KEY = {"api_key": "test-key-not-real"}


async def _never_answers(**_params):
    await asyncio.sleep(3600)


def test_hung_request_becomes_mechanism_error():
    svc = OpenRouterService(LLMModel.OR_DEEPSEEK_V4_FLASH, call_timeout=0.05, **KEY)
    svc.async_client.chat.completions.create = _never_answers
    started = time.monotonic()
    text = svc.chat("hello")
    assert is_mechanism_error(text)
    assert time.monotonic() - started < 5
