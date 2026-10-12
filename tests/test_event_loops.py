"""t22: successive calls on one OpenAIService never reuse a connection from an
earlier event loop.

Every entry point runs its own ``asyncio.run``, so each call sees a new loop. A
real connection belongs to the loop that opened it; reused from the next loop,
the request is written and the read fails, and the SDK's retry resends a request
the server already has. The fake network backend below gives the SDK's real
HTTP stack (httpx over httpcore's pool) loop-bound connections without opening
a socket, so the suite stays offline.
"""
import asyncio
import json

import httpcore
import pytest

from llm_utils import LLMModel, OpenAIService

_BODY = json.dumps({
    "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "pong"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}).encode()
_RESPONSE = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
             b"Content-Length: %d\r\n\r\n" % len(_BODY)) + _BODY


class _LoopBoundStream(httpcore.AsyncNetworkStream):
    """One connection: answers one request, and fails when used from another loop."""

    def __init__(self, backend):
        self.backend = backend
        self.loop = asyncio.get_running_loop()
        self.buffer = [_RESPONSE]

    def _check_loop(self):
        if asyncio.get_running_loop() is not self.loop:
            self.backend.foreign_loop_uses += 1
            raise httpcore.ReadError("Event loop is closed")

    async def read(self, max_bytes, timeout=None):
        self._check_loop()
        return self.buffer.pop(0) if self.buffer else b""

    async def write(self, buffer, timeout=None):
        self._check_loop()
        if buffer.startswith(b"POST "):
            self.backend.requests += 1

    async def aclose(self):
        pass

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        return self

    def get_extra_info(self, info):
        return None


class _LoopBoundBackend(httpcore.AsyncNetworkBackend):
    def __init__(self):
        self.connects = 0
        self.requests = 0
        self.foreign_loop_uses = 0

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connects += 1
        return _LoopBoundStream(self)

    async def sleep(self, seconds):
        pass


@pytest.fixture
def svc(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
                "all_proxy", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    service = OpenAIService(LLMModel.GPT_4O_MINI, api_key="k", use_batch_api=False)
    backend = _LoopBoundBackend()
    service.async_client._client._transport._pool._network_backend = backend
    yield service, backend
    asyncio.run(service.async_client.close())


def test_direct_route_keeps_no_idle_connections(svc):
    service, _ = svc
    pool = service.async_client._client._transport._pool
    assert pool._max_keepalive_connections == 0


def test_successive_calls_each_send_one_request_on_their_own_loop(svc):
    service, backend = svc
    assert service.chat("ping") == "pong"
    assert service.chat("ping") == "pong"
    # Two calls, two requests: none written on a connection from an earlier
    # loop, so no failed read and no resend through the SDK's retry.
    assert backend.foreign_loop_uses == 0
    assert backend.requests == 2
    assert backend.connects == 2
