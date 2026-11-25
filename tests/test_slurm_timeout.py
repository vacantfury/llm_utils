"""Guard the long-vLLM-decode timeout bug reported by a consumer on 2026-11-19.

Exercise the real SDK against a stub transport that models decode time without
sleeping or contacting a cluster. A 32,768-token response at 20 tokens/second
exceeded the former 600-second read timeout and triggered duplicate SDK retries.
"""

import json
from types import SimpleNamespace

import httpx2
import pytest

from llm_utils import LLMModel, SlurmClusterService, config
from llm_utils.llm_services import slurm_cluster_service


@pytest.mark.parametrize("with_logprobs", [False, True])
@pytest.mark.parametrize(
    "max_tokens,service_params,call_params,expected_read,expected_connect",
    [
        (32768, {}, {}, 3576.8, 60.0),
        (16, {}, {}, 301.6, 60.0),
        (32768, {"timeout": 3600.0}, {}, 3600.0, 3600.0),
        (32768, {"timeout": 600.0}, {"timeout": 3600.0}, 3600.0, 3600.0),
        (32768, {"timeout": 600.0}, {"timeout": None}, None, None),
        (32768, {}, {"timeout": httpx2.Timeout(7200.0, connect=60.0)}, 7200.0, 60.0),
    ],
)
def test_decode_timeout_and_explicit_overrides(
    monkeypatch, with_logprobs, max_tokens, service_params, call_params,
    expected_read, expected_connect,
):
    requests = []
    clients = []
    released = []

    async def respond(request):
        requests.append(request)
        body = json.loads(request.content)
        assert body["max_tokens"] == max_tokens
        assert "timeout" not in body  # SDK option, not a vLLM generation parameter.
        timeout = request.extensions["timeout"]
        decode_seconds = body["max_tokens"] / 20.0
        if timeout["read"] is not None and timeout["read"] <= decode_seconds:
            raise httpx2.ReadTimeout("decode exceeded read timeout", request=request)
        return httpx2.Response(
            200,
            json={
                "id": "completion-stub",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "decoded"},
                    "finish_reason": "stop",
                    "logprobs": {"content": []} if with_logprobs else None,
                }],
            },
        )

    class StubAsyncClient(httpx2.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx2.MockTransport(respond), **kwargs)
            clients.append(self)

    monkeypatch.setattr(slurm_cluster_service.httpx2, "AsyncClient", StubAsyncClient)
    endpoint = "https://vllm.invalid/v1"
    manager = SimpleNamespace(
        acquire_endpoint=lambda model_id: endpoint,
        release_endpoint=lambda model_id, url: released.append((model_id, url)),
    )
    service = SlurmClusterService(
        LLMModel.LLAMA3_1_8B_CLUSTER,
        server_manager=manager,
        api_params=service_params,
    )
    batch = [("row", [("Decode a response.", None)])]
    if with_logprobs:
        result = service.batch_chat_with_logprobs(
            batch, max_tokens=max_tokens, api_params=call_params,
        )
        assert result[0][:2] == ("row", "decoded")
        assert result[0][2]["content"] == []
    else:
        result = service.batch_chat(batch, max_tokens=max_tokens, api_params=call_params)
        assert result == [("row", "decoded")]

    assert len(requests) == 1  # No timeout-induced SDK retry of a long decode.
    timeout = requests[0].extensions["timeout"]
    if expected_read is None:
        assert timeout["read"] is None
    else:
        assert timeout["read"] == pytest.approx(expected_read)
    assert timeout["connect"] == expected_connect
    if not service_params and not call_params:
        assert timeout["write"] == timeout["pool"] == 600.0
    assert len(clients) == 1 and clients[0].is_closed
    assert released == [(service.model.model_id, endpoint)]


@pytest.mark.parametrize(
    "service_kwargs,call_kwargs,settings,expected_tokens,expected_read",
    [
        # Omitted call budget inherits the configured service maximum.
        ({"max_tokens": 8192}, {}, {}, 8192, 1119.2),
        # None in a request uses that same maximum to size its timeout.
        ({"max_tokens": 8192}, {"max_tokens": None}, {}, None, 1119.2),
        # With neither budget set, the packaged/project route maximum applies.
        ({"max_tokens": None}, {}, {"max_tokens": 16384}, None, 1938.4),
        ({}, {}, {"max_tokens": 8192}, 8192, 1119.2),
        # api_params may replace the ordinary max_tokens argument.
        ({"api_params": {"max_tokens": 32768}}, {"max_tokens": 16}, {}, 32768, 3576.8),
        (
            {"api_params": {"max_tokens": 16}},
            {"api_params": {"max_tokens": 32768}}, {}, 32768, 3576.8,
        ),
        # Both latency allowances come from config, not hardcoded defaults.
        (
            {}, {"max_tokens": 16},
            {"decode_seconds_per_token": 0.2, "queue_prefill_seconds": 120.0},
            16, 123.2,
        ),
    ],
)
def test_timeout_uses_final_budget_and_route_config(
    monkeypatch, service_kwargs, call_kwargs, settings, expected_tokens, expected_read,
):
    config.configure(slurm_cluster=settings)
    calls = []

    async def create(**params):
        calls.append(params)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="decoded"))],
            usage=None,
        )

    async def close():
        pass

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close,
    )
    manager = SimpleNamespace(
        acquire_endpoint=lambda model_id: "https://vllm.invalid/v1",
        release_endpoint=lambda model_id, url: None,
    )
    service = SlurmClusterService(
        LLMModel.LLAMA3_1_8B_CLUSTER, server_manager=manager, **service_kwargs,
    )
    monkeypatch.setattr(service, "_make_async_client", lambda url: client)
    assert service.batch_chat([("row", [("Decode.", None)])], **call_kwargs) == [
        ("row", "decoded"),
    ]
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == expected_tokens
    assert calls[0]["timeout"].read == pytest.approx(expected_read)
    assert calls[0]["timeout"].connect == 60.0
