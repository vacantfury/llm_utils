"""Broker contract and SDK serialization against a keyless loopback server."""
import asyncio
import builtins
import json
import os
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from llm_utils import (
    BaseLLMService, BedrockService, BrokerError, BrokerModeUnsupportedError,
    ClaudeAPINotAllowed, ClaudeService, DeepSeekService, GoogleService, LLMModel,
    MoonshotService, OpenAIService, OpenRouterService, XAIService, ZAIService,
)
from llm_utils.broker import BrokerRoute, consumer_route


BASE_PATHS = {
    "openai": "/v1", "deepseek": "", "zai": "/api/paas/v4", "xai": "/v1",
    "moonshot": "/v1", "openrouter": "/api/v1", "anthropic": "", "gemini": "/v1beta",
    "bedrock": "",
}
CASES = [
    (OpenAIService, LLMModel.GPT_5, "openai", "chat"),
    (DeepSeekService, LLMModel.DEEPSEEK_V4_FLASH, "deepseek", "chat"),
    (ZAIService, LLMModel.GLM_5_2, "zai", "chat"),
    (XAIService, LLMModel.GROK_4_5, "xai", "chat"),
    (MoonshotService, LLMModel.KIMI_K3, "moonshot", "chat"),
    (OpenRouterService, LLMModel.OR_GLM_5_2, "openrouter", "chat"),
    (ClaudeService, LLMModel.CLAUDE_SONNET_5, "anthropic", "messages"),
    (GoogleService, LLMModel.GEMINI_2_5_FLASH, "gemini", "generate"),
    (BedrockService, LLMModel.BEDROCK_GLM_5, "bedrock", "converse"),
]
KEYS = {
    "OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_WEBHOOK_SECRET", "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_WEBHOOK_SIGNING_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY", "ZAI_API_KEY", "XAI_API_KEY", "MOONSHOT_API_KEY", "OPENROUTER_API_KEY",
    "OPENROUTER_MANAGEMENT_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
}
CANARY = "synthetic-provider-key-must-never-be-read"


@pytest.fixture
def fake_broker(monkeypatch):
    state = SimpleNamespace(requests=[], grants=[], revoked=[], status=200,
                            grant_status=200, revoke_status=200, mutate=lambda g: g)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            headers = dict(self.headers.items())
            state.requests.append((self.path, headers, body))
            code = 200
            if self.path == "/grants":
                code = state.grant_status
                routes = body["routes"]
                number = len(state.grants)
                result = state.mutate(dict(id=str(number), surrogate=f"test-surrogate-{number}",
                                          expires=time.time() + 3600, routes=routes,
                                          base_paths={r: BASE_PATHS[r] for r in routes}))
                state.grants.append(result)
            elif self.path == "/revoke":
                code = state.revoke_status
                state.revoked.append(self.headers.get("authorization"))
                result = {"revoked": True}
            else:
                code = state.status
                if self.path.startswith("/proxy/anthropic"):
                    result = dict(id="msg-test", type="message", role="assistant", model="test",
                                  content=[dict(type="text", text="ok")], stop_reason="end_turn",
                                  usage=dict(input_tokens=1, output_tokens=1))
                elif self.path.startswith("/proxy/gemini"):
                    result = dict(candidates=[dict(content=dict(role="model", parts=[dict(text="ok")]))],
                                  usageMetadata=dict(promptTokenCount=1, candidatesTokenCount=1))
                elif self.path.startswith("/proxy/bedrock"):
                    result = dict(output=dict(message=dict(content=[dict(text="ok")])),
                                  usage=dict(inputTokens=1, outputTokens=1))
                else:
                    result = dict(id="chat-test", object="chat.completion", created=0, model="test",
                                  choices=[dict(index=0, message=dict(role="assistant", content="ok"),
                                                finish_reason="stop")],
                                  usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2))
            if code >= 400:
                result = {"error": {"message": "synthetic refusal", "type": "server_error", "code": code}}
            encoded = json.dumps(result).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Set-Cookie", "ambient=must-not-return")
            if 300 <= code < 400:
                self.send_header("Location", state.url + "/redirect-target")
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("LLM_UTILS_BROKER_URL", state.url)
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def no_provider_keys(monkeypatch):
    for key in KEYS:
        monkeypatch.setenv(key, CANARY)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "http://127.0.0.1:1")
    for env in ("OPENAI_CUSTOM_HEADERS", "ANTHROPIC_CUSTOM_HEADERS"):
        monkeypatch.setenv(env, f"Authorization: {CANARY}\nCookie: {CANARY}\nX-Api-Key: {CANARY}")
    original = type(os.environ).__getitem__

    def guarded(env, key):
        if key in KEYS:
            raise AssertionError(f"Provider key read: {key}")
        return original(env, key)

    monkeypatch.setattr(type(os.environ), "__getitem__", guarded)


@pytest.mark.parametrize("cls,model,route,operation", CASES, ids=[c[2] for c in CASES])
def test_provider_inference(fake_broker, no_provider_keys, cls, model, route, operation):
    with cls(model, api_key=CANARY, use_batch_api=True, max_retries=5) as service:
        assert service.max_retries == 0
        assert service.use_batch_api is False
        assert service.chat("synthetic public test prompt") == "ok"
        assert service.chat("second synthetic prompt") == "ok"
        assert service.get_usage()["total"]["input_tokens"] == 2
    assert fake_broker.requests[0][2]["routes"] == {route: [operation]}
    assert len(fake_broker.revoked) == 1
    inference = [r for r in fake_broker.requests if r[0].startswith("/proxy/")]
    assert len(inference) == 2
    for path, headers, body in inference:
        headers = {k.lower(): v for k, v in headers.items()}
        assert headers["authorization"] == "Bearer test-surrogate-0"
        assert not {"cookie", "x-api-key", "x-goog-api-key", "proxy-authorization"} & headers.keys()
        assert path.startswith(f"/proxy/{route}{BASE_PATHS[route]}/")
    assert CANARY not in json.dumps(fake_broker.requests)


@pytest.mark.parametrize("cls,model,route,operation", CASES, ids=[c[2] for c in CASES])
def test_refused_lookups(fake_broker, no_provider_keys, cls, model, route, operation):
    with cls(model) as service:
        for method, args in [("get_account_status", ()), ("submit_batch_chat", ([],)),
                             ("batch_chat_status", ("job",)), ("harvest_batch_chat", ("job",))]:
            with pytest.raises(BrokerModeUnsupportedError):
                getattr(service, method)(*args)
        if hasattr(service, "_fetch_account_status"):
            with pytest.raises(BrokerModeUnsupportedError):
                service._fetch_account_status()
    assert [r[0] for r in fake_broker.requests] == ["/grants", "/revoke"]


@pytest.mark.parametrize("value", ["", "https://127.0.0.1:8000", "http://localhost:8000",
    "http://127.0.0.2:8000", "http://[::1]:8000", "http://example.com", "http://127.0.0.1.evil",
    "http://user@127.0.0.1", "http://127.0.0.1/path", "http://127.0.0.1?key=x",
    "http://127.0.0.1#fragment", "http://127.0.0.1:0", "http://127.0.0.1:bad",
    "http://127.0.0.1:99999", " http://127.0.0.1", "http://127.0.0.1?", "http://127.0.0.1#"])
def test_non_loopback_url_refused(monkeypatch, value):
    monkeypatch.setenv("LLM_UTILS_BROKER_URL", value)
    with pytest.raises(BrokerError):
        OpenAIService(LLMModel.GPT_5, api_key="dummy")


def test_off_mode_and_construction_timing(fake_broker, monkeypatch):
    monkeypatch.delenv("LLM_UTILS_BROKER_URL")
    direct = OpenAIService(LLMModel.GPT_5, api_key="dummy")
    assert direct.api_key == "dummy" and direct._broker_route is None
    assert direct._supports_native_batch()
    monkeypatch.setenv("LLM_UTILS_BROKER_URL", fake_broker.url)
    with OpenAIService(LLMModel.GPT_5) as broker:
        monkeypatch.delenv("LLM_UTILS_BROKER_URL")
        assert broker.chat("synthetic prompt") == "ok"
        assert direct._broker_route is None
    asyncio.run(direct.async_client.close())


def test_claude_opt_in_precedes_grant(fake_broker, monkeypatch):
    monkeypatch.delenv("LLM_UTILS_ALLOW_CLAUDE_API")
    with pytest.raises(ClaudeAPINotAllowed):
        ClaudeService(LLMModel.CLAUDE_SONNET_5)
    assert fake_broker.requests == []


def test_bedrock_never_imports_boto(fake_broker, monkeypatch, no_provider_keys):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"boto3", "botocore"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with BedrockService(LLMModel.BEDROCK_GLM_5) as service:
        assert service.chat("synthetic prompt") == "ok"
    body = fake_broker.requests[1][2]
    assert body["model"] == LLMModel.BEDROCK_GLM_5.model_id
    assert "modelId" not in body


@pytest.mark.parametrize("status", [307, 429, 500])
def test_openai_no_retries_or_redirects(fake_broker, status):
    fake_broker.status = status
    with OpenAIService(LLMModel.GPT_5) as service:
        result = service.chat("synthetic prompt")
        assert result != "ok"
    assert len(fake_broker.requests) == 3  # grant, one attempt, revoke


@pytest.mark.parametrize("status", [307, 403, 500])
def test_grant_failure_has_no_fallback(fake_broker, no_provider_keys, status):
    fake_broker.grant_status = status
    with pytest.raises(BrokerError):
        OpenAIService(LLMModel.GPT_5, api_key=CANARY)
    assert len(fake_broker.requests) == 1


@pytest.mark.parametrize("change", [
    {"expires": 0}, {"expires": float("nan")}, {"routes": {}},
    {"base_paths": {"openai": "//evil"}}, {"base_paths": {"openai": "/../evil"}},
    {"surrogate": "bad\nheader"}, {"surrogate": ""}, {"base_paths": {"openai": "https://evil"}},
])
def test_invalid_grant(fake_broker, change):
    fake_broker.mutate = lambda grant: {**grant, **change}
    with pytest.raises(BrokerError):
        consumer_route("openai", "chat")


def test_grant_close_and_expiry(fake_broker):
    route = consumer_route("openai", "chat")
    request = httpx.Request("POST", route.base_url + "/chat/completions")
    route.grant = replace(route.grant, expires=0)
    with pytest.raises(BrokerError, match="expired"):
        route.authenticate(request)
    route.close()
    route.close()
    assert len(fake_broker.revoked) == 1


def test_failed_revoke_is_not_replayed(fake_broker):
    route = consumer_route("openai", "chat")
    fake_broker.revoke_status = 500
    with pytest.raises(BrokerError):
        route.close()
    route.close()
    assert len(fake_broker.revoked) == 1
    with pytest.raises(BrokerError, match="closed"):
        route.authenticate(httpx.Request("POST", route.base_url + "/chat/completions"))


def test_auth_strips_ambient_values_and_pins_destination(fake_broker):
    with consumer_route("openai", "chat") as route:
        headers = {k: CANARY for k in ["Authorization", "Proxy-Authorization", "Cookie", "X-Api-Key",
                                     "X-Goog-Api-Key", "X-Amz-Security-Token", "Host", "OpenAI-Project"]}
        request = httpx.Request("POST", route.base_url + "/chat/completions?key=canary", headers=headers)
        route.authenticate(request)
        assert CANARY not in str(request.headers)
        assert "key=" not in str(request.url)
        assert request.headers["authorization"] == "Bearer " + route.surrogate
        for target in ["http://example.com/v1/chat/completions", fake_broker.url + "/proxy/xai/v1/chat/completions",
                       route.base_url + "other/chat/completions", route.base_url + "/%2e%2e/chat/completions"]:
            with pytest.raises(BrokerError):
                route.authenticate(httpx.Request("POST", target))
        for path in ["/files", "/batches", "/models", "/credits", "/key"]:
            with pytest.raises(BrokerModeUnsupportedError):
                route.authenticate(httpx.Request("POST", route.base_url + path))


def test_native_sdk_resources_fail_locally(fake_broker):
    with OpenAIService(LLMModel.GPT_5) as service:
        for client in (service.async_client, service.sync_client):
            with pytest.raises(BrokerModeUnsupportedError):
                client.files.create(file=b"synthetic", purpose="batch")
            with pytest.raises(BrokerModeUnsupportedError):
                client.batches.retrieve("job")
            with pytest.raises(BrokerModeUnsupportedError):
                client.uploads.create(bytes=1, filename="test", mime_type="text/plain", purpose="batch")
    assert len(fake_broker.requests) == 2


def test_async_close_and_closed_service(fake_broker):
    async def run():
        async with OpenAIService(LLMModel.GPT_5) as service:
            assert await service.achat("synthetic prompt") == "ok"
        assert service.async_client.is_closed()
        with pytest.raises(BrokerError):
            await service.achat("synthetic prompt")
    asyncio.run(run())
    assert len(fake_broker.revoked) == 1


def test_constants_import_skips_keys_and_dotenv(fake_broker, no_provider_keys, monkeypatch):
    import runpy
    from pathlib import Path
    import dotenv

    def fail(*args, **kwargs):
        pytest.fail("Broker import must not load dotenv")

    monkeypatch.setattr(dotenv, "load_dotenv", fail)
    result = runpy.run_path(str(Path(__file__).parents[1] / "src/llm_utils/constants.py"))
    for key in KEYS.intersection(result):
        assert result[key] is None


def test_transport_construction_failure_revokes(fake_broker, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("synthetic construction failure")

    monkeypatch.setattr(BrokerRoute, "http_client", fail)
    with pytest.raises(ValueError, match="synthetic construction"):
        OpenAIService(LLMModel.GPT_5)
    assert len(fake_broker.revoked) == 1


def test_google_wire_format_without_sdk_auth(fake_broker, no_provider_keys, monkeypatch):
    from PIL import Image
    from google import genai

    # Response typing is separate from this request-serialization contract.
    monkeypatch.setattr(genai, "types", SimpleNamespace(GenerateContentResponse=SimpleNamespace(
        model_validate=lambda payload: SimpleNamespace(text="ok"))), raising=False)
    with GoogleService(LLMModel.GEMINI_2_5_FLASH, call_timeout=123) as service:
        assert service.client._timeout == 123
        assert service.batch_chat([("item", [("synthetic text", Image.new("RGB", (1, 1)))])],
                                  system_message="synthetic system", temperature=0.5) == [("item", "ok")]
    path, headers, body = fake_broker.requests[1]
    assert path == "/proxy/gemini/v1beta/models/gemini-2.5-flash:generateContent"
    assert body["systemInstruction"] == {"parts": [{"text": "synthetic system"}]}
    assert body["generationConfig"]["temperature"] == 0.5
    assert body["contents"][0]["parts"][1] == {"text": "synthetic text"}
    inline = body["contents"][0]["parts"][0]["inlineData"]
    assert inline["mimeType"].startswith("image/") and inline["data"]
    assert CANARY not in json.dumps(fake_broker.requests)


def test_http_client_ignores_redirect_override(fake_broker):
    fake_broker.status = 307
    with consumer_route("openai", "chat") as route:
        client = route.http_client()
        response = client.post(route.base_url + "/chat/completions", json={}, follow_redirects=True)
        assert response.status_code == 307
    assert len(fake_broker.requests) == 3
