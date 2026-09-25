"""Optional loopback credential-broker transport, with no broker-package dependency.

The broker owns policy, provider keys, grant lifetimes and signing. This client
only requests an inference grant, carries its surrogate, and revokes it on close.
"""
from __future__ import annotations

import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

import httpx2 as httpx

from .exceptions import BrokerError, BrokerModeUnsupportedError

BROKER_URL_ENV = "LLM_UTILS_BROKER_URL"


def _base_url(value: str) -> str:
    try:
        url = urlsplit(value)
        if (url.scheme != "http" or url.hostname != "127.0.0.1"
                or url.username is not None or url.password is not None
                or url.path not in {"", "/"} or "?" in value or "#" in value
                or url.port == 0 or any(c.isspace() for c in value)):
            raise ValueError
    except (ValueError, TypeError):
        raise BrokerError("Broker URL must be http://127.0.0.1 with an optional port") from None
    return f"http://127.0.0.1:{url.port or 80}"


def _path(value: str) -> bool:
    """Accept literal path components only, never URL or traversal syntax."""
    return isinstance(value, str) and (value == "" or bool(
        re.fullmatch(r"(?:/[A-Za-z0-9_-]+)+", value)))


@dataclass(frozen=True)
class Grant:
    id: str
    surrogate: str = field(repr=False)
    expires: float
    routes: dict[str, list[str]]
    base_paths: dict[str, str]


class BrokerClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0):
        self.base_url = _base_url(base_url)
        self.timeout = timeout

    def _request(self, path: str, payload: dict, surrogate: str | None = None):
        # Fresh clients prevent cookies/auth persisting between control requests.
        try:
            with httpx.Client(trust_env=False, follow_redirects=False, verify=False,
                              timeout=self.timeout) as client:
                headers = {"Authorization": "Bearer " + surrogate} if surrogate else {}
                response = client.post(self.base_url + path, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception:
            # Neither server bodies nor HTTP exception URLs may expose tokens.
            raise BrokerError("Broker unavailable or request denied; no direct-key fallback") from None

    def grant(self, routes: dict[str, list[str]]) -> Grant:
        if not routes or any(not re.fullmatch(r"[a-z][a-z0-9_]*", r) for r in routes):
            raise BrokerError("Invalid broker routes")
        result = self._request("/grants", {"run_id": "llm-utils-" + uuid.uuid4().hex,
                                          "routes": routes})
        try:
            grant = Grant(**result)
            if (not isinstance(grant.id, str) or not grant.id
                    or not isinstance(grant.surrogate, str) or not grant.surrogate
                    or any(ord(c) < 33 or ord(c) > 126 for c in grant.surrogate)
                    or isinstance(grant.expires, bool) or not math.isfinite(grant.expires)
                    or grant.expires <= time.time() or grant.routes != routes
                    or set(grant.base_paths) != set(routes)
                    or not all(_path(p) for p in grant.base_paths.values())):
                raise ValueError
            return grant
        except (TypeError, ValueError, AttributeError):
            raise BrokerError("Invalid broker grant") from None

    def revoke(self, grant: Grant) -> None:
        result = self._request("/revoke", {}, grant.surrogate)
        if not isinstance(result, dict) or result.get("revoked") is not True:
            raise BrokerError("Broker did not confirm grant revocation")


class BrokerRoute:
    def __init__(self, name: str, client: BrokerClient, grant: Grant):
        self.name, self.client, self.grant = name, client, grant
        self.base_url = client.base_url + "/proxy/" + name + grant.base_paths[name]
        self._closed = False
        self._sync_clients = []
        self._async_clients = []

    @property
    def surrogate(self) -> str:
        return self.grant.surrogate

    def _ensure_open(self) -> None:
        if self._closed or time.time() >= self.grant.expires:
            raise BrokerError("Broker grant is closed or expired")

    def authenticate(self, request) -> None:
        self._ensure_open()
        target, base = urlsplit(str(request.url)), urlsplit(self.base_url)
        decoded = unquote(target.path)
        if (target.scheme != base.scheme or target.netloc != base.netloc
                or target.fragment or "\\" in decoded
                or any(p in {".", ".."} for p in decoded.split("/"))
                or not (target.path == base.path or target.path.startswith(base.path + "/"))):
            raise BrokerError("SDK destination outside broker route")
        # Only the inference operation is exposed. SDK file/batch/admin helpers
        # fail locally, before a body (or surrogate) can leave this process.
        suffix = target.path[len(base.path):]
        operations = self.grant.routes[self.name]
        allowed = (
            ("chat" in operations and suffix == "/chat/completions")
            or ("messages" in operations and self.name == "anthropic" and suffix == "/v1/messages")
            or ("generate" in operations and self.name == "gemini"
                and bool(re.fullmatch(r"/models/[^/]+:generateContent", suffix)))
            or ("converse" in operations and self.name == "bedrock" and suffix == "/converse")
        )
        if request.method != "POST" or not allowed:
            raise BrokerModeUnsupportedError("Operation unsupported in broker mode")
        for header in ("authorization", "proxy-authorization", "x-api-key", "api-key",
                       "x-goog-api-key", "x-amz-security-token", "cookie",
                       "openai-organization", "openai-project"):
            request.headers.pop(header, None)
        # Do not forward ambient API-key query parameters either.
        request.url = request.url.copy_with(params=[
            (k, v) for k, v in request.url.params.multi_items()
            if k.lower() not in {"key", "api_key", "api-key", "access_token"}])
        request.headers["host"] = base.netloc
        request.headers["authorization"] = "Bearer " + self.surrogate

    async def authenticate_async(self, request) -> None:
        self.authenticate(request)

    def http_client(self, *, asynchronous: bool = False):
        # Use the SDKs' native HTTP types, without changing process-wide imports.
        import httpx as sdk_httpx

        class SyncClient(sdk_httpx.Client):
            def send(self, request, **kwargs):
                return super().send(request, **{**kwargs, "follow_redirects": False, "auth": None})

        class AsyncClient(sdk_httpx.AsyncClient):
            async def send(self, request, **kwargs):
                return await super().send(request, **{**kwargs, "follow_redirects": False, "auth": None})

        cls = AsyncClient if asynchronous else SyncClient
        hook = self.authenticate_async if asynchronous else self.authenticate
        # The pinned route is HTTP loopback only, so no TLS trust store is used.
        client = cls(trust_env=False, follow_redirects=False, verify=False,
                     # Services can be used from successive asyncio.run loops.
                     limits=sdk_httpx.Limits(max_keepalive_connections=0),
                     event_hooks={"request": [hook]})
        (self._async_clients if asynchronous else self._sync_clients).append(client)
        return client

    def converse(self, **request):
        self._ensure_open()
        payload = dict(request)
        payload["model"] = payload.pop("modelId")
        return self.client._request("/proxy/" + self.name + "/converse", payload, self.surrogate)

    def close(self) -> None:
        import asyncio
        try:
            self._revoke()
        finally:
            for client in self._sync_clients:
                client.close()
            if self._async_clients:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(self._close_async_clients())
                else:
                    loop.create_task(self._close_async_clients())

    async def _close_async_clients(self) -> None:
        for client in self._async_clients:
            await client.aclose()

    async def aclose(self) -> None:
        import asyncio
        try:
            await asyncio.to_thread(self._revoke)
        finally:
            for client in self._sync_clients:
                client.close()
            await self._close_async_clients()

    def _revoke(self) -> None:
        if self._closed:
            return
        # An uncertain revoke must never be retried automatically. Locally the
        # route is dead even when the daemon cannot confirm the revocation.
        self._closed = True
        self.client.revoke(self.grant)

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *args):
        self.close()


def consumer_route(name: str, operation: str) -> BrokerRoute | None:
    """Read the opt-in once, at service construction; failure never selects OFF."""
    value = os.getenv(BROKER_URL_ENV)
    if value is None:
        return None
    client = BrokerClient(value)
    return BrokerRoute(name, client, client.grant({name: [operation]}))


class UnsupportedAPI:
    """Local refusal for native SDK resources, including nested helpers."""

    def __getattr__(self, name):
        raise BrokerModeUnsupportedError("Native batch/file API unsupported in broker mode")


class BrokerGoogleClient:
    """Only GoogleService's generate_content surface, with no SDK auth discovery.

    The Google SDK reads key environment variables even with an explicit key.
    Use its response types but send this small inference surface directly.
    """

    def __init__(self, route: BrokerRoute, *, timeout: float):
        self._route = route
        self._http = route.http_client()
        self._timeout = timeout
        self.models = self
        self.batches = self.files = UnsupportedAPI()

    def generate_content(self, *, model, contents, config):
        from google.genai import types
        from .media_utils import encode_image_to_b64
        from urllib.parse import quote

        parts = []
        for content in contents:
            if isinstance(content, str):
                parts.append({"text": content})
            else:
                data, mime = encode_image_to_b64(content)
                parts.append({"inlineData": {"mimeType": mime, "data": data}})
        generation = {"maxOutputTokens": config["max_output_tokens"]}
        if "temperature" in config:
            generation["temperature"] = config["temperature"]
        if "top_p" in config:
            generation["topP"] = config["top_p"]
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation}
        if config.get("system_instruction"):
            body["systemInstruction"] = {"parts": [{"text": config["system_instruction"]}]}
        model = model.removeprefix("models/")
        response = self._http.post(
            self._route.base_url + "/models/" + quote(model, safe="") + ":generateContent",
            json=body, timeout=self._timeout)
        if response.is_error:
            # Preserve the SDK's status/error classification used by the service.
            from google.genai.errors import APIError
            raise APIError(response.status_code, response.json(), response)
        response.raise_for_status()
        return types.GenerateContentResponse.model_validate(response.json())
