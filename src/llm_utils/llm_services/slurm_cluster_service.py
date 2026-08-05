"""
SLURM cluster service — uses ``AsyncOpenAI`` + ``asyncio.gather`` pointed at
a vLLM OpenAI-compatible HTTP endpoint.

Acquires an endpoint from the injected server manager for each batch call
and releases it afterwards so multiple tasks can share the server pool.
"""
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx2
from openai import AsyncOpenAI

from ..base_llm_service import (
    BaseLLMService, _backoff_seconds, is_rate_limit_error, make_mechanism_error,
)
from ..llm_model import LLMModel
from ..media_utils import encode_image_to_b64
from .._logging import get_logger

logger = get_logger(__name__)


class SlurmClusterService(BaseLLMService):
    """Service for models served on a SLURM cluster via a vLLM HTTP server."""

    def __init__(self, model: LLMModel, **kwargs):
        # Duck-typed endpoint-provider contract — any object exposing
        # acquire_endpoint(model_id: str) -> str and
        # release_endpoint(model_id: str, endpoint: str). The serving
        # lifecycle (SLURM jobs, health, pooling) lives with the consumer's
        # cluster/device layer, not in this package.
        server_manager = kwargs.pop("server_manager", None)
        if not server_manager:
            raise ValueError(
                "SlurmClusterService requires 'server_manager' kwarg: an "
                "object with acquire_endpoint(model_id)/release_endpoint("
                "model_id, endpoint). Start the vLLM servers with your "
                "serving-lifecycle manager first."
            )

        super().__init__(
            max_concurrency=kwargs.pop("max_concurrency", 20),
            max_retries=kwargs.pop("max_retries", 5),
            batch_poll_interval=kwargs.pop("batch_poll_interval", 30),
            batch_timeout=kwargs.pop("batch_timeout", 3600),
        )
        self.model = model
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        # Extra request params merged verbatim into every API call (vLLM's
        # OpenAI-compatible endpoint accepts most OpenAI params) — parity
        # with the OpenAI family's api_params seam.
        self.api_params: Dict[str, Any] = kwargs.get("api_params") or {}
        self.server_manager = server_manager

        logger.info(f"Initialized cluster service for {model.model_id} (dynamic pool)")

    def _make_async_client(self, server_url: str) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=server_url,
            api_key="unused",
            http_client=httpx2.AsyncClient(
                trust_env=False,
                timeout=httpx2.Timeout(600.0, connect=60.0),
            ),
        )

    # ------------------------------------------------------------------
    # Message formatting (text + optional images for VLMs via vLLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_conversation(
        messages: List[Tuple[str, Optional[Any]]], system_message: Optional[str],
    ) -> List[Dict[str, Any]]:
        openai_msgs: List[Dict[str, Any]] = []
        if system_message:
            openai_msgs.append({"role": "system", "content": system_message})
        for text, image in messages:
            if image is None:
                openai_msgs.append({"role": "user", "content": text})
            else:
                images = image if isinstance(image, list) else [image]
                content: list = [{"type": "text", "text": text}]
                for img in images:
                    if img is not None:
                        b64, mime = encode_image_to_b64(img)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        })
                openai_msgs.append({"role": "user", "content": content})
        return openai_msgs

    # ------------------------------------------------------------------
    # Async execution
    # ------------------------------------------------------------------

    async def _one_call(
        self,
        client: AsyncOpenAI,
        sem: asyncio.Semaphore,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        is_test: bool,
        top_logprobs: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[dict]]:
        params: Dict[str, Any] = {
            "model": self.model.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # Shared quirk rule — see BaseLLMService._accepts_temperature.
        if self._accepts_temperature():
            params["temperature"] = temperature
        if self.api_params:
            params.update(self.api_params)
        if extra_params:
            params.update(extra_params)
        if top_logprobs is not None:
            params.update({"logprobs": True, "top_logprobs": top_logprobs})
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.chat.completions.create(**params)

                    choice = response.choices[0]
                    text = choice.message.content or ""
                    if not text.strip():
                        text = (f"[LLM response filtered out due to: "
                                f"{choice.finish_reason}]")

                    logprobs: Optional[dict] = None
                    if top_logprobs is not None:
                        lp = response.choices[0].logprobs
                        if lp is not None:
                            logprobs = lp.model_dump()

                    if hasattr(response, "usage") and response.usage:
                        in_tok = response.usage.prompt_tokens or 0
                        out_tok = response.usage.completion_tokens or 0
                        self._record_usage(in_tok, out_tok, 0.0, is_test)

                    return text, logprobs

                except Exception as e:
                    err = str(e)
                    if is_rate_limit_error(e) and attempt < self.max_retries:
                        wait = _backoff_seconds(attempt)
                        logger.warning(
                            f"Rate limit hit, retry {attempt + 1}/{self.max_retries} "
                            f"in {wait:.1f}s"
                        )
                        await asyncio.sleep(wait)
                        continue
                    # Per-model 404 (a model the server never loaded) →
                    # FatalModelError, same contract as the API services.
                    self._check_fatal_error(e, self.model.model_id)
                    logger.error(f"vLLM API error: {err}")
                    return make_mechanism_error(err), None
        return make_mechanism_error("retries exhausted (unreachable)"), None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _batch_chat_impl(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        system_message: Optional[str],
        is_test: bool,
        top_logprobs: Optional[int],
        **kwargs,
    ) -> List[Tuple[str, str, Optional[dict]]]:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        extra_params = kwargs.get("api_params")
        endpoint = self.server_manager.acquire_endpoint(self.model.model_id)
        try:
            prepared = [
                (cid, self._format_conversation(msgs, system_message))
                for cid, msgs in conversations
            ]

            logger.info(
                f"Sending {len(prepared)} requests async to vLLM "
                f"(concurrency={self.max_concurrency})"
            )

            async def _run() -> List[Tuple[str, Optional[dict]]]:
                # Client is created AND closed inside the event loop — one
                # per call, never leaked (httpx2 pools die with the client).
                client = self._make_async_client(endpoint)
                try:
                    sem = asyncio.Semaphore(self.max_concurrency)
                    tasks = [
                        self._one_call(
                            client, sem, msgs, temperature, max_tokens, is_test,
                            top_logprobs, extra_params,
                        )
                        for _, msgs in prepared
                    ]
                    return await asyncio.gather(*tasks)
                finally:
                    await client.close()

            responses = asyncio.run(_run())
            return [
                (cid, text, logprobs)
                for (cid, _), (text, logprobs) in zip(prepared, responses)
            ]
        finally:
            self.server_manager.release_endpoint(self.model.model_id, endpoint)

    def batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        system_message: Optional[str] = None,
        is_test: bool = False,
        **kwargs,
    ) -> List[Tuple[str, str]]:
        return [
            (cid, text)
            for cid, text, _ in self._batch_chat_impl(
                conversations, system_message, is_test, None, **kwargs
            )
        ]

    def batch_chat_with_logprobs(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        system_message: Optional[str] = None,
        is_test: bool = False,
        top_logprobs: int = 5,
        **kwargs,
    ) -> List[Tuple[str, str, Optional[dict]]]:
        """``batch_chat`` plus the OpenAI-schema per-token logprob payload.

        vLLM's OpenAI-compatible endpoint returns logprobs when asked
        (``logprobs=True, top_logprobs=N``); the payload lands as the third
        element of each result tuple — None on mechanism error or when the
        server sent none. See ``BaseLLMService.batch_chat_with_logprobs``.
        """
        if top_logprobs is None:
            raise ValueError(
                "top_logprobs must be a positive int — use batch_chat for a "
                "no-logprobs run")
        return self._batch_chat_impl(
            conversations, system_message, is_test, top_logprobs, **kwargs
        )
