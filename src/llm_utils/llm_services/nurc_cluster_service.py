"""
NURC Cluster service — uses ``AsyncOpenAI`` + ``asyncio.gather`` pointed at
a vLLM OpenAI-compatible HTTP endpoint.

Acquires an endpoint from ``ClusterModelServerManager`` for each batch call
and releases it afterwards so multiple tasks can share the server pool.
"""
import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

from ..base_llm_service import BaseLLMService, make_mechanism_error
from ..llm_model import LLMModel
from ..media_utils import encode_image_to_b64
from .._logging import get_logger

logger = get_logger(__name__)


class NURCClusterService(BaseLLMService):
    """Service for models running on NURC cluster via vLLM HTTP server."""

    def __init__(self, model: LLMModel, config=None, **kwargs):
        server_manager = kwargs.pop("server_manager", None)
        if not server_manager:
            raise ValueError(
                "NURCClusterService requires 'server_manager' kwarg. "
                "Use ClusterModelServerManager to start the vLLM server first."
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
        self.server_manager = server_manager

        logger.info(f"Initialized cluster service for {model.model_id} (dynamic pool)")

    def _make_async_client(self, server_url: str) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=server_url,
            api_key="unused",
            http_client=httpx.AsyncClient(
                trust_env=False,
                timeout=httpx.Timeout(600.0, connect=60.0),
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
    ) -> str:
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.chat.completions.create(
                        model=self.model.model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    text = response.choices[0].message.content or ""
                    if not text.strip():
                        text = "[Empty response from vLLM server]"

                    if hasattr(response, "usage") and response.usage:
                        in_tok = response.usage.prompt_tokens or 0
                        out_tok = response.usage.completion_tokens or 0
                        self._record_usage(in_tok, out_tok, 0.0, is_test)

                    return text

                except Exception as e:
                    err = str(e)
                    if ("429" in err or "rate" in err.lower()) and attempt < self.max_retries:
                        wait = (2 ** attempt) + random.random()
                        logger.warning(
                            f"Rate limit hit, retry {attempt + 1}/{self.max_retries} "
                            f"in {wait:.1f}s"
                        )
                        await asyncio.sleep(wait)
                        continue
                    logger.error(f"vLLM API error: {err}")
                    return make_mechanism_error(err)
        return make_mechanism_error("retries exhausted (unreachable)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        system_message: Optional[str] = None,
        is_test: bool = False,
        **kwargs,
    ) -> List[Tuple[str, str]]:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        endpoint = self.server_manager.acquire_endpoint(self.model)
        try:
            client = self._make_async_client(endpoint)
            prepared = [
                (cid, self._format_conversation(msgs, system_message))
                for cid, msgs in conversations
            ]

            logger.info(
                f"Sending {len(prepared)} requests async to vLLM "
                f"(concurrency={self.max_concurrency})"
            )

            async def _run() -> List[str]:
                sem = asyncio.Semaphore(self.max_concurrency)
                tasks = [
                    self._one_call(client, sem, msgs, temperature, max_tokens, is_test)
                    for _, msgs in prepared
                ]
                return await asyncio.gather(*tasks)

            responses = asyncio.run(_run())
            return [(cid, resp) for (cid, _), resp in zip(prepared, responses)]
        finally:
            self.server_manager.release_endpoint(self.model, endpoint)
