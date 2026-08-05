"""
OpenAI service — two serving routes behind one ``batch_chat``:

- **realtime**: ``AsyncOpenAI`` + ``asyncio.gather`` concurrent requests
  (full list price, seconds-to-minutes turnaround);
- **native Batch API** (real OpenAI endpoint only): in-memory JSONL upload →
  queued batch (24h completion window, usually much faster) at 50% price.

``batch_chat`` auto-routes by estimated job cost (``batch_threshold_usd``);
``use_batch_api`` forces either route. OpenAI-compatible endpoints (DeepSeek,
Z.AI, xAI, …) don't implement ``/v1/batches`` and always run realtime.
"""
import asyncio
import io
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI

from ..base_llm_service import (
    BaseLLMService, _backoff_seconds, is_rate_limit_error, make_mechanism_error,
)
from ..llm_model import LLMModel, ModelQuirk
from .. import constants as _constants  # noqa: F401  (side effect: load_dotenv)
from ..media_utils import encode_image_to_b64
from .._logging import get_logger

logger = get_logger(__name__)

# Batch API terminal statuses. "expired" still carries partial results
# (finished items are returned and billed); "failed" = wholesale validation
# failure with no per-item results.
_BATCH_TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})


class OpenAIService(BaseLLMService):
    """Service for OpenAI models (GPT-4o, GPT-5, etc.).

    Also the base for OpenAI-COMPATIBLE third-party endpoints (DeepSeek, Z.AI,
    xAI) — subclasses override the three class attributes below and inherit all
    request logic unchanged (see llm_services/openai_compatible_services.py).
    """

    # Provider overrides — subclasses change these for OpenAI-compatible endpoints.
    API_KEY_ENV: str = "OPENAI_API_KEY"       # env var holding the key
    BASE_URL: Optional[str] = None            # None → default OpenAI endpoint
    SERVICE_NAME: str = "OpenAI"              # for logs / error messages
    # OpenAI exposes NO balance endpoint to any API key (the old dashboard
    # credit_grants route now demands a browser session token; verified
    # 2026-08-04). The org Costs/Usage APIs exist but need an Admin key —
    # see repo TODO. BALANCE_QUERY_VIA stays None (base default).

    def __init__(self, model: LLMModel, **kwargs):
        super().__init__(
            max_concurrency=kwargs.pop("max_concurrency", 20),
            max_retries=kwargs.pop("max_retries", 5),
            batch_poll_interval=kwargs.pop("batch_poll_interval", 30),
            batch_timeout=kwargs.pop("batch_timeout", 3600),
            # Native Batch API routing (real OpenAI endpoint only):
            #   None → auto (cost threshold) · True → always batch (NB: `chat`
            #   funnels through `batch_chat`, so forcing True queues single
            #   calls too) · False → never batch. See BaseLLMService.
            use_batch_api=kwargs.pop("use_batch_api", None),
            batch_threshold_usd=kwargs.pop("batch_threshold_usd", None),
        )
        self.model = model
        self.api_key = kwargs.get("api_key") or os.getenv(self.API_KEY_ENV)
        if not self.api_key:
            raise ValueError(
                f"{self.SERVICE_NAME} API key not found. Set {self.API_KEY_ENV} "
                f"in the environment (or a repo .env) or pass the api_key parameter"
            )
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        # Extra request params passed verbatim to every API call (e.g.
        # response_format, seed, top_p). Per-call kwargs["api_params"] merges
        # on top of these.
        self.api_params: Dict[str, Any] = kwargs.get("api_params") or {}

        if self.use_batch_api and self.BASE_URL is not None:
            logger.warning(
                f"{self.SERVICE_NAME}: use_batch_api=True ignored — "
                f"OpenAI-compatible endpoints do not implement /v1/batches")

        client_kwargs = {"api_key": self.api_key}
        if self.BASE_URL:
            client_kwargs["base_url"] = self.BASE_URL
        self.async_client = AsyncOpenAI(**client_kwargs)
        # Sync client for the blocking batch flow (files + batches endpoints);
        # created lazily so realtime-only use never builds it.
        self._sync_client: Optional[OpenAI] = None
        logger.info(f"Initialized {self.SERVICE_NAME} service with {model.model_id}")

    # ------------------------------------------------------------------
    # Model-specific helpers
    # ------------------------------------------------------------------

    def _build_api_params(
        self, messages: List[Dict], temperature: float, max_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"model": self.model.model_id, "messages": messages}

        if self._accepts_temperature():
            params["temperature"] = temperature

        if self.model.has_quirk(ModelQuirk.USES_MAX_COMPLETION_TOKENS):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens

        if self.api_params:
            params.update(self.api_params)
        if extra:
            params.update(extra)
        return params

    @staticmethod
    def _extract_response(response) -> str:
        choice = response.choices[0]
        text = choice.message.content
        if not text or not text.strip():
            return f"[LLM response filtered out due to: {choice.finish_reason}]"
        return text

    # ------------------------------------------------------------------
    # Message formatting
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
        sem: asyncio.Semaphore,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        is_test: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    params = self._build_api_params(messages, temperature, max_tokens, extra)
                    response = await self.async_client.chat.completions.create(**params)

                    if hasattr(response, "usage") and response.usage:
                        in_tok = response.usage.prompt_tokens or 0
                        out_tok = response.usage.completion_tokens or 0
                        cost = (
                            in_tok * self.model.input_price
                            + out_tok * self.model.output_price
                        ) / 1_000_000
                        self._record_usage(in_tok, out_tok, cost, is_test)

                    return self._extract_response(response)

                except Exception as e:
                    err = str(e)
                    # Account-global failures (invalid key / no credits) can't
                    # recover mid-run — abort the whole run fast instead of
                    # retrying every cell. Checked BEFORE the rate-limit branch
                    # because OpenAI returns `insufficient_quota` as a 429.
                    self._raise_if_account_fatal(e)
                    if is_rate_limit_error(e) and attempt < self.max_retries:
                        wait = _backoff_seconds(attempt)
                        logger.warning(
                            f"Rate limit hit, retry {attempt + 1}/{self.max_retries} "
                            f"in {wait:.1f}s"
                        )
                        await asyncio.sleep(wait)
                        continue
                    self._check_fatal_error(e, self.model.model_id)
                    logger.error(f"{self.SERVICE_NAME} API error: {err}")
                    return make_mechanism_error(err)
        return make_mechanism_error("retries exhausted (unreachable)")

    # ------------------------------------------------------------------
    # Native Batch API (real OpenAI endpoint only)
    #
    # Same three-helper shape as ClaudeService/GoogleService (submit → poll →
    # collect); the JSONL file transport is an implementation detail hidden
    # here: the file is built in memory and uploaded via the Files API —
    # nothing touches disk — and the batch's files are deleted after harvest.
    # ------------------------------------------------------------------

    def _supports_native_batch(self) -> bool:
        # Compatible endpoints (DeepSeek, Z.AI, xAI, …) don't implement
        # /v1/batches — only the real OpenAI endpoint batches.
        return self.BASE_URL is None

    @property
    def sync_client(self) -> OpenAI:
        if self._sync_client is None:
            client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.BASE_URL:
                client_kwargs["base_url"] = self.BASE_URL
            self._sync_client = OpenAI(**client_kwargs)
        return self._sync_client

    def _submit_batch(
        self,
        prepared: List[Tuple[str, List[Dict]]],
        temperature: float,
        max_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ):
        lines = []
        for cid, messages in prepared:
            body = self._build_api_params(messages, temperature, max_tokens, extra)
            lines.append(json.dumps({
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }))
        jsonl = ("\n".join(lines) + "\n").encode("utf-8")

        logger.info(
            f"Submitting {self.SERVICE_NAME} batch with {len(prepared)} requests "
            f"({len(jsonl) / 1e6:.1f} MB JSONL)")
        try:
            input_file = self._retry_rate_limit_sync(
                lambda: self.sync_client.files.create(
                    file=("batch.jsonl", io.BytesIO(jsonl)), purpose="batch"),
                label=f"{self.SERVICE_NAME} files.create ({self.model.model_id})",
            )
            return self._retry_rate_limit_sync(
                lambda: self.sync_client.batches.create(
                    input_file_id=input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                ),
                label=f"{self.SERVICE_NAME} batches.create ({self.model.model_id})",
            )
        except Exception as e:
            # Bad key / empty balance surfaces here at submit time and dooms
            # every request in the run — abort fast, don't fail one task.
            self._raise_if_account_fatal(e)
            raise

    def _poll_until_done(self, batch):
        elapsed = 0
        while batch.status not in _BATCH_TERMINAL_STATUSES:
            if elapsed >= self.batch_timeout:
                raise TimeoutError(
                    f"{self.SERVICE_NAME} batch {batch.id} not done after "
                    f"{self.batch_timeout}s — it keeps running server-side; "
                    f"recover later with harvest_batch_chat('{batch.id}')")
            time.sleep(self.batch_poll_interval)
            elapsed += self.batch_poll_interval
            batch = self._retry_rate_limit_sync(
                lambda: self.sync_client.batches.retrieve(batch.id),
                label=f"{self.SERVICE_NAME} batches.retrieve ({batch.id})",
            )
            counts = batch.request_counts
            logger.info(
                f"Batch {batch.id}: {batch.status} "
                f"(completed={counts.completed}, failed={counts.failed}, "
                f"total={counts.total})")
        return batch

    def _download_jsonl(self, file_id: str) -> List[dict]:
        content = self._retry_rate_limit_sync(
            lambda: self.sync_client.files.content(file_id),
            label=f"{self.SERVICE_NAME} files.content ({file_id})",
        )
        return [json.loads(line) for line in content.text.splitlines() if line.strip()]

    def _collect_results(self, batch, is_test: bool) -> Dict[str, str]:
        """Map custom_id → response text (or mechanism-error string) from a
        terminal batch; records usage at batch price; deletes the batch's files."""
        results: Dict[str, str] = {}

        if batch.status == "failed":
            # Wholesale validation failure — no per-item results exist; the
            # caller fills every id with a mechanism error naming the status.
            errs = "; ".join(
                f"{e.code}: {e.message}" for e in (batch.errors.data or [])
            ) if getattr(batch, "errors", None) else "unknown error"
            logger.error(f"{self.SERVICE_NAME} batch {batch.id} failed: {errs}")

        if batch.output_file_id:
            for entry in self._download_jsonl(batch.output_file_id):
                cid = entry.get("custom_id")
                resp = entry.get("response")
                if entry.get("error") or not resp or resp.get("status_code") != 200:
                    detail = entry.get("error") or {
                        "status_code": resp.get("status_code") if resp else None}
                    results[cid] = make_mechanism_error(f"batch item error: {detail}")
                    continue
                body = resp["body"]
                usage = body.get("usage") or {}
                in_tok = usage.get("prompt_tokens", 0) or 0
                out_tok = usage.get("completion_tokens", 0) or 0
                # Batch API bills at half the realtime list price.
                cost = (
                    in_tok * self.model.input_price
                    + out_tok * self.model.output_price
                ) / 1_000_000 * self.BATCH_COST_DISCOUNT
                self._record_usage(in_tok, out_tok, cost, is_test)
                choice = body["choices"][0]
                text = (choice.get("message") or {}).get("content")
                if not text or not text.strip():
                    text = (f"[LLM response filtered out due to: "
                            f"{choice.get('finish_reason')}]")
                results[cid] = text

        if batch.error_file_id:
            for entry in self._download_jsonl(batch.error_file_id):
                cid = entry.get("custom_id")
                if cid and cid not in results:
                    results[cid] = make_mechanism_error(
                        f"batch item error: {entry.get('error')}")

        self._cleanup_batch_files(batch)
        return results

    def _cleanup_batch_files(self, batch) -> None:
        """Best-effort deletion of the batch's input/output/error files —
        they otherwise persist in the org's file storage."""
        for fid in (batch.input_file_id, batch.output_file_id, batch.error_file_id):
            if fid:
                try:
                    self.sync_client.files.delete(fid)
                except Exception as e:  # noqa: BLE001 — housekeeping only
                    logger.warning(f"could not delete batch file {fid}: {e}")

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
        extra = kwargs.get("api_params")

        prepared = [
            (cid, self._format_conversation(msgs, system_message))
            for cid, msgs in conversations
        ]

        if self._route_to_native_batch(conversations, max_tokens):
            batch = self._submit_batch(prepared, temperature, max_tokens, extra)
            batch = self._poll_until_done(batch)
            results_map = self._collect_results(batch, is_test)
            return [
                (cid, results_map.get(cid, make_mechanism_error(
                    f"missing from batch results (status={batch.status})")))
                for cid, _ in prepared
            ]

        logger.info(
            f"Sending {len(prepared)} requests async (concurrency={self.max_concurrency})"
        )

        async def _run() -> List[str]:
            sem = asyncio.Semaphore(self.max_concurrency)
            tasks = [
                self._one_call(sem, msgs, temperature, max_tokens, is_test, extra)
                for _, msgs in prepared
            ]
            return await asyncio.gather(*tasks)

        responses = asyncio.run(_run())
        return [(cid, resp) for (cid, _), resp in zip(prepared, responses)]

    def chat_structured(
        self,
        prompt: str,
        output_schema: Any,
        system_message: Optional[str] = None,
        *,
        is_test: bool = False,
        **kwargs,
    ) -> Any:
        """One prompt → one `output_schema` instance via `chat.completions.parse`
        (OpenAI structured outputs). Raises on API failure after retries; may
        return None when the model refused or produced no parsable output.

        On OpenAI-COMPATIBLE endpoints support varies by provider: xAI and
        some OpenRouter models accept strict structured outputs, DeepSeek and
        Moonshot do not (their server returns a 400, which raises here)."""
        messages = self._format_conversation([(prompt, None)], system_message)
        params = self._build_api_params(
            messages,
            kwargs.get("temperature", self.temperature),
            kwargs.get("max_tokens", self.max_tokens),
            kwargs.get("api_params"),
        )
        params["response_format"] = output_schema

        async def _call():
            return await self.async_client.chat.completions.parse(**params)

        try:
            response = asyncio.run(self._retry_rate_limit_async(
                _call,
                label=f"{self.SERVICE_NAME} completions.parse ({self.model.model_id})",
            ))
        except Exception as e:
            self._raise_if_account_fatal(e)
            self._check_fatal_error(e, self.model.model_id)
            raise
        if getattr(response, "usage", None):
            in_tok = response.usage.prompt_tokens or 0
            out_tok = response.usage.completion_tokens or 0
            cost = (
                in_tok * self.model.input_price
                + out_tok * self.model.output_price
            ) / 1_000_000
            self._record_usage(in_tok, out_tok, cost, is_test)
        return response.choices[0].message.parsed

    # ------------------------------------------------------------------
    # Resumable batch API — same contract as ClaudeService: submit / status /
    # harvest as separate calls, so bulk work survives process exits. Persist
    # the batch id anywhere and harvest from a later invocation. Real OpenAI
    # endpoint only (compatible endpoints lack /v1/batches).
    # ------------------------------------------------------------------

    def submit_batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Submit a native batch WITHOUT waiting; returns the provider batch id."""
        if self.BASE_URL is not None:
            raise NotImplementedError(
                f"{self.SERVICE_NAME}: /v1/batches exists only on the real "
                f"OpenAI endpoint")
        prepared = [
            (cid, self._format_conversation(msgs, system_message))
            for cid, msgs in conversations
        ]
        batch = self._submit_batch(
            prepared,
            kwargs.get("temperature", self.temperature),
            kwargs.get("max_tokens", self.max_tokens),
            kwargs.get("api_params"),
        )
        return batch.id

    def batch_chat_status(self, batch_id: str) -> str:
        """The provider's batch status ("validating", "in_progress",
        "completed", "failed", "expired", …)."""
        if self.BASE_URL is not None:
            raise NotImplementedError(
                f"{self.SERVICE_NAME}: /v1/batches exists only on the real "
                f"OpenAI endpoint")
        batch = self._retry_rate_limit_sync(
            lambda: self.sync_client.batches.retrieve(batch_id),
            label=f"{self.SERVICE_NAME} batches.retrieve ({batch_id})",
        )
        return batch.status

    def harvest_batch_chat(
        self, batch_id: str, *, is_test: bool = False,
    ) -> Optional[List[Tuple[str, str]]]:
        """Results of a previously submitted batch, or None while still running.

        Failed/expired entries come back as mechanism-error strings (same
        contract as `batch_chat`); usage/cost is recorded at batch price for
        succeeded entries. A wholesale-failed batch returns []."""
        if self.BASE_URL is not None:
            raise NotImplementedError(
                f"{self.SERVICE_NAME}: /v1/batches exists only on the real "
                f"OpenAI endpoint")
        batch = self._retry_rate_limit_sync(
            lambda: self.sync_client.batches.retrieve(batch_id),
            label=f"{self.SERVICE_NAME} batches.retrieve ({batch_id})",
        )
        if batch.status not in _BATCH_TERMINAL_STATUSES:
            return None
        results_map = self._collect_results(batch, is_test)
        return list(results_map.items())
