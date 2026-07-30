"""
Anthropic Claude service — two serving routes behind one ``batch_chat``:

- **realtime**: concurrent Messages-API requests (full list price, seconds
  turnaround);
- **native Message Batches API**: submit → poll ``processing_status ==
  "ended"`` → collect by ``custom_id``, at 50 % price.

``batch_chat`` auto-routes by estimated job cost (``batch_threshold_usd``);
``use_batch_api`` forces either route. ``chat`` / ``chat_structured`` are
always realtime.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic

from ..base_llm_service import BaseLLMService, make_mechanism_error
from ..llm_model import LLMModel, ModelQuirk
from ..media_utils import encode_image_to_b64
from .._logging import get_logger

logger = get_logger(__name__)

# Thinking models (THINKING_SHARES_OUTPUT_BUDGET quirk) spend max_tokens on
# thought before a word of visible text appears — the caller's max_tokens
# means VISIBLE text, so grant this much extra for the thinking share.
# (Same convention as GoogleService.)
_THINKING_HEADROOM = 8192


def _extract_text(message) -> str:
    """Extract the visible text from a Claude message object: all text blocks
    joined, in order (thinking blocks carry ``.thinking``, not ``.text``, and
    are skipped)."""
    if not message.content:
        return ""
    return "".join(
        block.text for block in message.content if hasattr(block, "text")
    )


class ClaudeService(BaseLLMService):
    """Service for Anthropic Claude models."""

    def __init__(self, model: LLMModel, **kwargs):
        super().__init__(
            max_concurrency=kwargs.pop("max_concurrency", 20),
            max_retries=kwargs.pop("max_retries", 5),
            batch_poll_interval=kwargs.pop("batch_poll_interval", 30),
            batch_timeout=kwargs.pop("batch_timeout", 3600),
            use_batch_api=kwargs.pop("use_batch_api", None),
            batch_threshold_usd=kwargs.pop("batch_threshold_usd", None),
        )
        self.model = model
        # Read the env var at construction (not import) so a key exported
        # after `import llm_utils` still works — same timing as OpenAIService.
        self.api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        # Extra request params merged verbatim into every API call
        # (parity with the OpenAI family's api_params seam).
        self.api_params: Dict[str, Any] = kwargs.get("api_params") or {}
        # Batch ids whose usage this process has already recorded — guards
        # harvest_batch_chat against double-billing the ledger when the same
        # id is harvested twice in one process.
        self._usage_recorded_batches: set = set()

        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)
        else:
            # No explicit key: fall back to the SDK's own credential resolution
            # (ANTHROPIC_AUTH_TOKEN, CLI-login keychain credentials, …). Still
            # fail EARLY — at construction, like the explicit-key path — when
            # that resolves nothing either.
            no_creds = ValueError(
                "Anthropic credentials not found. Set ANTHROPIC_API_KEY in "
                ".env, pass api_key, or authenticate the SDK (auth token / "
                "CLI login)"
            )
            try:
                self.client = Anthropic()
            except Exception as e:
                raise no_creds from e
            if not any(
                getattr(self.client, attr, None)
                for attr in ("api_key", "auth_token", "credentials")
            ):
                raise no_creds
        logger.info(f"Initialized Claude service with {model.model_id}")

    def _supports_native_batch(self) -> bool:
        return True

    def _output_budget(self, max_tokens: int) -> int:
        """Effective max_tokens: always-thinking models (Opus 5 / Fable 5)
        get headroom so thought tokens can't starve the caller's visible-text
        budget. Clamped to the model's hard output cap when the registry
        declares one."""
        if self.model.has_quirk(ModelQuirk.THINKING_SHARES_OUTPUT_BUDGET):
            budget = max_tokens + _THINKING_HEADROOM
        else:
            budget = max_tokens
        hard_cap = getattr(self.model, "max_output_tokens", None)
        return min(budget, hard_cap) if hard_cap else budget

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_conversation(
        messages: List[Tuple[str, Optional[Any]]],
    ) -> List[Dict[str, Any]]:
        anthropic_msgs: List[Dict[str, Any]] = []
        for text, image in messages:
            if image is None:
                anthropic_msgs.append({"role": "user", "content": text})
            else:
                images = image if isinstance(image, list) else [image]
                content: list = []
                for img in images:
                    if img is not None:
                        b64, media_type = encode_image_to_b64(img)
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        })
                content.append({"type": "text", "text": text})
                anthropic_msgs.append({"role": "user", "content": content})
        return anthropic_msgs

    def _build_request_params(
        self,
        messages: List[Dict],
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model.model_id,
            "max_tokens": self._output_budget(max_tokens),
            "messages": messages,
        }
        # Temperature is gated by the shared quirk rule (Opus 4.7+ rejects
        # it with a 400). See BaseLLMService._accepts_temperature.
        if self._accepts_temperature():
            params["temperature"] = temperature
        if system_message:
            params["system"] = system_message
        if self.api_params:
            params.update(self.api_params)
        if extra:
            params.update(extra)
        return params

    # ------------------------------------------------------------------
    # Realtime route
    # ------------------------------------------------------------------

    def _realtime_one(
        self,
        messages: List[Dict],
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        is_test: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """One realtime Messages-API call → response text or mechanism-error
        string. Account-fatal and model-404 errors raise (abort the run)."""
        params = self._build_request_params(
            messages, system_message, temperature, max_tokens, extra)
        try:
            msg = self._retry_rate_limit_sync(
                lambda: self.client.messages.create(**params),
                label=f"Anthropic messages.create ({self.model.model_id})",
            )
        except Exception as e:  # noqa: BLE001 — same contract as batch path
            self._raise_if_account_fatal(e)
            self._check_fatal_error(e, self.model.model_id)
            logger.error(f"Claude API error: {e}")
            return make_mechanism_error(str(e))
        if hasattr(msg, "usage") and msg.usage:
            in_tok = msg.usage.input_tokens or 0
            out_tok = msg.usage.output_tokens or 0
            cost = (
                in_tok * self.model.input_price
                + out_tok * self.model.output_price
            ) / 1_000_000
            self._record_usage(in_tok, out_tok, cost, is_test)
        return _extract_text(msg)

    # ------------------------------------------------------------------
    # Native batch helpers
    # ------------------------------------------------------------------

    def _submit_batch(
        self,
        prepared: List[Tuple[str, List[Dict]]],
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ):
        requests = []
        for item_id, messages in prepared:
            params = self._build_request_params(
                messages, system_message, temperature, max_tokens, extra)
            requests.append({"custom_id": item_id, "params": params})

        logger.info(f"Submitting Claude batch with {len(requests)} requests")
        try:
            return self._retry_rate_limit_sync(
                lambda: self.client.messages.batches.create(requests=requests),
                label=f"Anthropic batches.create ({self.model.model_id})",
            )
        except Exception as e:
            # Bad key / empty credit balance surfaces here at submit time and
            # dooms every request in the run — abort fast, don't fail one task.
            self._raise_if_account_fatal(e)
            self._check_fatal_error(e, self.model.model_id)
            raise

    def _poll_until_done(self, batch):
        elapsed = 0
        while batch.processing_status != "ended":
            if elapsed >= self.batch_timeout:
                raise TimeoutError(
                    f"Claude batch {batch.id} not done after "
                    f"{self.batch_timeout}s — it keeps running server-side; "
                    f"recover later with harvest_batch_chat('{batch.id}')")
            time.sleep(self.batch_poll_interval)
            elapsed += self.batch_poll_interval
            batch = self._retry_rate_limit_sync(
                lambda: self.client.messages.batches.retrieve(batch.id),
                label=f"Anthropic batches.retrieve ({batch.id})",
            )

            counts = batch.request_counts
            logger.info(
                f"Batch {batch.id}: {batch.processing_status} "
                f"(succeeded={counts.succeeded}, processing={counts.processing}, "
                f"errored={counts.errored})"
            )
        return batch

    def _collect_results(
        self, batch, is_test: bool, record_usage: bool = True,
    ) -> Dict[str, str]:
        results: Dict[str, str] = {}
        # Materialize the result stream inside the retry wrapper: the stream
        # is itself a network transfer, and a transient error mid-iteration
        # would otherwise raise out of an already-paid batch.
        entries = self._retry_rate_limit_sync(
            lambda: list(self.client.messages.batches.results(batch.id)),
            label=f"Anthropic batches.results ({batch.id})",
        )
        for entry in entries:
            cid = entry.custom_id
            result = entry.result
            if result.type == "succeeded":
                msg = result.message
                text = _extract_text(msg)
                if record_usage and hasattr(msg, "usage") and msg.usage:
                    in_tok = msg.usage.input_tokens or 0
                    out_tok = msg.usage.output_tokens or 0
                    # Batches API bills at half the realtime list price.
                    cost = (
                        in_tok * self.model.input_price
                        + out_tok * self.model.output_price
                    ) / 1_000_000 * self.BATCH_COST_DISCOUNT
                    self._record_usage(in_tok, out_tok, cost, is_test)
                results[cid] = text
            else:
                # Non-"succeeded" item = a real processing failure (errored /
                # expired / canceled). Content refusals come back as
                # type="succeeded" with refusal text, so this is a mechanism error.
                detail = getattr(result, "error", None)
                results[cid] = make_mechanism_error(
                    f"batch result type={result.type}"
                    + (f": {detail}" if detail else ""))
        self._usage_recorded_batches.add(batch.id)
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        *,
        is_test: bool = False,
        **kwargs,
    ) -> str:
        """One prompt → one response, via the REAL-TIME Messages API.

        Overrides the base (which funnels singles through `batch_chat`) to
        skip the routing estimate entirely: an interactive single call always
        goes realtime."""
        return self._realtime_one(
            self._format_conversation([(prompt, None)]),
            system_message,
            kwargs.get("temperature", self.temperature),
            kwargs.get("max_tokens", self.max_tokens),
            is_test,
            kwargs.get("api_params"),
        )

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

        # Per-item formatting guard: one unreadable image path must fail that
        # item, not the whole call (and it fails before any spend).
        prepared: List[Tuple[str, List[Dict]]] = []
        format_failed: Dict[str, str] = {}
        for cid, msgs in conversations:
            try:
                prepared.append((cid, self._format_conversation(msgs)))
            except Exception as e:  # noqa: BLE001 — per-item contract
                format_failed[cid] = make_mechanism_error(
                    f"message formatting failed: {e}")

        if prepared and self._route_to_native_batch(conversations, max_tokens):
            batch = self._submit_batch(
                prepared, system_message, temperature, max_tokens, extra)
            batch = self._poll_until_done(batch)
            results_map = self._collect_results(batch, is_test)
        elif prepared:
            logger.info(
                f"Sending {len(prepared)} realtime requests "
                f"(concurrency={self.max_concurrency})")
            with ThreadPoolExecutor(
                max_workers=max(1, min(self.max_concurrency, len(prepared)))
            ) as pool:
                futures = {
                    cid: pool.submit(
                        self._realtime_one, msgs, system_message,
                        temperature, max_tokens, is_test, extra)
                    for cid, msgs in prepared
                }
                results_map = {cid: f.result() for cid, f in futures.items()}
        else:
            results_map = {}

        return [
            (cid, format_failed.get(cid)
             or results_map.get(
                 cid, make_mechanism_error("missing from batch results")))
            for cid, _ in conversations
        ]

    def chat_structured(
        self,
        prompt: str,
        output_schema: Any,
        system_message: Optional[str] = None,
        *,
        is_test: bool = False,
        **kwargs,
    ) -> Any:
        """One prompt → one `output_schema` instance via `messages.parse`
        (real-time, not batch). Raises on API failure after retries; returns
        None when the model's output failed schema validation."""
        params = self._build_request_params(
            self._format_conversation([(prompt, None)]),
            system_message,
            kwargs.get("temperature", self.temperature),
            kwargs.get("max_tokens", self.max_tokens),
            kwargs.get("api_params"),
        )
        params["output_format"] = output_schema
        try:
            response = self._retry_rate_limit_sync(
                lambda: self.client.messages.parse(**params),
                label=f"Anthropic messages.parse ({self.model.model_id})",
            )
        except Exception as e:
            self._raise_if_account_fatal(e)
            self._check_fatal_error(e, self.model.model_id)
            raise
        if hasattr(response, "usage") and response.usage:
            in_tok = response.usage.input_tokens or 0
            out_tok = response.usage.output_tokens or 0
            cost = (
                in_tok * self.model.input_price
                + out_tok * self.model.output_price
            ) / 1_000_000
            self._record_usage(in_tok, out_tok, cost, is_test)
        return response.parsed_output

    # ------------------------------------------------------------------
    # Resumable batch API — submit / status / harvest as separate calls, so
    # bulk work survives process exits: persist the batch id anywhere and
    # harvest from a later invocation. `batch_chat` above is the blocking
    # submit-poll-harvest composition of the same pieces.
    # ------------------------------------------------------------------

    def submit_batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Submit a batch WITHOUT waiting; returns the provider batch id.
        Always uses the native Batches API regardless of job size."""
        prepared = [
            (cid, self._format_conversation(msgs))
            for cid, msgs in conversations
        ]
        batch = self._submit_batch(
            prepared,
            system_message,
            kwargs.get("temperature", self.temperature),
            kwargs.get("max_tokens", self.max_tokens),
            kwargs.get("api_params"),
        )
        return batch.id

    def batch_chat_status(self, batch_id: str) -> str:
        """The provider's processing status ("in_progress", "ended", …)."""
        batch = self._retry_rate_limit_sync(
            lambda: self.client.messages.batches.retrieve(batch_id),
            label=f"Anthropic batches.retrieve ({batch_id})",
        )
        return batch.processing_status

    def harvest_batch_chat(
        self, batch_id: str, *, is_test: bool = False,
    ) -> Optional[List[Tuple[str, str]]]:
        """Results of a previously submitted batch, or None while still running.

        Failed entries come back as mechanism-error strings; results arrive in
        PROVIDER order (only the batch id is known here, so input order cannot
        be reconstructed — match by your own ids). Usage/cost is recorded for
        succeeded entries the FIRST time this process harvests a given batch
        id; repeat harvests of the same id return results without re-recording.
        (A harvest from a *different* process records again — persist ledger
        state accordingly.)"""
        batch = self._retry_rate_limit_sync(
            lambda: self.client.messages.batches.retrieve(batch_id),
            label=f"Anthropic batches.retrieve ({batch_id})",
        )
        if batch.processing_status != "ended":
            return None
        results_map = self._collect_results(
            batch, is_test,
            record_usage=batch_id not in self._usage_recorded_batches)
        return list(results_map.items())
