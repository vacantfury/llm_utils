"""
Anthropic Claude service — uses the native **Message Batches API** for
concurrent processing at 50 % reduced cost.

Flow: submit batch → poll until ``processing_status == "ended"`` → collect
results by ``custom_id``.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic

from ..base_llm_service import BaseLLMService, make_mechanism_error
from ..llm_model import LLMModel
from ..constants import ANTHROPIC_API_KEY
from ..media_utils import encode_image_to_b64
from .._logging import get_logger

logger = get_logger(__name__)


def _extract_text(message) -> str:
    """Safely extract text from a Claude message object."""
    if not message.content:
        return ""
    for block in message.content:
        if hasattr(block, "text"):
            return block.text
    return ""


class ClaudeService(BaseLLMService):
    """Service for Anthropic Claude models."""

    def __init__(self, model: LLMModel, config=None, **kwargs):
        super().__init__(
            max_concurrency=kwargs.pop("max_concurrency", 20),
            max_retries=kwargs.pop("max_retries", 5),
            batch_poll_interval=kwargs.pop("batch_poll_interval", 30),
            batch_timeout=kwargs.pop("batch_timeout", 3600),
        )
        self.model = model
        self.api_key = kwargs.get("api_key") or ANTHROPIC_API_KEY
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 4096)

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

    # ------------------------------------------------------------------
    # Native batch helpers
    # ------------------------------------------------------------------

    def _submit_batch(
        self,
        prepared: List[Tuple[str, List[Dict]]],
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
    ):
        requests = []
        for item_id, messages in prepared:
            params: Dict[str, Any] = {
                "model": self.model.model_id,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            # Temperature is gated by the shared quirk rule (Opus 4.7+ rejects
            # it with a 400). See BaseLLMService._accepts_temperature.
            if self._accepts_temperature():
                params["temperature"] = temperature
            if system_message:
                params["system"] = system_message
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
            raise

    def _poll_until_done(self, batch):
        elapsed = 0
        while batch.processing_status != "ended":
            if elapsed >= self.batch_timeout:
                raise TimeoutError(
                    f"Claude batch {batch.id} not done after {self.batch_timeout}s"
                )
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

    def _collect_results(self, batch, is_test: bool) -> Dict[str, str]:
        results: Dict[str, str] = {}
        for entry in self.client.messages.batches.results(batch.id):
            cid = entry.custom_id
            result = entry.result
            if result.type == "succeeded":
                msg = result.message
                text = _extract_text(msg)
                if hasattr(msg, "usage") and msg.usage:
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
                results[cid] = make_mechanism_error(
                    f"batch result type={result.type}")
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

        Overrides the base (which funnels singles through `batch_chat`): the
        Batches API is for bulk work — 50% cheaper but queued for minutes,
        which makes an interactive single call glacial. Bulk callers still get
        batch pricing via `batch_chat` below.
        """
        params: Dict[str, Any] = {
            "model": self.model.model_id,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._accepts_temperature():
            params["temperature"] = kwargs.get("temperature", self.temperature)
        if system_message:
            params["system"] = system_message
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

    def batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        system_message: Optional[str] = None,
        is_test: bool = False,
        **kwargs,
    ) -> List[Tuple[str, str]]:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        prepared = [
            (cid, self._format_conversation(msgs))
            for cid, msgs in conversations
        ]

        batch = self._submit_batch(prepared, system_message, temperature, max_tokens)
        batch = self._poll_until_done(batch)
        results_map = self._collect_results(batch, is_test)

        return [
            (cid, results_map.get(cid, make_mechanism_error("missing from batch results")))
            for cid, _ in prepared
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
        params: Dict[str, Any] = {
            "model": self.model.model_id,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
            "output_format": output_schema,
        }
        if self._accepts_temperature():
            params["temperature"] = kwargs.get("temperature", self.temperature)
        if system_message:
            params["system"] = system_message
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
        """Submit a batch WITHOUT waiting; returns the provider batch id."""
        prepared = [
            (cid, self._format_conversation(msgs))
            for cid, msgs in conversations
        ]
        batch = self._submit_batch(
            prepared,
            system_message,
            kwargs.get("temperature", self.temperature),
            kwargs.get("max_tokens", self.max_tokens),
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

        Failed entries come back as mechanism-error strings (same contract as
        `batch_chat`); usage/cost is recorded for succeeded entries."""
        batch = self._retry_rate_limit_sync(
            lambda: self.client.messages.batches.retrieve(batch_id),
            label=f"Anthropic batches.retrieve ({batch_id})",
        )
        if batch.processing_status != "ended":
            return None
        results_map = self._collect_results(batch, is_test)
        return list(results_map.items())
