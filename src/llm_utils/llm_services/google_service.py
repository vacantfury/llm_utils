"""
Google Gemini service — uses the native **Batch API** with inline requests
for concurrent processing at 50 % reduced cost.

Flow: build inline requests → ``client.batches.create`` → poll until
``JOB_STATE_SUCCEEDED`` → collect ``inlined_responses``.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

try:  # Pillow is only needed for image messages; text-only use works without it
    import PIL.Image
except ImportError:  # pragma: no cover
    PIL = None
import google.genai as genai

from ..base_llm_service import BaseLLMService, make_mechanism_error
from ..llm_model import LLMModel, ModelQuirk
from ..constants import GOOGLE_API_KEY
from .._logging import get_logger

logger = get_logger(__name__)

# Thinking models (THINKING_SHARES_OUTPUT_BUDGET quirk) spend max_output_tokens
# on thought before a word of visible text appears — the caller's max_tokens
# means VISIBLE text, so grant this much extra for the thinking share.
_THINKING_HEADROOM = 8192

_TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
})


class GoogleService(BaseLLMService):
    """Service for Google Gemini models."""

    def __init__(self, model: LLMModel, config=None, **kwargs):
        super().__init__(
            max_concurrency=kwargs.pop("max_concurrency", 20),
            max_retries=kwargs.pop("max_retries", 5),
            batch_poll_interval=kwargs.pop("batch_poll_interval", 30),
            batch_timeout=kwargs.pop("batch_timeout", 3600),
        )
        self.model = model
        self.api_key = kwargs.get("api_key") or GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY in .env "
                "or pass api_key parameter"
            )
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.top_p = kwargs.get("top_p", 1.0)

        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"Initialized Google service with {model.model_id}")

    def _output_budget(self, max_tokens: int) -> int:
        """The effective max_output_tokens: thinking models get headroom so
        thought tokens can't starve the caller's visible-text budget."""
        if self.model.has_quirk(ModelQuirk.THINKING_SHARES_OUTPUT_BUDGET):
            return max_tokens + _THINKING_HEADROOM
        return max_tokens

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _build_content_parts(
        messages: List[Tuple[str, Optional[Any]]],
    ) -> list:
        """Convert conversation messages to a flat list of content parts
        (strings and PIL images) that Google's API accepts."""
        parts: list = []
        for text, image in messages:
            if image is None:
                parts.append(text)
            else:
                images = image if isinstance(image, list) else [image]
                if PIL is None:
                    raise RuntimeError("image messages need Pillow: uv add pillow")
                for img in images:
                    if img is None:
                        continue
                    if isinstance(img, PIL.Image.Image):
                        parts.append(img)
                    else:
                        parts.append(PIL.Image.open(str(img)))
                parts.append(text)
        return parts

    # ------------------------------------------------------------------
    # Native batch helpers
    # ------------------------------------------------------------------

    def _build_inline_requests(
        self,
        prepared: List[Tuple[str, list]],
        temperature: float,
        max_tokens: int,
        system_message: Optional[str] = None,
    ) -> list:
        inline_requests = []
        for _item_id, parts in prepared:
            contents = []
            for p in parts:
                if isinstance(p, str) or (PIL is not None and isinstance(p, PIL.Image.Image)):
                    contents.append(p)
                else:
                    contents.append(str(p))

            config: Dict[str, Any] = {"max_output_tokens": self._output_budget(max_tokens)}
            # Same shared quirk rule as the other providers (Gemini accepts
            # temperature today, so this is normally on — but a future
            # reasoning-only Gemini marked NO_CUSTOM_TEMPERATURE is handled here).
            if self._accepts_temperature():
                config["temperature"] = temperature
                if temperature > 0:
                    config["top_p"] = self.top_p
            if system_message:
                config["system_instruction"] = system_message

            inline_requests.append({
                "contents": contents,
                "config": config,
            })
        return inline_requests

    def _submit_batch(self, inline_requests: list):
        logger.info(f"Submitting Google batch with {len(inline_requests)} inline requests")
        try:
            return self._retry_rate_limit_sync(
                lambda: self.client.batches.create(
                    model=self.model.model_id,
                    src=inline_requests,
                    config={"display_name": f"batch-{self.model.model_id}"},
                ),
                label=f"Google batches.create ({self.model.model_id})",
            )
        except Exception as e:
            # Bad key ("API key not valid") / disabled billing surfaces here at
            # submit time and dooms every request — abort fast, don't fail one task.
            self._raise_if_account_fatal(e)
            raise

    def _poll_until_done(self, batch_job):
        elapsed = 0
        while batch_job.state.name not in _TERMINAL_STATES:
            if elapsed >= self.batch_timeout:
                raise TimeoutError(
                    f"Google batch {batch_job.name} not done after {self.batch_timeout}s"
                )
            time.sleep(self.batch_poll_interval)
            elapsed += self.batch_poll_interval
            batch_job = self._retry_rate_limit_sync(
                lambda: self.client.batches.get(name=batch_job.name),
                label=f"Google batches.get ({batch_job.name})",
            )
            logger.info(f"Batch {batch_job.name}: {batch_job.state.name}")

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(
                f"Google batch failed with state: {batch_job.state.name}"
            )
        return batch_job

    def _collect_results(
        self,
        batch_job,
        prepared: List[Tuple[str, list]],
        is_test: bool,
    ) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []

        for (item_id, _), inline_resp in zip(
            prepared, batch_job.dest.inlined_responses
        ):
            if inline_resp.response:
                resp = inline_resp.response
                text = resp.text if resp.text else "[Empty response]"

                if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                    um = resp.usage_metadata
                    in_tok = getattr(um, "prompt_token_count", 0) or 0
                    out_tok = getattr(um, "candidates_token_count", 0) or 0
                    cost = (
                        in_tok * self.model.input_price
                        + out_tok * self.model.output_price
                    ) / 1_000_000
                    self._record_usage(in_tok, out_tok, cost, is_test)
            else:
                # No inline response = the item errored (a content/safety block
                # instead returns a response with empty text → "[Empty response]"
                # above, kept as a refusal). So this is a mechanism failure.
                text = make_mechanism_error("no response in batch result")

            results.append((item_id, text))
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
        """One prompt → one response, via the REAL-TIME generate_content API.

        Overrides the base (which funnels singles through `batch_chat`): the
        Batch API queues for minutes — right for bulk work, wrong for an
        interactive call (same fix as ClaudeService.chat)."""
        config: Dict[str, Any] = {
            "max_output_tokens": self._output_budget(kwargs.get("max_tokens", self.max_tokens))
        }
        if self._accepts_temperature():
            temperature = kwargs.get("temperature", self.temperature)
            config["temperature"] = temperature
            if temperature > 0:
                config["top_p"] = self.top_p
        if system_message:
            config["system_instruction"] = system_message
        try:
            resp = self._retry_rate_limit_sync(
                lambda: self.client.models.generate_content(
                    model=self.model.model_id, contents=prompt, config=config,
                ),
                label=f"Google generate_content ({self.model.model_id})",
            )
        except Exception as e:  # noqa: BLE001 — same contract as batch path
            self._raise_if_account_fatal(e)
            self._check_fatal_error(e, self.model.model_id)
            logger.error(f"Google API error: {e}")
            return make_mechanism_error(str(e))
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            um = resp.usage_metadata
            in_tok = getattr(um, "prompt_token_count", 0) or 0
            out_tok = getattr(um, "candidates_token_count", 0) or 0
            cost = (
                in_tok * self.model.input_price
                + out_tok * self.model.output_price
            ) / 1_000_000
            self._record_usage(in_tok, out_tok, cost, is_test)
        return resp.text if resp.text else "[Empty response]"

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
            (cid, self._build_content_parts(msgs))
            for cid, msgs in conversations
        ]

        inline_reqs = self._build_inline_requests(
            prepared, temperature, max_tokens, system_message
        )
        batch_job = self._submit_batch(inline_reqs)
        batch_job = self._poll_until_done(batch_job)
        return self._collect_results(batch_job, prepared, is_test)
