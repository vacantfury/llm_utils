"""
Google Gemini service — two serving routes behind one ``batch_chat``:

- **realtime**: concurrent ``generate_content`` requests (full list price,
  seconds turnaround);
- **native Batch API** with inline requests: ``client.batches.create`` →
  poll to a terminal state → collect ``inlined_responses``, at 50 % price.

``batch_chat`` auto-routes by estimated job cost (``batch_threshold_usd``);
``use_batch_api`` forces either route. ``chat`` is always realtime.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Tuple

try:  # Pillow is only needed for image messages; text-only use works without it
    import PIL.Image
except ImportError:  # pragma: no cover
    PIL = None
import google.genai as genai

from ..base_llm_service import BaseLLMService, make_mechanism_error
from ..call_ledger import STATUS_TIMEOUT
from ..llm_model import LLMModel, ModelQuirk
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


def _usage_tokens(um) -> Tuple[int, int]:
    """(input_tokens, output_tokens) from usage metadata. Thought tokens bill
    and cap as OUTPUT tokens (see the registry quirk note) but Google reports
    them separately from ``candidates_token_count`` — sum both."""
    in_tok = getattr(um, "prompt_token_count", 0) or 0
    out_tok = (getattr(um, "candidates_token_count", 0) or 0) + (
        getattr(um, "thoughts_token_count", 0) or 0)
    return in_tok, out_tok


class GoogleService(BaseLLMService):
    """Service for Google Gemini models."""

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
        self.api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY in .env "
                "or pass api_key parameter"
            )
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.top_p = kwargs.get("top_p", 1.0)

        self.client = genai.Client(api_key=self.api_key)
        # Batch names whose outcome rows this process has already recorded —
        # guards harvest_batch_chat against recording the same job twice
        # (same guard as ClaudeService).
        self._usage_recorded_batches: set = set()
        logger.info(f"Initialized Google service with {model.model_id}")

    def _supports_native_batch(self) -> bool:
        return True

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

    def _build_config(
        self,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str],
    ) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "max_output_tokens": self._output_budget(max_tokens)}
        # Same shared quirk rule as the other providers (Gemini accepts
        # temperature today, so this is normally on — but a future
        # reasoning-only Gemini marked NO_CUSTOM_TEMPERATURE is handled here).
        if self._accepts_temperature():
            config["temperature"] = temperature
            if temperature > 0:
                config["top_p"] = self.top_p
        if system_message:
            config["system_instruction"] = system_message
        return config

    # ------------------------------------------------------------------
    # Realtime route
    # ------------------------------------------------------------------

    def _realtime_one(
        self,
        contents: Any,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        is_test: bool,
    ) -> str:
        """One realtime generate_content call → response text or
        mechanism-error string. Account-fatal / model-404 errors raise."""
        config = self._build_config(temperature, max_tokens, system_message)
        try:
            resp = self._retry_rate_limit_sync(
                lambda: self.client.models.generate_content(
                    model=self.model.model_id, contents=contents, config=config,
                ),
                label=f"Google generate_content ({self.model.model_id})",
            )
        except Exception as e:  # noqa: BLE001 — same contract as batch path
            self._record_failure(e, is_test=is_test)
            self._raise_if_account_fatal(e)
            self._check_fatal_error(e, self.model.model_id)
            logger.error(f"Google API error: {e}")
            return make_mechanism_error(str(e))
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            in_tok, out_tok = _usage_tokens(resp.usage_metadata)
            cost = (
                in_tok * self.model.input_price
                + out_tok * self.model.output_price
            ) / 1_000_000
            self._record_usage(in_tok, out_tok, cost, is_test)
        return resp.text if resp.text else "[Empty response]"

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
            inline_requests.append({
                "contents": contents,
                "config": self._build_config(
                    temperature, max_tokens, system_message),
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
            # A failed submit is one failed call (no item reached the model).
            self._record_failure(e)
            # Bad key ("API key not valid") / disabled billing surfaces here at
            # submit time and dooms every request — abort fast, don't fail one task.
            self._raise_if_account_fatal(e)
            self._check_fatal_error(e, self.model.model_id)
            raise

    @staticmethod
    def _job_state(batch_job) -> str:
        state = getattr(batch_job, "state", None)
        return getattr(state, "name", None) or "JOB_STATE_UNSPECIFIED"

    def _poll_until_done(self, batch_job):
        """Poll to ANY terminal state and return the job — failed / cancelled
        / expired jobs are returned too, so partial results (already billed)
        can be collected instead of thrown away."""
        elapsed = 0
        while self._job_state(batch_job) not in _TERMINAL_STATES:
            if elapsed >= self.batch_timeout:
                raise TimeoutError(
                    f"Google batch {batch_job.name} not done after "
                    f"{self.batch_timeout}s — it keeps running server-side; "
                    f"recover later with harvest_batch_chat('{batch_job.name}')")
            time.sleep(self.batch_poll_interval)
            elapsed += self.batch_poll_interval
            batch_job = self._retry_rate_limit_sync(
                lambda: self.client.batches.get(name=batch_job.name),
                label=f"Google batches.get ({batch_job.name})",
            )
            logger.info(f"Batch {batch_job.name}: {self._job_state(batch_job)}")
        return batch_job

    @staticmethod
    def _state_class(state: str) -> str:
        """Batch-level error class: JOB_STATE_FAILED → batch_failed."""
        return "batch_" + state.lower().removeprefix("job_state_")

    @staticmethod
    def _request_count(batch_job) -> Optional[int]:
        """How many requests the job was submitted with: the inlined source
        when the job object carries it, else the completion stats; None when
        the API object says neither."""
        src = getattr(batch_job, "src", None)
        inlined_src = getattr(src, "inlined_requests", None) if src else None
        if inlined_src:
            return len(inlined_src)
        stats = getattr(batch_job, "completion_stats", None)
        if stats is not None:
            counts = [getattr(stats, f, None) for f in
                      ("successful_count", "failed_count", "incomplete_count")]
            if any(isinstance(c, int) for c in counts):
                return sum(c for c in counts if isinstance(c, int))
        return None

    def _collect_results(
        self,
        batch_job,
        item_ids: List[str],
        is_test: bool,
        record_usage: bool = True,
    ) -> List[Tuple[str, str]]:
        """Zip item ids against the job's inlined responses (Google's inline
        batch carries no custom_id — correspondence is positional). Missing
        responses — count mismatch, or a failed/expired job with no output —
        become mechanism errors naming the job state, never silent drops.
        Usage and failed-call rows are recorded unless ``record_usage`` is
        False."""
        state = self._job_state(batch_job)
        if state != "JOB_STATE_SUCCEEDED":
            logger.error(f"Google batch {batch_job.name} ended {state}")

        dest = getattr(batch_job, "dest", None)
        inlined = (getattr(dest, "inlined_responses", None) or []) if dest else []
        if len(inlined) > len(item_ids):
            logger.warning(
                f"Google batch returned {len(inlined)} responses for "
                f"{len(item_ids)} requests — extra responses ignored")

        results: List[Tuple[str, str]] = []
        for item_id, inline_resp in zip_longest(
            item_ids, inlined[:len(item_ids)]
        ):
            if inline_resp is None:
                if record_usage:
                    self._record_failure(
                        "batch_missing" if state == "JOB_STATE_SUCCEEDED"
                        else self._state_class(state),
                        status=STATUS_TIMEOUT if state == "JOB_STATE_EXPIRED" else None,
                        is_test=is_test)
                results.append((item_id, make_mechanism_error(
                    f"missing from batch results (state={state})")))
                continue
            if inline_resp.response:
                resp = inline_resp.response
                text = resp.text if resp.text else "[Empty response]"

                if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                    in_tok, out_tok = _usage_tokens(resp.usage_metadata)
                    # Batch mode bills at half the realtime list price.
                    cost = (
                        in_tok * self.model.input_price
                        + out_tok * self.model.output_price
                    ) / 1_000_000 * self.BATCH_COST_DISCOUNT
                    if record_usage:
                        self._record_usage(in_tok, out_tok, cost, is_test)
            else:
                # No inline response = the item errored (a content/safety block
                # instead returns a response with empty text → "[Empty response]"
                # above, kept as a refusal). So this is a mechanism failure.
                detail = getattr(inline_resp, "error", None)
                if record_usage:
                    self._record_failure("batch_item_error", is_test=is_test)
                text = make_mechanism_error(
                    "no response in batch result"
                    + (f": {detail}" if detail else ""))

            results.append((item_id, text))
        self._usage_recorded_batches.add(batch_job.name)
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

        Overrides the base (which funnels singles through `batch_chat`) to
        skip the routing estimate entirely: an interactive single call always
        goes realtime."""
        return self._realtime_one(
            prompt,
            system_message,
            kwargs.get("temperature", self.temperature),
            kwargs.get("max_tokens", self.max_tokens),
            is_test,
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

        # Per-item formatting guard: one unreadable image path must fail that
        # item, not the whole call (and it fails before any spend).
        prepared: List[Tuple[str, list]] = []
        format_failed: Dict[str, str] = {}
        for cid, msgs in conversations:
            try:
                prepared.append((cid, self._build_content_parts(msgs)))
            except Exception as e:  # noqa: BLE001 — per-item contract
                format_failed[cid] = make_mechanism_error(
                    f"message formatting failed: {e}")

        if prepared and self._route_to_native_batch(conversations, max_tokens):
            inline_reqs = self._build_inline_requests(
                prepared, temperature, max_tokens, system_message
            )
            batch_job = self._submit_batch(inline_reqs)
            batch_job = self._poll_until_done(batch_job)
            results_map = dict(self._collect_results(
                batch_job, [cid for cid, _ in prepared], is_test))
        elif prepared:
            logger.info(
                f"Sending {len(prepared)} realtime requests "
                f"(concurrency={self.max_concurrency})")
            with ThreadPoolExecutor(
                max_workers=max(1, min(self.max_concurrency, len(prepared)))
            ) as pool:
                futures = {
                    cid: pool.submit(
                        self._realtime_one, parts, system_message,
                        temperature, max_tokens, is_test)
                    for cid, parts in prepared
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

    # ------------------------------------------------------------------
    # Resumable batch API — submit / status / harvest as separate calls.
    # CAVEAT vs the Claude/OpenAI trios: Google's inline batch carries no
    # custom_id, so results are POSITIONAL. `submit_batch_chat` preserves
    # your submission order server-side; `harvest_batch_chat` returns ids
    # "0", "1", … by position — keep your own id list from submit time and
    # zip by index.
    # ------------------------------------------------------------------

    def submit_batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[Any]]]]],
        system_message: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Submit a native batch WITHOUT waiting; returns the provider job
        name. Always uses the native Batch API regardless of job size."""
        prepared = [
            (cid, self._build_content_parts(msgs))
            for cid, msgs in conversations
        ]
        inline_reqs = self._build_inline_requests(
            prepared,
            kwargs.get("temperature", self.temperature),
            kwargs.get("max_tokens", self.max_tokens),
            system_message,
        )
        batch_job = self._submit_batch(inline_reqs)
        return batch_job.name

    def batch_chat_status(self, batch_id: str) -> str:
        """The provider's job state ("JOB_STATE_RUNNING",
        "JOB_STATE_SUCCEEDED", …)."""
        batch_job = self._retry_rate_limit_sync(
            lambda: self.client.batches.get(name=batch_id),
            label=f"Google batches.get ({batch_id})",
        )
        return self._job_state(batch_job)

    def harvest_batch_chat(
        self, batch_id: str, *, is_test: bool = False,
    ) -> Optional[List[Tuple[str, str]]]:
        """Results of a previously submitted batch, or None while running.

        Ids are POSITIONAL ("0", "1", … in submission order) — Google's
        inline batch has no custom_id, so keep your own id list from submit
        time and zip by index. Failed entries come back as mechanism-error
        strings; usage/cost is recorded at batch price for succeeded entries
        and one failed-call row per failed or never-answered request, the
        FIRST time this process harvests a given job (a harvest from a
        different process records again). A failed or expired job with no
        responses records one row per submitted request when the job object
        reports the count, else ONE row for the whole job."""
        batch_job = self._retry_rate_limit_sync(
            lambda: self.client.batches.get(name=batch_id),
            label=f"Google batches.get ({batch_id})",
        )
        if self._job_state(batch_job) not in _TERMINAL_STATES:
            return None
        dest = getattr(batch_job, "dest", None)
        inlined = (getattr(dest, "inlined_responses", None) or []) if dest else []
        item_ids = [str(i) for i in range(len(inlined))]
        record = batch_id not in self._usage_recorded_batches
        results = self._collect_results(
            batch_job, item_ids, is_test, record_usage=record)
        state = self._job_state(batch_job)
        if record and state != "JOB_STATE_SUCCEEDED":
            # Requests that never got a response line: the count comes from
            # the job object; when it reports none, one row for the job.
            expected = self._request_count(batch_job)
            unanswered = (max(0, expected - len(inlined)) if expected is not None
                          else (0 if inlined else 1))
            for _ in range(unanswered):
                self._record_failure(
                    self._state_class(state),
                    status=STATUS_TIMEOUT if state == "JOB_STATE_EXPIRED" else None,
                    is_test=is_test)
        return results
