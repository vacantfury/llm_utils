"""
AWS Bedrock service — boto3 ``bedrock-runtime.converse``.

Unlike the OpenAI-compatible providers (DeepSeek / Z.AI / xAI / Moonshot, which
subclass ``OpenAIService``), Bedrock is NOT OpenAI-wire-compatible, so this is a
from-scratch ``BaseLLMService`` subclass — closest in shape to
``NURCClusterService``. boto3 is sync-only, so each ``converse`` call runs in
``asyncio.to_thread`` inside the same semaphore-gathered concurrency pattern the
other services use, keeping the ``batch_chat`` seam identical.

Credentials come from the standard AWS credential chain (a named profile / env
``AWS_*`` / instance role) — NOT a bearer token. On the xc cluster the box's AWS
profile resolves automatically when the run environment sets ``AWS_PROFILE`` (see
TODO item 2). Same code on every cluster; only the per-host AWS credential
provisioning differs, exactly like the other API keys.

Config knobs (from ``conf/llm`` merge or kwargs), all optional:
  - ``aws_profile``  — named profile; else ``AWS_PROFILE`` env; else default chain.
  - ``aws_region``   — else ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` env; else us-east-1.
"""
import asyncio
import base64
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from ..base_llm_service import BaseLLMService, make_mechanism_error
from ..llm_model import LLMModel, ModelQuirk
from ..media_utils import encode_image_to_b64
from .._logging import get_logger

logger = get_logger(__name__)

# MIME → Bedrock converse image `format` enum (converse wants raw bytes + a
# format tag, not a base64 data-URI like the OpenAI-shaped `image_url`).
_BEDROCK_IMAGE_FORMATS = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

# boto3 error substrings that mean "back off and retry" vs "fail this row".
_THROTTLE_MARKERS = ("ThrottlingException", "TooManyRequests", "Throttling", "429")

# Errors that mean the AWS CREDENTIALS are bad/expired — they will NOT recover
# mid-run, so retrying every row is pointless (slow) and the right response is to
# stop the whole run FAST with an actionable message. On the xc cluster the
# `arise-beta` creds are temporary STS creds that expire every few hours and are
# NOT auto-refreshed — they must be re-minted ON the box (only the box owner can,
# via the Amazon-internal SSO/Kiro login). See cluster_files/xc_cluster_properties.md.
_CREDENTIAL_MARKERS = (
    "ExpiredToken", "ExpiredTokenException",
    "UnrecognizedClientException", "InvalidClientTokenId",
    "SignatureDoesNotMatch", "InvalidSignatureException",
    "CredentialsError", "NoCredentialsError", "TokenRefreshRequired",
)
# Errors that mean this model isn't invocable with the current creds (not enabled
# in the account / no invoke permission / wrong id) — also won't recover mid-run.
_ACCESS_MARKERS = ("AccessDeniedException", "AccessDenied")


class BedrockCredentialsError(RuntimeError):
    """Raised (fail-fast) when Bedrock creds are expired/invalid, so a run stops
    with a clear message instead of grinding every row into a cryptic error."""


class BedrockAccessError(RuntimeError):
    """Raised (fail-fast) when a Bedrock model can't be invoked with the current
    account/creds (not enabled, no permission, or a bad model id)."""


class BedrockService(BaseLLMService):
    """AWS Bedrock via bedrock-runtime.converse. Region/profile via config/env."""

    DEFAULT_REGION = "us-east-1"

    def __init__(self, model: LLMModel, config=None, **kwargs):
        super().__init__(
            # Conservative default for a beta allocation with unknown TPM/RPM;
            # throttling is retried with backoff. Raise via conf/llm if the
            # account's limits allow.
            max_concurrency=kwargs.pop("max_concurrency", 8),
            max_retries=kwargs.pop("max_retries", 5),
            batch_poll_interval=kwargs.pop("batch_poll_interval", 30),
            batch_timeout=kwargs.pop("batch_timeout", 3600),
        )
        self.model = model
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.region = (
            kwargs.get("aws_region")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or self.DEFAULT_REGION
        )
        # None → boto3 default chain (which itself honours AWS_PROFILE env).
        self.aws_profile = kwargs.get("aws_profile") or os.environ.get("AWS_PROFILE")

        # Lazy import so the whole llm_utils import tree still loads in envs
        # without boto3 installed (only instantiating a Bedrock service needs it).
        try:
            import boto3
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "boto3 is required for Provider.BEDROCK — `uv add boto3` (or install "
                "into this cluster's env). See TODO item 2."
            ) from e

        session = (
            boto3.Session(profile_name=self.aws_profile)
            if self.aws_profile
            else boto3.Session()
        )
        self._client = session.client("bedrock-runtime", region_name=self.region)
        logger.info(
            "Initialized Bedrock service for %s (region=%s, profile=%s)",
            model.model_id, self.region, self.aws_profile or "default-chain",
        )

    # ------------------------------------------------------------------
    # Message formatting — Bedrock `converse` shape
    # ------------------------------------------------------------------

    @staticmethod
    def _format_messages(
        messages: List[Tuple[str, Optional[Any]]],
    ) -> List[Dict[str, Any]]:
        bedrock_msgs: List[Dict[str, Any]] = []
        for text, image in messages:
            content: List[Dict[str, Any]] = [{"text": text}]
            if image is not None:
                images = image if isinstance(image, list) else [image]
                for img in images:
                    if img is None:
                        continue
                    b64, mime = encode_image_to_b64(img)
                    raw = base64.b64decode(b64)
                    fmt = _BEDROCK_IMAGE_FORMATS.get((mime or "").lower(), "png")
                    content.append(
                        {"image": {"format": fmt, "source": {"bytes": raw}}}
                    )
            bedrock_msgs.append({"role": "user", "content": content})
        return bedrock_msgs

    def _inference_config(self, temperature: float, max_tokens: int) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"maxTokens": max_tokens}
        if not self.model.has_quirk(ModelQuirk.NO_CUSTOM_TEMPERATURE):
            cfg["temperature"] = temperature
        return cfg

    def _converse(self, bedrock_msgs, system_message, temperature, max_tokens):
        """Sync boto3 converse call (run via asyncio.to_thread)."""
        params: Dict[str, Any] = {
            "modelId": self.model.model_id,
            "messages": bedrock_msgs,
            "inferenceConfig": self._inference_config(temperature, max_tokens),
        }
        if system_message:
            params["system"] = [{"text": system_message}]
        return self._client.converse(**params)

    @staticmethod
    def _extract_response(response: Dict[str, Any]) -> str:
        try:
            blocks = response["output"]["message"]["content"]
        except (KeyError, TypeError) as e:
            return make_mechanism_error(f"bedrock response parse error: {e}")
        text = "\n".join(b["text"] for b in blocks if isinstance(b, dict) and "text" in b)
        if not text.strip():
            stop = response.get("stopReason", "unknown")
            return f"[LLM response filtered out due to: {stop}]"
        return text

    # ------------------------------------------------------------------
    # Async execution
    # ------------------------------------------------------------------

    async def _one_call(
        self,
        sem: asyncio.Semaphore,
        bedrock_msgs: List[Dict],
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        is_test: bool,
    ) -> str:
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await asyncio.to_thread(
                        self._converse,
                        bedrock_msgs, system_message, temperature, max_tokens,
                    )
                    usage = response.get("usage") or {}
                    in_tok = usage.get("inputTokens", 0) or 0
                    out_tok = usage.get("outputTokens", 0) or 0
                    cost = (
                        in_tok * self.model.input_price
                        + out_tok * self.model.output_price
                    ) / 1_000_000
                    self._record_usage(in_tok, out_tok, cost, is_test)
                    return self._extract_response(response)

                except Exception as e:
                    err = str(e)
                    # Fail FAST on an exhausted credit balance / billing quota —
                    # account-global, never recovers mid-run (Bedrock creds are
                    # handled by the AWS-specific markers below).
                    self._raise_if_account_fatal(e)
                    # Fail FAST on dead creds / no-access — these never recover
                    # mid-run, so raise once (aborts the whole batch) instead of
                    # retrying every row into a cryptic MECHANISM_ERROR.
                    if any(m in err for m in _CREDENTIAL_MARKERS):
                        raise BedrockCredentialsError(
                            f"AWS Bedrock credentials are expired/invalid "
                            f"({self.model.model_id}): {err}\n"
                            f"On the xc box these temp STS creds expire every few "
                            f"hours and are NOT auto-refreshed — re-mint them ON "
                            f"the box (box owner's Amazon SSO/Kiro login), then "
                            f"re-run. See cluster_files/xc_cluster_properties.md."
                        ) from e
                    if any(m in err for m in _ACCESS_MARKERS):
                        raise BedrockAccessError(
                            f"AWS Bedrock model not invocable "
                            f"({self.model.model_id}): {err}\n"
                            f"Check the model is enabled in the account and the id "
                            f"is exactly the invocable id (Claude = us.*-prefixed "
                            f"inference profile; qwen/deepseek/nova = bare on-demand id)."
                        ) from e
                    is_throttle = any(m in err for m in _THROTTLE_MARKERS)
                    if is_throttle and attempt < self.max_retries:
                        wait = (2 ** attempt) + random.random()
                        logger.warning(
                            "Bedrock throttled, retry %d/%d in %.1fs",
                            attempt + 1, self.max_retries, wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    logger.error("Bedrock converse error (%s): %s", self.model.model_id, err)
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
        # Clamp to the model's provider-enforced output ceiling. Bedrock 400s if
        # maxTokens exceeds a model's hard limit (Amazon Nova = 10000 vs the
        # project default 16384); the registry carries the per-model cap.
        cap = getattr(self.model, "max_output_tokens", None)
        if cap is not None and max_tokens > cap:
            logger.debug(
                "Clamping maxTokens %d → %d (model cap for %s)",
                max_tokens, cap, self.model.model_id,
            )
            max_tokens = cap

        prepared = [
            (cid, self._format_messages(msgs)) for cid, msgs in conversations
        ]
        logger.info(
            "Sending %d Bedrock requests async (concurrency=%d)",
            len(prepared), self.max_concurrency,
        )

        async def _run() -> List[str]:
            sem = asyncio.Semaphore(self.max_concurrency)
            tasks = [
                self._one_call(sem, msgs, system_message, temperature, max_tokens, is_test)
                for _, msgs in prepared
            ]
            return await asyncio.gather(*tasks)

        responses = asyncio.run(_run())
        return [(cid, resp) for (cid, _), resp in zip(prepared, responses)]
