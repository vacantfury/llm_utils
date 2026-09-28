"""Route availability: which serving routes reach a logical model, and which of
them this environment can use.

A LOGICAL model is the model itself, independent of who serves it: the
registry's ``weights`` label where one is set (``"glm-5"`` is served by Z.AI,
OpenRouter and Bedrock), else the row's ``model_id``. A ROUTE is one way to
reach it, named by the provider value (``"zai"``, ``"openrouter"``,
``"bedrock"``, ``"slurm_cluster"``, ...), one registry row per route.

A single-route model's logical name is its ``model_id``; registering a second
route later gives both rows a shared ``weights`` label, which becomes the
logical name. Lookups accept any route's ``model_id`` as well, so a stored
name keeps resolving across that change; compare stored names through
``route_models``, not by string equality.

Two questions, answered from this package's own facts only:

- ``routes_for(logical)``: every registered route (pure registry lookup);
- ``route_status(model)`` / ``usable_routes(logical)``: whether THIS process
  could build and call the route now. The default check reads configuration
  only (key variables, the Claude API opt-in, the broker switch, installed
  extras, a registered SLURM endpoint manager); ``probe=True`` adds one
  unbilled request per API route (a model listing or an identity call).

Transport facts only. Which route a payload SHOULD take, and which machine
holds which credentials, are consumer and device-layer decisions made on top
of these answers; this module knows nothing about clusters or devices.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import httpx2

from . import config
from .claude_api_policy import ALLOW_ENV as CLAUDE_API_ALLOW_ENV, claude_api_allowed
from .constants import OPENAI_API_URL, OPENROUTER_API_URL
from .llm_model import LLMModel, Provider

BROKER_URL_ENV = "LLM_UTILS_BROKER_URL"   # same switch as llm_utils.broker
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
GOOGLE_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# Probe URLs where the model listing would not test the key: OpenRouter's
# listing is public, its per-key endpoint needs the key.
_PROBE_URLS = {Provider.OPENROUTER: f"{OPENROUTER_API_URL}/key"}
# Status codes meaning "this credential will not work", per provider, beyond
# 401/403/402. Google answers an invalid key with 400 API_KEY_INVALID.
_REJECT_CODES = {Provider.GOOGLE: (400,)}
# Environment variables that configure AWS credentials without resolving
# them (resolution may call instance metadata or run a credential_process).
_AWS_CONFIG_ENVS = ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_WEB_IDENTITY_TOKEN_FILE",
                    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                    "AWS_CONTAINER_CREDENTIALS_FULL_URI")

# Providers whose credentials a broker can stand in for (llm_utils.broker).
_BROKERED = frozenset({
    Provider.OPENAI, Provider.DEEPSEEK, Provider.ZAI, Provider.XAI,
    Provider.MOONSHOT, Provider.OPENROUTER, Provider.ANTHROPIC,
    Provider.GOOGLE, Provider.BEDROCK,
})

ModelRef = Union[str, LLMModel]


@dataclass(frozen=True)
class RouteStatus:
    """Whether one route (registry row) is usable in this process.

    ``reason`` says why in plain words, for both outcomes. ``probed`` is True
    when a network probe decided the answer; False means configuration only.
    """
    model: LLMModel
    usable: bool
    reason: str
    probed: bool = False

    @property
    def route(self) -> str:
        return self.model.provider.value

    @property
    def jurisdiction(self) -> str:
        return self.model.jurisdiction


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------

def logical_models() -> Dict[str, Tuple[LLMModel, ...]]:
    """Every logical model name mapped to its registry rows (routes), in
    registry order."""
    out: Dict[str, list] = {}
    for m in LLMModel:
        out.setdefault(m.logical_name, []).append(m)
    return {name: tuple(rows) for name, rows in out.items()}


def route_models(logical: ModelRef) -> Tuple[LLMModel, ...]:
    """Every registry row serving this logical model.

    Accepts a logical name (``"glm-5"``), any route's ``model_id``, an enum
    member name (``"OR_GLM_5"``) or an ``LLMModel``. Raises ``ValueError`` for
    an unknown name: an empty answer would read as "needs no route".
    """
    if isinstance(logical, LLMModel):
        name = logical.logical_name
    else:
        groups = logical_models()
        if logical in groups:
            return groups[logical]
        # A model_id registered on several routes names one logical model
        # when those rows share it (LLMModel.from_string refuses to pick one).
        by_id = {m.logical_name for m in LLMModel if m.model_id == logical}
        if len(by_id) > 1:
            raise ValueError(f"model id {logical!r} names several logical models: {sorted(by_id)}")
        name = by_id.pop() if by_id else LLMModel.from_string(logical).logical_name
    return tuple(m for m in LLMModel if m.logical_name == name)


def routes_for(logical: ModelRef) -> frozenset:
    """Route names (provider values) registered for this logical model."""
    return frozenset(m.provider.value for m in route_models(logical))


# ---------------------------------------------------------------------------
# Usability in this environment
# ---------------------------------------------------------------------------

def _service_class(provider: Provider):
    from .llm_service_factory import LLMServiceFactory
    return LLMServiceFactory._PROVIDER_REGISTRY.get(provider)


def _key_env(provider: Provider) -> Optional[str]:
    if provider is Provider.ANTHROPIC:
        return "ANTHROPIC_API_KEY"
    if provider is Provider.GOOGLE:
        return "GOOGLE_API_KEY"
    return getattr(_service_class(provider), "API_KEY_ENV", None)


def _broker_mode() -> bool:
    return os.getenv(BROKER_URL_ENV) is not None


def _broker_status() -> Tuple[bool, str]:
    from .broker import _base_url
    from .exceptions import BrokerError
    try:
        _base_url(os.getenv(BROKER_URL_ENV, ""))
    except BrokerError:
        return False, f"broker mode, but {BROKER_URL_ENV} is not a valid loopback URL"
    return True, "broker mode: the broker holds the credential and decides the grant per call"


def _bedrock_credentials() -> Tuple[bool, str]:
    """Configuration only: never resolves the credential chain (that may call
    instance metadata or run a credential_process); probe=True resolves it."""
    if importlib.util.find_spec("boto3") is None:
        return False, "boto3 not installed (install the llm_utils[bedrock] extra)"
    for env in _AWS_CONFIG_ENVS:
        if os.getenv(env):
            return True, f"AWS credentials configured ({env})"
    for env, default in (("AWS_SHARED_CREDENTIALS_FILE", "~/.aws/credentials"),
                         ("AWS_CONFIG_FILE", "~/.aws/config")):
        path = os.path.expanduser(os.getenv(env) or default)
        if os.path.exists(path):
            return True, f"AWS credentials configured ({env if os.getenv(env) else default})"
    return False, ("no AWS credential configuration (env, profile or files); an instance "
                   "role is only detected by probe=True")


def _config_status(model: LLMModel) -> Tuple[bool, str]:
    p = model.provider
    if p is Provider.ANTHROPIC and not claude_api_allowed():
        return False, f"Anthropic API calls are off in this process ({CLAUDE_API_ALLOW_ENV} is not 1)"
    if p in _BROKERED and _broker_mode():
        return _broker_status()
    if p is Provider.BEDROCK:
        return _bedrock_credentials()
    if p is Provider.LOCAL:
        missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
        if missing:
            return False, f"{', '.join(missing)} not installed (install the llm_utils[local] extra)"
        return True, "local inference libraries installed"
    if p is Provider.SLURM_CLUSTER:
        from .llm_service_factory import LLMServiceFactory
        if LLMServiceFactory._server_manager is None:
            return False, "no endpoint manager registered (LLMServiceFactory.set_server_manager)"
        return True, "endpoint manager registered"
    env = _key_env(p)
    if env is None:
        return False, f"no availability rule for provider {p.value!r}"
    if not os.getenv(env):
        return False, f"{env} not set"
    return True, f"{env} set"


_QUOTA_MARKERS = ("insufficient_quota", "insufficient balance", "credit balance",
                  "quota exceeded", "billing")


def _http_probe(url: str, headers: dict, timeout: float,
                reject: Tuple[int, ...] = ()) -> Tuple[bool, str]:
    try:
        resp = httpx2.get(url, headers=headers, timeout=timeout)
    except httpx2.TimeoutException:
        return False, f"unreachable: no answer within {timeout}s"
    except Exception as e:  # noqa: BLE001 — any failure to ask means "not usable"
        return False, f"unreachable: {type(e).__name__}"
    code = resp.status_code
    if 200 <= code < 300:
        return True, f"reachable, credential accepted (HTTP {code})"
    if code in (401, 403) or code in reject:
        return False, f"credential rejected (HTTP {code})"
    if code == 402:
        return False, "account cannot pay (HTTP 402)"
    if code == 429:
        body = (resp.text or "").lower()
        if any(m in body for m in _QUOTA_MARKERS):
            return False, "account out of quota or credit (HTTP 429)"
        return True, "reachable but rate-limited now (HTTP 429)"
    if code >= 500:
        return False, f"endpoint error (HTTP {code})"
    return True, f"reachable (HTTP {code} on the probe URL); credential not confirmed"


def _bedrock_probe(timeout: float) -> Tuple[bool, str]:
    import boto3
    from botocore.config import Config as BotoConfig
    profile = os.getenv("AWS_PROFILE")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    try:
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        sts = session.client("sts", region_name=region, config=BotoConfig(
            connect_timeout=timeout, read_timeout=timeout, retries={"max_attempts": 1}))
        sts.get_caller_identity()
    except Exception as e:  # noqa: BLE001 — any failure means "not usable"
        return False, f"AWS identity check failed ({type(e).__name__})"
    return True, "AWS identity confirmed (model access is not checked)"


def _probe(model: LLMModel, timeout: float) -> Optional[Tuple[bool, str]]:
    """Network check for API routes; None where no probe applies."""
    p = model.provider
    if p is Provider.BEDROCK:
        return _bedrock_probe(timeout)
    if p is Provider.ANTHROPIC:
        return _http_probe(ANTHROPIC_MODELS_URL, {
            "x-api-key": os.getenv("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01"}, timeout)
    if p is Provider.GOOGLE:
        return _http_probe(GOOGLE_MODELS_URL, {
            "x-goog-api-key": os.getenv("GOOGLE_API_KEY", "")}, timeout,
            reject=_REJECT_CODES.get(p, ()))
    cls = _service_class(p)
    env = getattr(cls, "API_KEY_ENV", None)
    probe_url = _PROBE_URLS.get(p)
    if probe_url is None and env is not None and hasattr(cls, "BASE_URL"):
        probe_url = f"{cls.BASE_URL or OPENAI_API_URL}/models"
    if probe_url is None or env is None:
        return None
    return _http_probe(probe_url, {"Authorization": f"Bearer {os.getenv(env, '')}"}, timeout,
                       reject=_REJECT_CODES.get(p, ()))


def route_status(model: ModelRef, *, probe: bool = False,
                 timeout: Optional[float] = None) -> RouteStatus:
    """Whether this one route (registry row) is usable here.

    ``model`` is an ``LLMModel`` row, or a string naming exactly one row (an
    enum name or a route's ``model_id``). A string that is the logical name of
    several routes (``"glm-5"``) raises ``ValueError``: pick the row, or use
    ``route_statuses`` for all of them.

    ``probe=True`` sends one unbilled request when the configuration check
    passes (skipped in broker mode: the broker decides per grant). ``timeout``
    defaults to the ``routes.probe_timeout_s`` config value.
    """
    if not isinstance(model, LLMModel):
        group = logical_models().get(model, ())
        if len(group) > 1:
            raise ValueError(
                f"{model!r} is the logical name of {len(group)} routes "
                f"({', '.join(m.name for m in group)}); pass one LLMModel row "
                f"or use route_statuses({model!r})")
        model = LLMModel.from_string(model)
    ok, reason = _config_status(model)
    if not ok or not probe or (model.provider in _BROKERED and _broker_mode()):
        return RouteStatus(model, ok, reason)
    t = config.get("routes.probe_timeout_s") if timeout is None else timeout
    result = _probe(model, t)
    if result is None:
        return RouteStatus(model, ok, reason)
    return RouteStatus(model, result[0], result[1], probed=True)


def route_statuses(logical: ModelRef, *, probe: bool = False,
                   timeout: Optional[float] = None) -> Tuple[RouteStatus, ...]:
    """``route_status`` for every route of this logical model, in registry order."""
    return tuple(route_status(m, probe=probe, timeout=timeout) for m in route_models(logical))


def usable_routes(logical: ModelRef, *, probe: bool = False,
                  timeout: Optional[float] = None) -> Tuple[LLMModel, ...]:
    """The rows of this logical model that are usable here, in registry order."""
    return tuple(s.model for s in route_statuses(logical, probe=probe, timeout=timeout)
                 if s.usable)


class RegistryRouteResolver:
    """Object form of the lookup, for consumers that inject a resolver
    (``routes_for(logical) -> frozenset[str]``)."""

    def routes_for(self, logical_model: str) -> frozenset:
        return routes_for(logical_model)

    def usable_routes_for(self, logical_model: str, *, probe: bool = False) -> frozenset:
        return frozenset(m.provider.value for m in usable_routes(logical_model, probe=probe))
