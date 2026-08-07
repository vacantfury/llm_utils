"""
LLM utility module for working with various language models.
"""

# Version — READ FROM THE INSTALLED DISTRIBUTION, never hardcoded here.
#
# A literal string in this file is a second source of truth next to
# pyproject.toml, and it drifts silently: releases v6.0.0 and v6.1.0 both
# shipped with this line still reading "5.2.0". That is not cosmetic — a
# consumer whose job log prints `llm_utils.__version__` for provenance then
# records a version that was never installed, which is exactly the failure the
# provenance line exists to prevent (caught 2026-08-07 by a consumer's
# run-identity print reporting 5.2.0 against a verified 5.3.0 install).
from importlib.metadata import PackageNotFoundError, version as _dist_version

try:
    __version__ = _dist_version("llm_utils")
except PackageNotFoundError:      # source tree with no install (e.g. `python -c` in-repo)
    __version__ = "0.0.0+unknown"

# Core components
from .llm_model import LLMModel, Provider, ModelQuirk
from .account_status import AccountStatus, burn_rate, days_to_empty
from .base_llm_service import (
    BaseLLMService,
    UsageStats,
    is_mechanism_error,
    make_mechanism_error,
    strip_mechanism_error,
)
from .llm_service_factory import LLMServiceFactory
from .exceptions import (
    FatalModelError,
    AccountFatalError,
    InvalidCredentialError,
    CreditsExhaustedError,
)

# Concrete service implementations — ALL serving routes are part of the
# public seam (heavy deps stay lazy: importing these classes needs no
# torch/boto3; only INSTANTIATING LocalLMService/BedrockService does).
from .llm_services import (
    OpenAIService,
    DeepSeekService,
    ZAIService,
    XAIService,
    MoonshotService,
    OpenRouterService,
    ClaudeService,
    GoogleService,
    LocalLMService,
    SlurmClusterService,
    BedrockService,
)

# Define what's exported
__all__ = [
    # Models and enums
    'LLMModel',
    'Provider',
    'ModelQuirk',

    # Base and factory
    'BaseLLMService',
    'UsageStats',
    'LLMServiceFactory',

    # Account status (credit balance) + pure spend math
    'AccountStatus',
    'burn_rate',
    'days_to_empty',

    # Mechanism-error helpers (transport-failure sentinel strings)
    'is_mechanism_error',
    'make_mechanism_error',
    'strip_mechanism_error',

    # Exceptions
    'FatalModelError',
    'AccountFatalError',
    'InvalidCredentialError',
    'CreditsExhaustedError',

    # Concrete services (one per serving route)
    'OpenAIService',
    'DeepSeekService',
    'ZAIService',
    'XAIService',
    'MoonshotService',
    'OpenRouterService',
    'ClaudeService',
    'GoogleService',
    'LocalLMService',
    'SlurmClusterService',
    'BedrockService',

    # Version
    '__version__'
]
