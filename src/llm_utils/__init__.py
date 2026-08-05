"""
LLM utility module for working with various language models.
"""

# Define version information
__version__ = "5.2.0"

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
from .cluster_server_manager import ClusterModelServerManager

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

    # Cluster serving lifecycle
    'ClusterModelServerManager',

    # Version
    '__version__'
]
