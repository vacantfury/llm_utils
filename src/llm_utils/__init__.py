"""
LLM utility module for working with various language models.
"""

# Define version information
__version__ = "3.0.0"

# Core components
from .llm_model import LLMModel, Provider, ModelQuirk
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

# Concrete service implementations
from .llm_services import OpenAIService, ClaudeService, GoogleService, LocalLMService

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

    # Mechanism-error helpers (transport-failure sentinel strings)
    'is_mechanism_error',
    'make_mechanism_error',
    'strip_mechanism_error',

    # Exceptions
    'FatalModelError',
    'AccountFatalError',
    'InvalidCredentialError',
    'CreditsExhaustedError',

    # Concrete services
    'OpenAIService',
    'ClaudeService',
    'GoogleService',
    'LocalLMService',

    # Version
    '__version__'
]
