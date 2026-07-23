"""
LLM utility module for working with various language models.
"""

# Define version information
__version__ = "2.3.0"

# Core components
from .llm_model import LLMModel, Provider, ModelQuirk
from .base_llm_service import BaseLLMService, UsageStats
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
