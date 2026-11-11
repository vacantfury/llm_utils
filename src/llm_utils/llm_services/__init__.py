"""
LLM service implementations for various providers.

Every service's ``batch_chat`` takes ``[(conv_id, messages), ...]`` where each
message is a ``(text, image_or_None)`` tuple (each provider places the images as
it always has) or an ``OrderedParts`` (text and image parts rendered in exactly
the order given, on every provider; see ``ordered_parts.py``).
"""

from .openai_service import OpenAIService
from .openai_compatible_services import (
    DeepSeekService, ZAIService, XAIService, MoonshotService, OpenRouterService,
)
from .claude_service import ClaudeService
from .google_service import GoogleService
from .local_lm_service import LocalLMService
from .slurm_cluster_service import SlurmClusterService
from .bedrock_service import BedrockService
from .ordered_parts import OrderedParts

__all__ = [
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
    'OrderedParts',
]
