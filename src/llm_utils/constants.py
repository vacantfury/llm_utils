"""
Constants for LLM configurations and endpoints.
"""
import os
from typing import Final, Optional, TYPE_CHECKING

from dotenv import load_dotenv

# Import LLMModel for type annotations
if TYPE_CHECKING:
    from .llm_model import LLMModel

# API keys are read as plain environment variables. They can be exported in the
# shell, injected by a secret manager, or placed in a gitignored `.env` in the
# CONSUMER's working tree — load_dotenv() searches upward from the CWD, which is
# what an installed package must do (a path relative to this file would point
# into site-packages). override=False: the real environment always wins.
load_dotenv(override=False)


# API Keys (loaded from environment variables)
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
# OpenAI-compatible third-party providers. DeepSeek + Z.AI + Moonshot are
# DIRECT MAINLAND endpoints — consumers must route ZERO personal data through
# them (bulk judge/eval work over public data only); see the Provider registry
# note in llm_model.py. xAI is US jurisdiction; OpenRouter is a US aggregator
# over hosted open weights.
DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
ZAI_API_KEY: Optional[str] = os.getenv("ZAI_API_KEY")
OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
XAI_API_KEY: Optional[str] = os.getenv("XAI_API_KEY")
MOONSHOT_API_KEY: Optional[str] = os.getenv("MOONSHOT_API_KEY")

# API endpoints
OPENAI_API_URL: Final[str] = "https://api.openai.com/v1"
DEEPSEEK_API_URL: Final[str] = "https://api.deepseek.com"          # direct mainland
ZAI_API_URL: Final[str] = "https://api.z.ai/api/paas/v4"           # direct mainland
OPENROUTER_API_URL: Final[str] = "https://openrouter.ai/api/v1"    # US aggregator
XAI_API_URL: Final[str] = "https://api.x.ai/v1"                    # US jurisdiction
MOONSHOT_API_URL: Final[str] = "https://api.moonshot.ai/v1"        # direct mainland (.ai = international)
OLLAMA_BASE_URL: Final[str] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VERSION_URL: Final[str] = f"{OLLAMA_BASE_URL}/api/version"
OLLAMA_CHAT_URL: Final[str] = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_GENERATE_URL: Final[str] = f"{OLLAMA_BASE_URL}/api/generate"

DEFAULT_SYSTEM_MESSAGE: Final[str] = "You are a helpful assistant."


# =============================================================================
# Per-model facts have moved to llm_model.py
# =============================================================================
# What used to be two scattered sets here:
#
#   MODELS_USING_MAX_COMPLETION_TOKENS  → ModelQuirk.USES_MAX_COMPLETION_TOKENS
#   MODELS_WITHOUT_TEMPERATURE_SUPPORT  → ModelQuirk.NO_CUSTOM_TEMPERATURE
#
# Both are now per-model `quirks` on each `ModelSpec`. Check via
# `model.has_quirk(ModelQuirk.X)`. See llm_model.py for the registry.
#
# Why moved: every static fact about a model now lives in one row of the
# enum, so adding a new fact = touching ModelSpec + selected rows, not
# scattering more sets across this file.


# =============================================================================
# Batch processing settings
# =============================================================================

# API batch processing (OpenAI, Claude)
# Note: Currently for documentation purposes. Use when implementing async batch API calls.
DEFAULT_API_BATCH_SIZE: Final[int] = 300  # Max API requests in a batch

# Local model GPU batch inference (HuggingFace Transformers)
# Adjust based on your model size and GPU:
# - Large models (7B-8B) on MPS: 1-2 (can hang, use sequential)
# - Small models (1B-3B) on MPS: 4-8
# - CUDA GPUs (RTX 3090, 4090): 8-16
# - High-end GPUs (A100, H100): 32-64
# Note: 1B models work well with batch_size=4, larger models may need batch_size=1
DEFAULT_LOCAL_BATCH_SIZE: Final[int] = 4  # Good for 1B models on MPS


# =============================================================================
# Vision-Language Model Support
# =============================================================================
# vLLM's llm.chat() handles images automatically via structured messages.
# No model-specific placeholders needed - just pass PIL.Image in the content:
#   {"type": "image_url", "image_url": {"url": pil_image}}


# =============================================================================
# NURC Cluster SLURM Limits
# =============================================================================
# Hard limits imposed by the NURC gpu partition QOS.
# See text_docs/shared/nurc_cluster_properties.md for details.
MAX_SLURM_TIME_LIMIT: Final[str] = "08:00:00"  # max wall time for gpu partition
