"""
Local language model service using HuggingFace Transformers.

This service runs models locally on your hardware and automatically detects:
- MPS (Apple Silicon M1/M2/M3/M4)
- CUDA (NVIDIA GPUs)
- CPU (fallback)
"""
import os
from typing import List, Tuple, Optional, Any

from ..base_llm_service import BaseLLMService, make_mechanism_error
from ..llm_model import LLMModel
from ..constants import DEFAULT_LOCAL_BATCH_SIZE
from .._logging import get_logger

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = get_logger(__name__)


class LocalLMService(BaseLLMService):
    """
    Service for local models using HuggingFace Transformers.

    Automatically uses the best available device:
    - MPS on M4 Mac
    - CUDA on Linux cluster with NVIDIA GPUs
    - CPU as fallback
    """

    def __init__(self, model: LLMModel, **kwargs):
        """
        Initialize local LM service.

        Args:
            model: The LLM model to use
            **kwargs: Additional parameters
                - temperature (float): Sampling temperature
                - max_tokens (int): Maximum tokens to generate
                - device (str): Force specific device ('cuda', 'mps', 'cpu')
                               If not specified, auto-detects best available
        """
        super().__init__(
            max_concurrency=kwargs.pop("max_concurrency", 20),
            max_retries=kwargs.pop("max_retries", 5),
            batch_poll_interval=kwargs.pop("batch_poll_interval", 30),
            batch_timeout=kwargs.pop("batch_timeout", 3600),
        )
        self.model = model
        self.temperature = kwargs.get('temperature', 0.0)
        self.max_tokens = kwargs.get('max_tokens', 4096)
        self.top_p = kwargs.get('top_p', 1.0)

        # Import required libraries
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            self.torch = torch
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
            self.pipeline_fn = pipeline
        except ImportError:
            raise ImportError(
                "Required packages not installed. Install with: "
                "pip install torch transformers"
            )

        # Detect or set device
        self.device = kwargs.get('device') or self._detect_device()
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        logger.info(f"Loading model: {model.model_id}")
        self._load_model()
        logger.info("Model loaded successfully")
        self._warned_images = False

    def _detect_device(self) -> str:
        """Detect best available device."""
        if self.torch.cuda.is_available():
            return "cuda"
        elif hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_model(self):
        """Load model and tokenizer."""
        model_id = self.model.model_id

        # HuggingFace token for gated models (e.g. meta-llama). Read at load
        # time — not import time — so a token exported after `import llm_utils`
        # still works; HF_TOKEN is the library's own conventional name.
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

        # Load tokenizer
        self.tokenizer = self.AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Set padding side to LEFT for decoder-only models (like Llama)
        # Right-padding can cause generation to hang or produce incorrect results
        self.tokenizer.padding_side = 'left'
        logger.info("Set tokenizer padding_side to 'left' for decoder-only model")

        # Load model with appropriate settings
        if self.device == "cuda":
            # Use float16 for GPU
            self.model_instance = self.AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.torch.float16,
                device_map="auto",
                token=hf_token
            )
        elif self.device == "mps":
            # MPS works best with float16
            self.model_instance = self.AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.torch.float16,
                token=hf_token
            )
            self.model_instance = self.model_instance.to(self.device)
        else:
            # CPU - use full precision
            self.model_instance = self.AutoModelForCausalLM.from_pretrained(
                model_id,
                token=hf_token
            )
            self.model_instance = self.model_instance.to(self.device)

        # Create pipeline for easier generation
        # Note: Override model's default generation_config to avoid warnings
        self.model_instance.generation_config.temperature = None
        self.model_instance.generation_config.top_p = None
        self.model_instance.generation_config.do_sample = None

        # When the model was loaded with device_map="auto" (the CUDA path),
        # accelerate owns device placement and the pipeline must NOT get a
        # `device` argument — transformers raises ValueError on that combo.
        pipeline_kwargs = {}
        if getattr(self.model_instance, "hf_device_map", None) is None:
            pipeline_kwargs["device"] = self.device
        self.pipeline = self.pipeline_fn(
            "text-generation",
            model=self.model_instance,
            tokenizer=self.tokenizer,
            **pipeline_kwargs,
        )

    def _render_prompt(
        self,
        conv_data: Tuple[str, List[Tuple[str, Any]]],
        system_message: Optional[str],
    ) -> Tuple[str, str]:
        """Render one conversation to the model's prompt string.

        Instruct/chat models get their own chat template
        (``tokenizer.apply_chat_template``) so role markers match training;
        models without a template fall back to newline-joining. Images are
        not supported on the local text pipeline and are skipped with one
        warning per service instance.
        """
        conv_id, messages = conv_data
        texts = []
        for text, image in messages:
            if image is not None and not self._warned_images:
                self._warned_images = True
                logger.warning(
                    "LocalLMService is text-only — images in conversations "
                    "are ignored")
            texts.append(text)

        if getattr(self.tokenizer, "chat_template", None):
            chat = []
            if system_message:
                chat.append({"role": "system", "content": system_message})
            chat.extend({"role": "user", "content": t} for t in texts)
            try:
                prompt = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True)
                return (conv_id, prompt)
            except Exception as e:  # noqa: BLE001 — fall back to plain join
                logger.warning(
                    f"apply_chat_template failed ({str(e)[:80]}) — "
                    f"falling back to plain-text prompt")

        parts = ([system_message] if system_message else []) + texts
        return (conv_id, "\n".join(parts))

    def _generation_kwargs(self, temperature, top_p, max_tokens, batch_size=None):
        gen_kwargs = {
            'max_new_tokens': max_tokens,
            'pad_token_id': self.tokenizer.pad_token_id,
            'return_full_text': False,
        }
        if batch_size is not None:
            gen_kwargs['batch_size'] = batch_size
        if temperature and temperature > 0:
            gen_kwargs['do_sample'] = True
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = top_p
        else:
            gen_kwargs['do_sample'] = False
        return gen_kwargs

    def batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Any]]]],
        system_message: Optional[str] = None,
        is_test: bool = False,
        **kwargs
    ) -> List[Tuple[str, str]]:
        """
        Generate responses for multiple chat conversations using GPU batch inference.

        Note: the local text pipeline does not support images — any image in a
        conversation is skipped (one warning per service instance).

        Args:
            conversations: List of (id, messages) tuples, where messages is
                a list of (prompt, image) tuples.
            **kwargs: Additional parameters (temperature, max_tokens, top_p, batch_size)

        Returns:
            List of (id, response) tuples. On errors, returns (id, mechanism-error string).
        """
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        top_p = kwargs.get('top_p', self.top_p)
        batch_size = kwargs.get('batch_size', DEFAULT_LOCAL_BATCH_SIZE)

        # MPS workaround: force batch_size=1 to avoid hanging
        if self.device == "mps" and batch_size > 1:
            logger.info("MPS device: reducing batch_size to 1 for stability")
            batch_size = 1

        prepared = [self._render_prompt(c, system_message) for c in conversations]
        conv_ids = [cid for cid, _ in prepared]
        formatted_conversations = [conv for _, conv in prepared]
        total_conversations = len(formatted_conversations)

        # Results collected per id as they complete — the sequential fallback
        # below only re-runs conversations NOT already completed, so nothing
        # is generated or usage-recorded twice.
        done: dict = {}

        pbar = None
        try:
            logger.info(
                f"Running batch inference on {total_conversations} conversations "
                f"(device: {self.device}, batch_size: {batch_size})")

            if TQDM_AVAILABLE and total_conversations:
                pbar = tqdm(
                    total=total_conversations,
                    desc=f"GPU Batch Inference ({self.device})",
                    unit="conv",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )

            for i in range(0, total_conversations, batch_size):
                batch_end = min(i + batch_size, total_conversations)
                batch_ids = conv_ids[i:batch_end]
                batch_conversations = formatted_conversations[i:batch_end]

                gen_kwargs = self._generation_kwargs(
                    temperature, top_p, max_tokens, batch_size)
                batch_outputs = self.pipeline(batch_conversations, **gen_kwargs)

                if len(batch_outputs) != len(batch_ids):
                    raise RuntimeError(
                        f"pipeline returned {len(batch_outputs)} outputs for "
                        f"{len(batch_ids)} prompts")

                for conv_id, output, conv_text in zip(
                        batch_ids, batch_outputs, batch_conversations):
                    response_text = output[0]['generated_text']
                    # Track usage via tokenizer (cost=0 for local)
                    in_tok = len(self.tokenizer.encode(conv_text))
                    out_tok = len(self.tokenizer.encode(response_text))
                    self._record_usage(in_tok, out_tok, 0.0, is_test)
                    done[conv_id] = response_text

                if pbar:
                    pbar.update(len(batch_ids))

            if pbar:
                pbar.close()
            return [(cid, done[cid]) for cid in conv_ids]

        except Exception as e:
            if pbar:
                pbar.close()

            # Fallback: sequential processing of the conversations that did
            # NOT complete in the batch pass (completed ones keep their
            # results — re-running them would double compute AND double the
            # recorded usage).
            remaining = [(cid, conv) for cid, conv in prepared if cid not in done]
            logger.warning(
                f"Batch inference failed: {e}. Falling back to sequential "
                f"processing of {len(remaining)} remaining conversations.")

            iterator = remaining
            if TQDM_AVAILABLE and remaining:
                iterator = tqdm(
                    remaining,
                    desc="Sequential Inference (fallback)",
                    unit="conv"
                )

            for conv_id, formatted_conv in iterator:
                try:
                    gen_kwargs = self._generation_kwargs(
                        temperature, top_p, max_tokens)
                    output = self.pipeline(formatted_conv, **gen_kwargs)
                    response_text = output[0]['generated_text']
                    # Track usage via tokenizer (cost=0 for local)
                    in_tok = len(self.tokenizer.encode(formatted_conv))
                    out_tok = len(self.tokenizer.encode(response_text))
                    self._record_usage(in_tok, out_tok, 0.0, is_test)
                    done[conv_id] = response_text
                except Exception as e2:
                    logger.error(
                        f"Generation error for conversation {conv_id}: {str(e2)}")
                    done[conv_id] = make_mechanism_error(str(e2))

            return [(cid, done[cid]) for cid in conv_ids]
