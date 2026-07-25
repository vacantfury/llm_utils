"""
Local language model service using HuggingFace Transformers.

This service runs models locally on your hardware and automatically detects:
- MPS (Apple Silicon M1/M2/M3/M4)
- CUDA (NVIDIA GPUs)
- CPU (fallback)
"""
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path

from ..base_llm_service import BaseLLMService, make_mechanism_error
from ..llm_model import LLMModel
from ..constants import HUGGINGFACE_TOKEN, DEFAULT_LOCAL_BATCH_SIZE
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
    
    def __init__(self, model: LLMModel, config=None, **kwargs):
        """
        Initialize local LM service.
        
        Args:
            model: The LLM model to use
            config: Optional config with default parameters
            **kwargs: Additional parameters
                - temperature (float): Sampling temperature
                - max_tokens (int): Maximum tokens to generate
                - device (str): Force specific device ('cuda', 'mps', 'cpu')
                               If not specified, auto-detects best available
        """
        super().__init__()
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
        logger.info(f"Model loaded successfully")
    
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
        
        # Use HuggingFace token if available (for gated models like Llama)
        hf_token = HUGGINGFACE_TOKEN
        
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
        logger.info(f"Set tokenizer padding_side to 'left' for decoder-only model")
        
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
        
        self.pipeline = self.pipeline_fn(
            "text-generation",
            model=self.model_instance,
            tokenizer=self.tokenizer,
            device=self.device if self.device != "auto" else None
        )
    
    @staticmethod
    def _flatten_conversation(
        conv_data: Tuple[str, List[Tuple[str, Any]]],
        system_message: Optional[str],
    ) -> Tuple[str, str]:
        """Flatten a conversation to a single text string (local models are text-only)."""
        conv_id, messages = conv_data
        parts = []
        if system_message:
            parts.append(system_message)
        for text, _image in messages:
            parts.append(text)
        return (conv_id, "\n".join(parts))
    
    def batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Any]]]],
        system_message: Optional[str] = None,
        is_test: bool = False,
        **kwargs
    ) -> List[Tuple[str, str]]:
        """
        Generate responses for multiple chat conversations using GPU batch inference.

        Note: Most local models don't support images natively. Images are converted
        to text placeholders.

        Args:
            conversations: List of (id, messages) tuples, where messages is
                a list of (prompt, image) tuples. Image can be file path, PIL Image, or list.
            **kwargs: Additional parameters (temperature, max_tokens, top_p, batch_size)

        Returns:
            List of (id, response) tuples. On errors, returns (id, error_message_string).
        """
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        top_p = kwargs.get('top_p', self.top_p)
        batch_size = kwargs.get('batch_size', DEFAULT_LOCAL_BATCH_SIZE)

        # MPS workaround: force batch_size=1 to avoid hanging
        if self.device == "mps" and batch_size > 1:
            logger.info(f"MPS device: reducing batch_size to 1 for stability")
            batch_size = 1

        prepared = [self._flatten_conversation(c, system_message) for c in conversations]
        
        # Step 2: Native GPU batch inference with progress bar
        conv_ids = [cid for cid, _ in prepared]
        formatted_conversations = [conv for _, conv in prepared]
        
        pbar = None  # Initialize to avoid UnboundLocalError
        
        try:
            # Process in chunks to show progress
            logger.info(f"Running batch inference on {len(formatted_conversations)} conversations (device: {self.device}, batch_size: {batch_size})")
            
            results = []
            
            # Create chunks based on batch_size
            total_conversations = len(formatted_conversations)
            
            # Setup progress bar
            if TQDM_AVAILABLE:
                pbar = tqdm(
                    total=total_conversations,
                    desc=f"GPU Batch Inference ({self.device})",
                    unit="conv",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )
            else:
                logger.info(f"Processing {total_conversations} conversations on {self.device}")
                pbar = None

            # Process in batches
            for i in range(0, total_conversations, batch_size):
                batch_end = min(i + batch_size, total_conversations)
                batch_ids = conv_ids[i:batch_end]
                batch_conversations = formatted_conversations[i:batch_end]
                
                # Prepare generation config
                gen_kwargs = {
                    'max_new_tokens': max_tokens,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'return_full_text': False,
                    'batch_size': batch_size
                }
                
                if temperature > 0:
                    gen_kwargs['do_sample'] = True
                    gen_kwargs['temperature'] = temperature
                    gen_kwargs['top_p'] = top_p
                else:
                    gen_kwargs['do_sample'] = False

                # Process this batch on GPU
                batch_outputs = self.pipeline(batch_conversations, **gen_kwargs)
                
                # Collect results
                for conv_id, output, conv_text in zip(batch_ids, batch_outputs, batch_conversations):
                    response_text = output[0]['generated_text']
                    # Track usage via tokenizer (cost=0 for local)
                    in_tok = len(self.tokenizer.encode(conv_text))
                    out_tok = len(self.tokenizer.encode(response_text))
                    self._record_usage(in_tok, out_tok, 0.0, is_test)
                    results.append((conv_id, response_text))
                
                # Update progress
                if pbar:
                    pbar.update(len(batch_ids))
            
            if pbar:
                pbar.close()
            
            return results
            
        except Exception as e:
            # Close progress bar if it exists
            if pbar:
                pbar.close()
            
            # Fallback to sequential processing if batch inference fails
            logger.warning(f"Batch inference failed: {e}. Falling back to sequential processing.")
            results = []

            # Setup progress bar for fallback
            iterator = prepared
            if TQDM_AVAILABLE:
                iterator = tqdm(
                    prepared,
                    desc="Sequential Inference (fallback)",
                    unit="conv"
                )
            else:
                logger.info(f"Sequential fallback: processing {len(prepared)} conversations")

            for conv_id, formatted_conv in iterator:
                try:
                    # Prepare generation config
                    gen_kwargs = {
                        'max_new_tokens': max_tokens,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'return_full_text': False
                    }

                    if temperature > 0:
                        gen_kwargs['do_sample'] = True
                        gen_kwargs['temperature'] = temperature
                        gen_kwargs['top_p'] = top_p
                    else:
                        gen_kwargs['do_sample'] = False

                    output = self.pipeline(formatted_conv, **gen_kwargs)
                    response_text = output[0]['generated_text']
                    # Track usage via tokenizer (cost=0 for local)
                    in_tok = len(self.tokenizer.encode(formatted_conv))
                    out_tok = len(self.tokenizer.encode(response_text))
                    self._record_usage(in_tok, out_tok, 0.0, is_test)
                    results.append((conv_id, response_text))
                except Exception as e2:
                    logger.error(f"Generation error for conversation {conv_id}: {str(e2)}")
                    results.append((conv_id, make_mechanism_error(str(e2))))
            
            return results
