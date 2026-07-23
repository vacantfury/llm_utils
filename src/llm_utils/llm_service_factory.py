"""
Factory for creating LLM service instances.

This is the single bridge between YAML configuration and service constructors.
Services themselves are dumb executors — they take params from kwargs only.
"""
from typing import Dict, Optional, Type, Union

from .llm_model import LLMModel, Provider
from .base_llm_service import BaseLLMService

# Import concrete service implementations
from .llm_services import (
    OpenAIService, DeepSeekService, ZAIService, XAIService, MoonshotService,
    ClaudeService, GoogleService, LocalLMService, NURCClusterService,
    BedrockService,
)


class LLMServiceFactory:
    """Factory for creating LLM service instances based on model provider.
    
    Acts as the single bridge between YAML configuration and services.
    Loads conf/llm/default.yaml + model-specific overrides, merges with
    caller-provided kwargs, and passes the result to the service constructor.
    """
    
    # Registry mapping providers to their service implementations
    _PROVIDER_REGISTRY: Dict[Provider, Type[BaseLLMService]] = {
        Provider.OPENAI: OpenAIService,
        Provider.DEEPSEEK: DeepSeekService,
        Provider.ZAI: ZAIService,
        Provider.XAI: XAIService,
        Provider.MOONSHOT: MoonshotService,
        Provider.BEDROCK: BedrockService,
        Provider.ANTHROPIC: ClaudeService,
        Provider.GOOGLE: GoogleService,
        Provider.LOCAL: LocalLMService,
        Provider.NU_CLUSTER: NURCClusterService,
    }
    
    # Cluster server manager (set by Experiment before running tasks)
    _server_manager: Optional[object] = None
    
    
    @classmethod
    def set_server_manager(cls, manager) -> None:
        """
        Register the ClusterModelServerManager.

        Called by Experiment.run_experiment() after starting servers,
        so factory can auto-fetch endpoint URLs for cluster models.

        Args:
            manager: ClusterModelServerManager instance with running servers.
        """
        cls._server_manager = manager

    @classmethod
    def clear_server_manager(cls) -> None:
        """Clear the registered manager (call on Experiment teardown).

        The manager lives as class-level singleton state; without an explicit
        clear, a subsequent Experiment in the same process inherits the
        previous run's (now shut-down) manager and tasks calling cluster
        models will fail with stale-reference errors instead of a clean
        ``RuntimeError: no manager registered``.
        """
        cls._server_manager = None
    
    @classmethod
    def get_registered_providers(cls) -> list[Provider]:
        """
        Get list of currently registered providers.
        
        Returns:
            List of provider enums that have registered services
        """
        return list(cls._PROVIDER_REGISTRY.keys())
    
    @classmethod
    def is_provider_supported(cls, provider: Provider) -> bool:
        """
        Check if a provider is currently supported.
        
        Args:
            provider: The provider to check
        
        Returns:
            True if provider has a registered service, False otherwise
        """
        return provider in cls._PROVIDER_REGISTRY

    @classmethod
    def register_provider(cls, provider: Provider, service_class: Type[BaseLLMService]) -> None:
        """
        Register a service class for a provider.
        
        This allows dynamic registration of new providers without modifying the factory.
        
        Args:
            provider: The provider enum
            service_class: The service class to handle this provider
        """
        cls._PROVIDER_REGISTRY[provider] = service_class

    @classmethod
    def _load_model_defaults(cls, model: LLMModel) -> dict:
        """Load model params from YAML: default.yaml → model-specific override."""
        from src.experiment.config import load_conf
        params = load_conf(
            "llm", section="model",
            match_field="model.model", match_value=model.model_id)
        # 'model' is already passed as a positional arg to service constructors,
        # so remove it from kwargs to avoid "got multiple values for argument"
        params.pop("model", None)
        return params

    @classmethod
    def create(cls, model: Union[str, LLMModel], **kwargs) -> BaseLLMService:
        """
        Create an LLM service instance for the given model.
        
        Loads YAML defaults for the model and merges with caller kwargs
        (caller kwargs take priority). For cluster models, automatically
        injects the server manager.
        
        Args:
            model: The LLM model (LLMModel enum or string model ID)
            **kwargs: Additional arguments passed to the service constructor.
                These override YAML defaults.
                Common kwargs:
                - temperature (float): Sampling temperature
                - max_tokens (int): Maximum tokens to generate
                - api_key (str): API key for API-based services
        
        Returns:
            Instance of the appropriate service implementation
        
        Raises:
            ValueError: If no service is registered for the model's provider
        """
        # Single conversion point: str → LLMModel
        if isinstance(model, str):
            model = LLMModel.from_string(model)
        
        service_class = cls._PROVIDER_REGISTRY.get(model.provider)
        
        if service_class is None:
            raise ValueError(
                f"No service registered for provider: {model.provider}. "
                f"Available providers: {list(cls._PROVIDER_REGISTRY.keys())}"
            )
        
        # Load YAML defaults, then let caller kwargs override
        yaml_defaults = cls._load_model_defaults(model)
        merged_kwargs = {**yaml_defaults, **kwargs}
        
        # For cluster models: inject server_manager from manager if not explicitly provided
        if model.provider == Provider.NU_CLUSTER and "server_manager" not in merged_kwargs:
            if cls._server_manager is None:
                raise RuntimeError(
                    f"Cannot create service for cluster model {model.model_id}: "
                    f"No ClusterModelServerManager registered. "
                    f"Call LLMServiceFactory.set_server_manager() first."
                )
            merged_kwargs["server_manager"] = cls._server_manager
        
        return service_class(model, **merged_kwargs)