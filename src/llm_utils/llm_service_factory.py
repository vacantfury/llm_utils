"""
Factory for creating LLM service instances.

This is the single bridge between configuration and service constructors.
Services themselves are dumb executors — they take params from kwargs only.
Where per-model defaults come from is the CONSUMER's choice: register a config
loader via ``set_config_loader`` (e.g. one backed by the host repo's YAML
config); without one, services are constructed from caller kwargs alone.
"""
from typing import Callable, Dict, Optional, Type, Union

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

    Acts as the single bridge between configuration and services. If a config
    loader is registered (``set_config_loader``), its per-model defaults are
    merged with caller-provided kwargs (kwargs win) and passed to the service
    constructor; with no loader, kwargs alone configure the service.
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

    # Per-model default-params loader (set by the consumer at startup).
    # Signature: loader(model: LLMModel) -> dict of service-constructor kwargs.
    _config_loader: Optional[Callable[[LLMModel], dict]] = None


    @classmethod
    def set_config_loader(cls, loader: Callable[[LLMModel], dict]) -> None:
        """
        Register a loader that supplies per-model default constructor params.

        The package itself has no config-file knowledge; a consumer that keeps
        model params in config (e.g. YAML) wires it in here once at startup:

            LLMServiceFactory.set_config_loader(
                lambda model: my_load_conf(model.model_id))

        Args:
            loader: Callable taking the LLMModel, returning a dict of kwargs
                for the service constructor (may be empty).
        """
        cls._config_loader = loader

    @classmethod
    def clear_config_loader(cls) -> None:
        """Unregister the config loader (services then use caller kwargs only)."""
        cls._config_loader = None


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
        """Load per-model default params from the registered config loader."""
        if cls._config_loader is None:
            return {}
        params = dict(cls._config_loader(model))
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