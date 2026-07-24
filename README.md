# llm_utils

One interface over LLM providers and serving routes: OpenAI, Anthropic (Claude),
Google (Gemini), OpenAI-compatible endpoints (DeepSeek, Z.AI, Moonshot, xAI,
OpenRouter), AWS Bedrock, local HuggingFace models, and SLURM-cluster vLLM
serving.

Originally built inside AI-security research projects and extracted into a
standalone base package: consumers track one versioned source instead of
vendored copies, and swap models or providers by changing one enum value while
the calling code stays identical.

## Install

Pin a release tag as a git dependency (recommended — upgrades are a deliberate
tag bump, so nothing changes under you mid-experiment):

```bash
uv add "llm_utils @ git+https://github.com/vacantfury/llm_utils@v2.3.0"
```

Heavy serving routes are extras; the core stays API-client-light:

```bash
uv add "llm_utils[local] @ git+https://github.com/vacantfury/llm_utils@v2.3.0"    # torch + transformers
uv add "llm_utils[bedrock] @ git+https://github.com/vacantfury/llm_utils@v2.3.0"  # boto3
```

**Stability contract:** the public seam is what `llm_utils/__init__.py` exports.
Releases follow semver (`vX.Y.Z` tags): MAJOR = breaking seam change, MINOR =
new capability, PATCH = fix. Pin a tag; never track a branch. `CHANGELOG.md`
records every release.

## Quick start

```python
from llm_utils import LLMServiceFactory, LLMModel

service = LLMServiceFactory.create(LLMModel.GPT_5_MINI, temperature=0.3)

# One prompt, one response
answer = service.chat("Explain semver in one sentence.")

# Async variant (safe inside an async runtime)
answer = await service.achat("Explain semver in one sentence.")
```

Switching provider is a one-token change — `LLMModel.CLAUDE_SONNET_4_6`,
`LLMModel.GEMINI_2_5_FLASH`, a local `LLMModel.LLAMA3_8B` — the rest of the
code is untouched.

### Batch processing

`batch_chat` takes `(id, messages)` conversations, where each message is a
`(text, image_or_None)` tuple; it returns `(id, response)` pairs in input
order. On providers with native batch APIs (OpenAI, Anthropic, Google) this
routes through the batch endpoint at ~50% of real-time cost.

```python
conversations = [
    ("q1", [("What is AI?", None)]),
    ("q2", [("What's in this image?", "/path/to/image.jpg")]),
]
results = service.batch_chat(conversations, system_message="Be concise.")
for conv_id, response in results:
    print(conv_id, response)
```

## What's in the package

- **`LLMModel`** — the model registry: one enum row per model, carrying the
  provider, API model id, per-million-token prices, and quirk flags.
- **`Provider`** — which service class the factory dispatches to.
- **`ModelQuirk`** — API-side behavior flags handled automatically (e.g.
  models that reject a custom temperature, models renaming `max_tokens` →
  `max_completion_tokens`, Gemini thinking models whose thought tokens share
  the output budget).
- **`LLMServiceFactory`** — `create(model, **kwargs)` builds the right service.
- **`BaseLLMService`** — the shared interface: `chat` / `achat` / `batch_chat`,
  rate-limit retry with backoff, and per-service usage tracking
  (`get_usage()` reports tokens + cost from the registry's prices).
- **Concrete services** — `OpenAIService`, `ClaudeService`, `GoogleService`,
  `LocalLMService` (HuggingFace on CUDA/MPS/CPU), plus Bedrock,
  OpenAI-compatible endpoints, and SLURM-cluster vLLM serving behind the same
  interface.

### Error contract

```python
from llm_utils import AccountFatalError, InvalidCredentialError, CreditsExhaustedError, FatalModelError
```

`AccountFatalError` (bad key, exhausted credits) means *stop the run* — no
retry will help; `FatalModelError` means *this model* is unusable (e.g. 404).
Long experiment loops catch these to fail fast instead of burning retries.

## Integration seams

Consumers plug project-specific behavior into the package without forking it:

```python
# Per-model default parameters from YOUR config system (called at create()):
LLMServiceFactory.set_config_loader(lambda model: my_defaults_for(model))

# Durable usage/cost accounting — invoked at the single usage choke point
# for every call made by any service:
BaseLLMService.set_usage_hook(my_ledger_writer)

# Label a service's calls for attribution in that hook:
service = LLMServiceFactory.create(LLMModel.GPT_5_MINI, label="eval-judge")

# Cluster serving lifecycle (SLURM/vLLM) — inject a server manager:
LLMServiceFactory.set_server_manager(my_cluster_manager)
```

## Credentials

Plain environment variables only — no secret files, no secret-manager
references. A `.env` in the working directory (or any parent) is loaded
automatically via `python-dotenv`.

| Provider | Env var |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Z.AI | `ZAI_API_KEY` |
| xAI | `XAI_API_KEY` |
| Moonshot | `MOONSHOT_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| AWS Bedrock | standard AWS credential chain (`AWS_PROFILE`, …) |

Note on jurisdictions: DeepSeek, Z.AI, and Moonshot are direct mainland-China
endpoints; OpenRouter is a US aggregator that can route to the same open-weight
models. This package is transport only — each consumer enforces its own data
routing policy on top.

## License

MIT — see [LICENSE](LICENSE).
