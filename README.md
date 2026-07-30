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
uv add "llm_utils @ git+https://github.com/vacantfury/llm_utils@v5.0.0"
```

Heavy serving routes are extras; the core stays API-client-light:

```bash
uv add "llm_utils[local] @ git+https://github.com/vacantfury/llm_utils@v5.0.0"    # torch + transformers
uv add "llm_utils[bedrock] @ git+https://github.com/vacantfury/llm_utils@v5.0.0"  # boto3
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
order. On OpenAI, Anthropic, AND Google, `batch_chat` **auto-routes by
estimated job cost**: jobs estimated at ≥ `batch_threshold_usd` (default $1)
go through the provider's native batch API (50% price, queued — usually much
faster than the 24h window); smaller jobs fan out concurrent real-time calls
(full price, seconds turnaround). Override with `use_batch_api=True/False` at
construction. OpenAI-COMPATIBLE endpoints (DeepSeek, Z.AI, xAI, …) have no
batch API and always run real-time concurrent.

```python
conversations = [
    ("q1", [("What is AI?", None)]),
    ("q2", [("What's in this image?", "/path/to/image.jpg")]),
]
results = service.batch_chat(conversations, system_message="Be concise.")
for conv_id, response in results:
    print(conv_id, response)
```

On `OpenAIService`, `ClaudeService`, and `GoogleService`, batches are also
**resumable across processes**: `submit_batch_chat(...)` returns the provider
batch id without waiting; persist it anywhere, then
`harvest_batch_chat(batch_id)` from any later invocation (None while still
running); `batch_chat_status(batch_id)` polls. Google caveat: its inline
batch carries no per-item id, so harvested results are positional (`"0"`,
`"1"`, … in submission order) — keep your own id list from submit time.

### Token logprobs

Where the serving route exposes them (the SLURM-cluster vLLM path),
`batch_chat_with_logprobs` returns `(id, text, logprobs)` triples —
`batch_chat` plus the OpenAI-schema per-token payload
(`{"content": [{"token", "logprob", "top_logprobs": [...]}, ...]}`; None on
transport failure). Typical use: turning a binary classifier verdict into a
continuous score via the verdict token's confidence.

```python
results = service.batch_chat_with_logprobs(conversations, top_logprobs=5)
for conv_id, text, logprobs in results:
    first_token = logprobs["content"][0] if logprobs else None
```

`batch_chat`'s `(id, text)` return shape is unchanged. Only
`SlurmClusterService` implements this today; every other service raises
`NotImplementedError` (Anthropic and Google APIs return no logprobs at all).

### Structured output

`chat_structured` returns a validated pydantic instance via the provider's
parse API (OpenAI structured outputs / Anthropic `messages.parse`):

```python
from pydantic import BaseModel

class Verdict(BaseModel):
    relevant: bool
    reason: str

verdict = service.chat_structured("Is this headline market-moving? ...", Verdict)
```

Unlike `chat` (which reports transport failures as mechanism-error strings —
see `is_mechanism_error`), `chat_structured` raises on API failure after
retries, and may return None when the model's output failed validation.

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
- **Concrete services** — all eleven serving routes are exported from the
  seam: `OpenAIService`, `ClaudeService`, `GoogleService`, `BedrockService`,
  `LocalLMService` (HuggingFace on CUDA/MPS/CPU), `SlurmClusterService` (+
  `ClusterModelServerManager`), and the OpenAI-compatible endpoint services
  (`DeepSeekService`, `ZAIService`, `XAIService`, `MoonshotService`,
  `OpenRouterService`) — one interface everywhere.

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

# Extra request params merged verbatim into OpenAI-family API calls
# (constructor-level here; also accepted per call):
service = LLMServiceFactory.create(
    LLMModel.GPT_5_MINI,
    api_params={"response_format": {"type": "json_object"}},
)
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
| AWS Bedrock | standard AWS credential chain (`AWS_PROFILE`, `AWS_REGION` / `AWS_DEFAULT_REGION`, or explicit `aws_profile` / `aws_region` kwargs) |
| HuggingFace (gated models, local + cluster serving) | `HUGGINGFACE_TOKEN` (or `HF_TOKEN`) |

Note on jurisdictions: DeepSeek, Z.AI, and Moonshot are direct mainland-China
endpoints; OpenRouter is a US aggregator that can route to the same open-weight
models. This package is transport only — each consumer enforces its own data
routing policy on top.

## License

MIT — see [LICENSE](LICENSE).
