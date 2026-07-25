# Changelog

All notable changes to the public seam are recorded here. Versioning follows
semver: MAJOR = breaking seam change · MINOR = new capability · PATCH = fix.

## v3.0.0 — 2026-07-24

Privacy scrub + model-registry refresh. **BREAKING:** the cluster serving route
is renamed to generic SLURM terms — `Provider.SLURM_CLUSTER`
(`"slurm_cluster"`), `SlurmClusterService`, module
`llm_services/slurm_cluster_service.py` (previously site-named identifiers).
Site-specific cluster/account details were removed from committed files; the
package was always config-driven, so runtime behavior is unchanged.

- **Added:** Claude 5 family on the Anthropic provider — `CLAUDE_OPUS_5`
  ($5/$25), `CLAUDE_SONNET_5` ($3/$15; intro $2/$10 through 2026-08-31),
  `CLAUDE_FABLE_5` ($10/$50). All three reject a custom temperature and run a
  1M context; Fable 5 requires 30-day data retention.
- **Added:** `GPT_5_6_LUNA` ($1/$6 — the 5.6 launch's third tier);
  `GEMINI_3_6_FLASH` ($1.50/$7.50) + `GEMINI_3_5_FLASH_LITE` ($0.30/$2.50),
  both GA 2026-07-21.
- **Removed:** `GEMINI_2_0_FLASH`, `GEMINI_2_0_FLASH_LITE` — the provider shut
  them down 2026-06-01; calls error.
- **Fixed:** prices — `GPT_5` input $1.25 (was $2.50), `GPT_5_NANO` $0.05/$0.40
  (was $0.20/$1.25), `GEMINI_2_5_FLASH_LITE` $0.10/$0.40 (was $0.075/$0.30).
  A wrong "retires 2026-07-23" warning on `GPT_5_1`/`GPT_5_2` removed (both
  live with no scheduled retirement, verified 2026-07-24).
- **Docs:** provider shutdown schedules annotated in the registry (OpenAI
  2026-10-23 and 2026-12-11 batches; Gemini 2.5 trio 2026-10-16).

## v2.4.0 — 2026-07-24

Migration-campaign release: capabilities the remaining consumers needed to
drop their hand-rolled SDK clients.

- **Added:** `chat_structured(prompt, output_schema, system_message=...)` —
  one prompt → one validated pydantic instance via the provider's parse API
  (`chat.completions.parse` on OpenAI-family services, `messages.parse` on
  Claude). Raises on API failure after retries (a typed return has no
  mechanism-error-string channel); may return None on schema-validation
  failure. Base declares it; OpenAI + Claude implement it.
- **Added:** resumable batch API on `ClaudeService` — `submit_batch_chat`
  (returns the provider batch id without waiting), `batch_chat_status`,
  `harvest_batch_chat` (None while running, results once ended). Persist the
  id anywhere and harvest from a later process; `batch_chat` remains the
  blocking composition of the same pieces.
- **Added:** `api_params` pass-through on `OpenAIService` (constructor and
  per-call kwarg) — extra request params (`response_format`, `seed`, `top_p`,
  …) merged verbatim into the API call. Applies to all OpenAI-compatible
  subclasses.
- **Added:** mechanism-error helpers (`is_mechanism_error`,
  `make_mechanism_error`, `strip_mechanism_error`) exported from the public
  seam (previously only importable via the `base_llm_service` submodule).
- **Changed:** `ClaudeService` without an explicit/env API key now falls back
  to the SDK's own credential resolution (auth token, CLI-login keychain)
  before failing; still raises `ValueError` at construction when nothing
  resolves. `anthropic` floor raised to >=0.116 (verified `messages.parse`
  support).

## v2.3.0 — 2026-07-23

Reconciliation release: generic improvements from a consumer's deliberate fork
merged up. Consumer-specific meaning (cost-ledger destination, routing-policy
bands) stays consumer-side as adapters.

- **Added:** `chat()` / `achat()` single-prompt convenience on
  `BaseLLMService`; `ClaudeService.chat` and `GoogleService.chat` override
  with the REAL-TIME APIs (Messages / generate_content) — the batch APIs
  queue singles for minutes.
- **Added:** consumer-installable usage hook (`BaseLLMService.set_usage_hook`)
  at the `_record_usage` choke point + `label=` accounting tag on
  `LLMServiceFactory.create` — the seam for durable cost ledgers.
- **Added:** `OpenRouterService` + `Provider.OPENROUTER` + `OR_*` model rows
  (US aggregator over hosted open weights); `OPENROUTER_API_KEY` /
  `OPENROUTER_API_URL`.
- **Added:** `ModelQuirk.THINKING_SHARES_OUTPUT_BUDGET` + thinking headroom in
  `GoogleService` (thought tokens no longer starve the caller's visible-text
  budget); quirk set on Gemini 2.5/3.x flash+pro rows.
- **Added:** registry rows — `CLAUDE_OPUS_4_8`, `GLM_5_TURBO`, `GLM_5V_TURBO`,
  `GROK_4_20_*` trio, `GROK_BUILD_0_1`. `ModelQuirk` and `UsageStats` now
  exported from the seam.
- **Fixed:** `CLAUDE_OPUS_4_7` marked `NO_CUSTOM_TEMPERATURE` (it rejects a
  custom temperature with a 400); Claude batch requests now gate temperature
  through the shared quirk rule (`BaseLLMService._accepts_temperature`).
- **Fixed:** `constants.py` loads `.env` via CWD-upward search (`load_dotenv()`)
  — the old path relative to the module file pointed into site-packages once
  installed as a dependency.
- **Changed:** `OpenAIService`-family key lookup via `API_KEY_ENV` class
  attribute (`os.getenv` at construction) instead of the import-time `API_KEY`
  constant. Migration note: any subclass overriding `API_KEY` must switch to
  `API_KEY_ENV` (no known consumer does; verified against all three).
- **Changed:** Pillow is now optional for `GoogleService` text-only use
  (image messages raise a clear error if Pillow is absent).

## v2.2.0 — 2026-07-23

Reconciliation release: a second host repo's vendored copy diffed against this
trunk — the trunk is a strict superset, no deltas to merge.

- **Fixed:** `exceptions.py` hierarchy restored to the hosts' original —
  `AccountFatalError` base class reinstated, with `InvalidCredentialError` and
  `CreditsExhaustedError` inheriting from it (the v2.1.0 extraction had
  flattened all three to bare `Exception`, breaking consumers that catch
  `AccountFatalError`).
- **Added:** exceptions exported from the public seam (`llm_utils.FatalModelError`,
  `AccountFatalError`, `InvalidCredentialError`, `CreditsExhaustedError`).
- **Changed (decoupling):** `LLMServiceFactory` no longer imports the host
  repo's `src.experiment.config.load_conf`. Per-model default params now come
  from an injectable loader: consumers call
  `LLMServiceFactory.set_config_loader(loader)` at startup; with no loader,
  services are built from caller kwargs alone. This removes the last host
  coupling — the package now stands fully alone.
- **Changed (deps):** `httpx` → `httpx2>=2.9` (owner tooling standard);
  `openai` floor raised to `>=2.47`, the verified version accepting
  `httpx2.AsyncClient` as `http_client`.

## v2.1.0 — 2026-07-23

First standalone release: extracted from the owner's research repos (seeded
from the most-current vendored copy among the host repos). See
`text_docs/design.md` for the founding record.
