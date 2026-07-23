# Changelog

All notable changes to the public seam are recorded here. Versioning follows
semver: MAJOR = breaking seam change · MINOR = new capability · PATCH = fix.

## v2.3.0 — 2026-07-23

Reconciliation release (psyche fork): generic improvements from
`psyche/src/psyche/llm_utils` merged up. Psyche-specific meaning (cost-ledger
sqlite destination, PRC routing policy bands) stays in psyche as adapters.

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

Reconciliation release (TODO task 1): `llm_agent_security`'s vendored copy
diffed against this trunk — the trunk is a strict superset, no deltas to merge.

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

First standalone release: extracted from the owner's research repos
(seeded from the `imaging_text_attacks` vendored copy, the most-current trunk).
See `text_docs/design.md` for the founding record and reconciliation map.
