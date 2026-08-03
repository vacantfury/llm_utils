# Changelog

All notable changes to the public seam are recorded here. Versioning follows
semver: MAJOR = breaking seam change · MINOR = new capability · PATCH = fix.

## v5.1.0 — 2026-08-02

`ClusterModelServerManager` hardening to the family cluster-job standard §5
(teardown & zombie prevention). Additive: the public seam is unchanged;
existing configs keep working (all new config keys have fail-safe defaults).

**Added:**

- **Upfront config validation (T1):** every required key for every instance
  is checked BEFORE the first sbatch, and all instances' sbatch scripts are
  generated before any submission — a bad config (missing key, missing chat
  template) now fails with zero jobs up instead of dying mid-batch. All
  missing keys are reported at once.
- **Durable job ledger (T2):** every submitted job id is persisted to
  `<log_dir>/slurm_jobs_<run_id>.json` at submission time (atomic writes),
  so even a SIGKILL'd orchestrator leaves a reap list. At the next run's
  first `start_server`, unclosed ledgers from prior runs are reaped —
  scancel only when the owning orchestrator is provably dead (same host,
  pid gone); a live pid (concurrent run) is left alone and a foreign-host
  ledger is reported, never guessed at. `reap_orphans: false` = report-only.
- **Graceful-death teardown (T3):** atexit + SIGTERM/SIGINT handlers scancel
  owned jobs (previously only `__del__` tried, which Python does not
  guarantee). Handlers chain to whatever was installed before;
  `install_signal_handlers: false` keeps atexit only.
- **Eviction hysteresis + recovery (T4):** a pool endpoint is evicted only
  after `eviction_failure_threshold` (default 3) CONSECUTIVE health-check
  failures (was: one failed ping = permanent eviction), with a structured
  `EVICTION …` log line as the threshold's tuning-data stream. Evicted
  endpoints whose SLURM job is still alive are re-probed on the maintenance
  cadence and re-added on recovery; once the job leaves the queue they are
  dropped for good. `acquire_endpoint`/`wait_for_first_server` no longer
  report "all failed" while an eviction is pending recovery;
  `get_server_status` gains `num_evicted`.
- **Per-model timeouts (T5):** `slurm_cmd_timeout`, `health_check_timeout`,
  and `health_recheck_interval` now come from each model's OWN config
  (was: first-registered-config-wins for every model).
- **Run-scoped log subdir:** `run_scoped_logs: true` nests sbatch
  stdout/stderr in `<log_dir>/run_<run_id>/` so concurrent runs from one
  CWD don't interleave; `log_dir` (default `logs`) is now configurable.
  Ledgers always stay in the base dir so later runs can discover them.

**Fixed:**

- The pool's Event-as-condvar (`_pool_changed`) replaced with a real
  `threading.Condition` — the old check-then-wait pattern could miss a
  wakeup slipped between a waiter's check and its wait under concurrent
  acquire/release (bounded by the poll timeout, now gone entirely).
- `sbatch` submission now retries transient NON-timeout failures too
  (e.g. "Socket timed out on send/recv operation" exits nonzero), with a
  short backoff; previously only `subprocess.TimeoutExpired` retried and
  any nonzero exit killed the run on the first blip.

36 new offline tests (`tests/test_cluster_manager.py`); suite total 113.

## v5.0.0 — 2026-07-30

Full-system audit release: every serving route audited against the base
contract; defects fixed, the batch seam made symmetric across the three major
providers, and a 60-test offline suite added (`tests/`, `uv run pytest`).

**BREAKING:**

- `ClaudeService.batch_chat` and `GoogleService.batch_chat` now auto-route
  exactly like OpenAI's: jobs under `batch_threshold_usd` (default $1) run
  CONCURRENT REALTIME (full price, seconds) instead of always queuing on the
  native batch API. Bulk callers keep 50% batch pricing automatically (big
  jobs still batch); pass `use_batch_api=True` to force native batch always.
  The routing seam (`use_batch_api`, `batch_threshold_usd`, estimator) moved
  to `BaseLLMService` — shared by all three.
- `LLMModel.from_string` now RAISES ValueError on a model id registered on
  multiple serving routes (e.g. `meta-llama/Meta-Llama-3-8B-Instruct` = local
  AND cluster) instead of silently returning the local row; enum names still
  resolve every row unambiguously.
- Recorded Gemini output tokens now include THINKING tokens (billed as output
  by Google but reported separately) — recorded costs on thinking Gemini
  models go UP; they were silently understated before.
- `GoogleService` no longer raises on failed/cancelled/expired batch jobs:
  terminal non-success states return per-item results — partial (already
  billed) results are collected, missing items become mechanism errors.
- Bedrock's `BedrockCredentialsError` / `BedrockAccessError` moved into the
  package exception hierarchy (`InvalidCredentialError` / `FatalModelError`
  subclasses respectively) so account-fatal handlers catch expired AWS creds.
- Removed the dead `config=None` constructor parameter from every service,
  and the never-used `OLLAMA_*` / `DEFAULT_API_BATCH_SIZE` constants.
- `LocalLMService` prompts now go through `tokenizer.apply_chat_template`
  when the tokenizer ships one (correct role markers for instruct models;
  outputs will differ from the old newline-join, which remains the fallback).

**Added:**

- **Dual-route registry for Chinese models** — every mainland-endpoint model
  that OpenRouter also serves now has a US-hosted twin row, and the routes are
  linked: `Provider.jurisdiction` (`"us"` / `"prc"` / `"self"`),
  `ModelSpec.weights` identifying the underlying model, and
  `LLMModel.route_twins()` / `routes()` / `us_route()` / `self_route()` to move
  between routes without hardcoding ids. 12 of 14 mainland rows resolve to a
  verified OpenRouter twin (`glm-4.7-flashx` and `kimi-k2.7-code-highspeed`
  are genuinely absent there, so they report no US route). New OpenRouter
  rows: GLM-5, GLM-5-turbo, GLM-5V-turbo, GLM-4.7, GLM-4.7-flash, GLM-4.6V,
  Kimi-K2.7-code. Claude direct↔Bedrock and DeepSeek-V3.2 Bedrock↔cluster are
  linked the same way. Jurisdiction is a transport FACT — the routing policy
  stays with consumers.
- Resumable batch trio on `GoogleService` (`submit_batch_chat` /
  `batch_chat_status` / `harvest_batch_chat`) — all three major providers now
  have it; base-class stubs give every other service a clean
  `NotImplementedError`. Google results are positional ("0", "1", … — its
  inline batch has no custom_id; keep your own id list).
- All eleven service classes + `ClusterModelServerManager` exported from the
  public seam (previously only four services were importable from
  `llm_utils`).
- `api_params` pass-through on `ClaudeService` and `SlurmClusterService`
  (parity with the OpenAI family); temperature quirk gating unified through
  `_accepts_temperature` on Bedrock and the cluster service.
- Thinking-headroom handling on `ClaudeService` for always-thinking models
  (Opus 5 / Fable 5, now marked `THINKING_SHARES_OUTPUT_BUDGET` in the
  registry) — thought tokens no longer starve the visible-text budget.
- Offline test suite: 60 tests over routing, cost math (batch discount on all
  three providers), registry integrity, factory dispatch, usage hook,
  mechanism-error contract. Registry prices spot-verified against official
  provider pages 2026-07-30: all 38 checked rows MATCH.

**Changed:** OpenRouter row prices re-verified against openrouter.ai
2026-07-30 and corrected — `OR_GLM_5_2` $0.93/$3.00 → $0.36/$0.75 (launch
promo), `OR_KIMI_K3` → $2.90/$15.00, `OR_KIMI_K2_6` → $0.60/$3.41,
`OR_QWEN_3_7_MAX` → $1.475/$4.425, `OR_MINIMAX_M3` → $0.24/$0.96 (promo).
Promo rows are marked in the registry: they will expire and need re-checking.

**Fixed:**

- `LocalLMService` crashed at construction on CUDA (`device_map="auto"` +
  pipeline `device=` conflict); batch-fallback double-generated and
  double-recorded usage for items that had already completed; images were
  silently dropped (now warned); HF token read at import time (now at load
  time, `HF_TOKEN` accepted as alias).
- Bedrock Claude 5.x registry rows lacked `NO_CUSTOM_TEMPERATURE` — every
  request to them 400'd; Bedrock retry now uses the shared case-insensitive
  rate-limit classifier with capped backoff, and 404s raise `FatalModelError`.
- `ClaudeService`: multi-block responses no longer truncate to the first text
  block; the batch result stream is retry-wrapped (was the one unwrapped
  network call); `harvest_batch_chat` no longer re-records usage on repeat
  harvests in the same process; batch item errors carry the provider's error
  detail.
- `GoogleService`: response-count mismatches fill with mechanism errors
  instead of silently truncating via `zip`; Optional SDK fields guarded.
- OpenAI/vLLM realtime retry matched any error containing "rate" (e.g.
  "failed to generate") — now uses the shared classifier;
  `batch_chat_status`/`harvest_batch_chat` guard compatible endpoints.
- `SlurmClusterService` leaked one `AsyncOpenAI`+httpx2 client per call
  (now closed in the event loop).
- `ClusterModelServerManager`: monitor thread no longer dies permanently on a
  teardown race (exception-guarded, shutdown-race guard on pool appends); a
  transient `squeue` failure no longer permanently blackholes a live server;
  sbatch `logs/` is pre-created (SLURM opens the log files before the script
  body runs); wall-time clamping compares numerically (lexicographic compare
  mis-clamped unpadded specs).
- API keys on Claude/Google read at construction (was import-time snapshot).

Migration notes: latency-sensitive small `batch_chat` jobs on Claude/Google
get FASTER but bill at full realtime price — pass `use_batch_api=True` to
keep forcing 50% batch pricing. Callers resolving the ambiguous local/cluster
Llama-3-8B id by string must switch to the enum member or enum name.

## v4.0.0 — 2026-07-30

**BREAKING (behavior, not signatures):** `OpenAIService.batch_chat` on the
real OpenAI endpoint now auto-routes big jobs through the native **Batch API**
(50% price, 24h completion window — usually much faster). Small jobs keep the
concurrent realtime path, so under-threshold behavior is unchanged.

- **Added:** cost-threshold auto-routing on `OpenAIService.batch_chat` —
  jobs whose estimated worst-case cost (chars/4 input estimate, `max_tokens`
  output ceiling) is ≥ `batch_threshold_usd` (default $1) go through the
  Batch API; below, the realtime concurrent path. Constructor overrides:
  `use_batch_api=None` (auto) / `True` (always batch — note `chat` funnels
  through `batch_chat`, so forcing True queues singles too) / `False` (never).
  The JSONL transport is in-memory (Files API upload, no disk); batch files
  are deleted after harvest. OpenAI-COMPATIBLE endpoints (DeepSeek, Z.AI,
  xAI, Moonshot, OpenRouter) have no `/v1/batches` and always run realtime.
- **Added:** resumable batch trio on `OpenAIService` — `submit_batch_chat` /
  `batch_chat_status` / `harvest_batch_chat`, same contract as
  `ClaudeService`'s (persist the batch id, harvest from any later process).
  A poll timeout in blocking `batch_chat` raises with the batch id in the
  message; the batch keeps running server-side and stays harvestable.
- **Fixed:** `.env` loading now finds a FIFO `.env` — secret managers that
  mount the file without plaintext on disk (1Password Environments) serve it
  as a FIFO, which python-dotenv's own `find_dotenv` (`isfile`) skips; the
  CWD-upward search is now ours and accepts regular files and FIFOs alike.
- **Fixed:** batch-path cost recording on `ClaudeService` and `GoogleService`
  overstated spend 2× — both recorded realtime list prices while the batch
  APIs bill at 50%. All batch paths now apply
  `BaseLLMService.BATCH_COST_DISCOUNT` (0.5); registry prices stay realtime
  list. Realtime paths (`chat`, `chat_structured`) were and remain full price.

Migration note for consumers: bump deliberately. If a run needs realtime
turnaround regardless of size, pass `use_batch_api=False` at construction.

## v3.1.0 — 2026-07-24

- **Added:** `batch_chat_with_logprobs(conversations, system_message=...,
  top_logprobs=5)` — `batch_chat` plus the OpenAI-schema per-token logprob
  payload as a third tuple element: `(id, text, logprobs)`, where *logprobs*
  is `{"content": [{"token", "logprob", "top_logprobs": [...]}, ...]}` or None
  on mechanism error / when the server sent none. A separate method on
  purpose: `batch_chat`'s `(id, text)` return shape is load-bearing for every
  consumer and stays untouched. Implemented on `SlurmClusterService` (vLLM's
  OpenAI-compatible endpoint returns logprobs when asked); the base default
  raises `NotImplementedError` — Anthropic and Google APIs do not expose
  logprobs at all, so their services keep the default (a permanent gap, not a
  TODO). Motivating consumer: safety-guard threshold calibration — binary
  `safe`/`unsafe` verdicts have no dial to sweep, the verdict token's
  confidence is that dial.

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
