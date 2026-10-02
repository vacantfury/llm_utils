# llm_utils — consumer contract

*RENDERED from `contract.yaml` by `project_manager contract` — never edit by hand (charter §3.7 provider half, Zeus-ratified 2026-09-03). Anything not declared below is private and may change without notice.*

- **version policy:** `semver` — 1.0+: breaking = major, additive = minor, fix = patch
- **last tag:** v9.0.0 · **pyproject version:** 9.0.0
- **consume it as:** a pinned git dependency by tag in your `pyproject.toml` (charter §3.7); bump only after reading the changelog section for every tag you skip.

## Public seams

### `python-api` — python-api
The package's __all__. LLMServiceFactory.create(model, *, label=None, launch_point=None, ...) builds a service; BaseLLMService records usage through a consumer-installed usage hook, else (v7.1.0+) through the optional agent_manager call ledger (call_ledger.record_call). A failed call (v7.3.0+) records one zero-cost row with status error|timeout and error_class; a usage hook receives it only if it declares status and error_class (or **kwargs). LLMModel members are the model registry; removing a member is a MAJOR change. (v9.0.0+) Paid calls are admitted against a per-process spend cap before sending (SpendCapExceededError, an AccountFatalError); llm_utils.testing is the consumer test kit (explicit import).
- package `llm_utils` — public = the declared list below
  - `llm_utils.call_ledger`
  - `llm_utils.config`
  - `llm_utils.spend`
  - `llm_utils.testing`
  - `llm_utils.SpendStatus`
  - `llm_utils.spend_status`
  - `llm_utils.max_usd_per_run`
  - `llm_utils.reset_run_spend`
  - `llm_utils.SpendCapExceededError`
  - `llm_utils.RetentionPolicyError`
  - `llm_utils.TypeSafeService`
  - `llm_utils.Evaluation`
  - `llm_utils.NoulAnswer`
  - `llm_utils.ChoiceAnswer`
  - `llm_utils.ScoreAnswer`
  - `llm_utils.noul_question`
  - `llm_utils.choice_question`
  - `llm_utils.score_question`
  - `llm_utils.RouteStatus`
  - `llm_utils.RegistryRouteResolver`
  - `llm_utils.logical_models`
  - `llm_utils.route_models`
  - `llm_utils.routes_for`
  - `llm_utils.route_status`
  - `llm_utils.route_statuses`
  - `llm_utils.usable_routes`
  - `llm_utils.CLAUDE_API_ALLOW_ENV`
  - `llm_utils.ClaudeAPINotAllowed`
  - `llm_utils.claude_api_allowed`
  - `llm_utils.is_claude_model`
  - `llm_utils.LLMModel`
  - `llm_utils.Provider`
  - `llm_utils.ModelQuirk`
  - `llm_utils.BaseLLMService`
  - `llm_utils.UsageStats`
  - `llm_utils.LLMServiceFactory`
  - `llm_utils.AccountStatus`
  - `llm_utils.burn_rate`
  - `llm_utils.days_to_empty`
  - `llm_utils.is_mechanism_error`
  - `llm_utils.make_mechanism_error`
  - `llm_utils.strip_mechanism_error`
  - `llm_utils.FatalModelError`
  - `llm_utils.AccountFatalError`
  - `llm_utils.InvalidCredentialError`
  - `llm_utils.CreditsExhaustedError`
  - `llm_utils.BrokerError`
  - `llm_utils.BrokerModeUnsupportedError`
  - `llm_utils.OpenAIService`
  - `llm_utils.DeepSeekService`
  - `llm_utils.ZAIService`
  - `llm_utils.XAIService`
  - `llm_utils.MoonshotService`
  - `llm_utils.OpenRouterService`
  - `llm_utils.ClaudeService`
  - `llm_utils.GoogleService`
  - `llm_utils.LocalLMService`
  - `llm_utils.SlurmClusterService`
  - `llm_utils.BedrockService`
- declared consumers: psyche, agent_manager, autoflow, auto_research, courier, prospector, ties, personal_trade, personal_passive_asset

## Deprecations
| element | since | removed in | replacement |
|---|---|---|---|
| `llm_utils.ClusterModelServerManager` | v6.0.0 | v6.0.0 | `the device layer's endpoint manager, injected with LLMServiceFactory.set_server_manager(manager)` |
| `llm_utils.cluster_server_manager` | v6.0.0 | v6.0.0 | `the device layer's serving lifecycle (sbatch, endpoint pool, health); llm_utils keeps only SlurmClusterService` |
| `llm_utils.constants.MAX_SLURM_TIME_LIMIT` | v6.0.0 | v6.0.0 | `the device layer's per-cluster wall limits (partition facts)` |

## Consumers (derived from their pyprojects — never hand-listed)
- agent_manager @ v8.1.0
- auto_research @ v7.3.0
- autoflow @ v8.2.0
- courier @ v8.2.0
- llm_agent_security @ v5.0.0
- llm_guardrail_security @ v8.2.0
- llm_guardrail_security_public @ v5.4.0
- model_internals_safety @ v5.0.0
- personal_passive_asset @ v8.1.0
- personal_trade @ v8.1.0
- prospector @ v8.1.0
- psyche @ v8.1.0

## Changelog head (`[Unreleased]`)
_empty_
