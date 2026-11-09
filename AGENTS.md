# llm_utils

**Visibility: public.** Open-source-safe rules are MANDATORY: no secret values, no secret-manager references, no
personal data, no hardcoded personal paths or hostnames anywhere in committed files. All credentials reach the code as
plain environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, AWS credential env vars).

## Purpose

**Shared infrastructure** (registered in the owner's private hub as an infrastructure repo, never a portfolio card).
The LLM-provider utility layer the owner's research repos once vendored, held in one base repo so updates propagate by
version bump instead of copy-paste.

One public seam over many serving routes: OpenAI / Anthropic / Gemini API clients, AWS Bedrock, local HuggingFace
models, and SLURM-served vLLM endpoints (the client side only; the SLURM serving lifecycle lives in the owner's
device-management layer). History: `docs/decisions/`.

## Scope boundary (the load-bearing line)

- **This repo owns:** provider transport: model/provider registry (`LLMModel`, `Provider`), the service interface
  (`BaseLLMService`), concrete services, usage-stat primitives, account-status primitives (per-provider credit-balance
  queries + pure burn-rate math). Example: adding a new provider client or a retry policy belongs HERE.
- **Stays in consumers:** all domain meaning. Example: a private consumer's cost-ledger wiring and its
  jurisdiction-based model-routing policy are that consumer's adapters ON TOP of this package; research repos'
  attack/eval logic stays in those repos. Infrastructure is domain-blind (transport belongs to infrastructure, meaning
  belongs to domains).
- **Stays in the device layer:** the SLURM serving lifecycle (sbatch generation/submission, endpoint pool, health
  monitoring) lives in the owner's private device-management repo. `SlurmClusterService` consumes it through a
  duck-typed contract, `acquire_endpoint(model_id) -> base_url` / `release_endpoint(model_id, endpoint)`, injected via
  `LLMServiceFactory.set_server_manager()`; this package never imports it. Cluster topology or fleet knowledge in the
  provider seam is a layering violation.

## Layout

- `src/llm_utils/`: the package (src layout, hatchling build).
- `tests/`: the offline test suite (`uv run pytest`).
- `text_docs/design.md`: founding record + reconciliation record (how the three vendored copies were merged into this
  trunk, and the rule for future divergence).
- `text_docs/private/`: gitignored; site-specific cluster/account facts and consumer records.
- `CHANGELOG.md`: the public seam's changes per release. `project_manager.yaml`, `.github/workflows/ci.yml`: build
  declaration and CI. `stores.yaml`: data declaration. `docs/decisions/`: history.

## Tasks

Tasks live in this repo's task store (truth: datastore `data/llm_utils/llm_utils.sqlite`); read them with
`uv run --project ~/oikos/infrastructure/datastore python -m datastore task list` from the repo root; write only
through the same command's store verbs (task-management standard). There is no TODO.md; the owner's view is the work
board. Finished items move to the owner's central task archive. The repo is registered in the owner's hub task
registry.

## Commands

- Land from the checkout: `uv run -q --project ~/oikos/infrastructure/project_manager project_manager lane push
  --repo . -m "<message>" <files>`.
- A consumer adds the dependency: `uv add "llm_utils @ git+https://github.com/vacantfury/llm_utils@v6.0.0"`.
- A release bumps `version` in `pyproject.toml`, records the seam delta in `CHANGELOG.md` and tags `vX.Y.Z`.

## Verification

- The tests run in this repo's CI (`.github/workflows/ci.yml`) after the push.
- Live verification stays a scratchpad smoke script (needs real keys via the consumer's own secret injection), never a
  committed test.

## Working here

### Dependency protocol (charter §3.7, the standard for every consumer)

- Before installing dependencies, downloading models, or operating an existing remote deployment, read
  `text_docs/private/site_notes.md` when available and follow its canonical host-policy references. Verify configured
  storage/cache paths before invoking installers; public library defaults do not establish a deployment's storage
  policy. Keep private host details out of public files.
- Consumers declare a **pinned uv git dependency by tag** and import only the public seam (the package `__init__`
  exports).
- Upgrades are deliberate: bump the tag in the consumer when the consumer chooses. Never track a branch; never vendor a
  copy back into a consumer.
- **Semver discipline:** MAJOR = breaking change to the public seam · MINOR = new capability · PATCH = fix. Every
  release gets a git tag `vX.Y.Z`; internals behind the seam may change freely.
- Heavy deps are extras, not core: `llm_utils[local]` (torch/transformers), `llm_utils[bedrock]` (boto3). Core stays
  API-client-light.
- **Anonymous paper mirrors never carry this repo as a git dep**: the dep URL and `uv.lock` carry the owner's GitHub
  handle. Blind mirrors VENDOR the package source (operative rule: the global research-workflow skill, S12 pre-submit
  checklist).
- **Shared-private repos MAY carry this dep:** the package is public, so outside collaborators install the same pinned
  git dep as any consumer. Only anonymous paper mirrors keep the vendoring rule above.

### Code and tests

- Tests: NO network, NO API keys: provider calls are monkeypatched/faked, services are constructed with a dummy
  `api_key`.
- Site-specific cluster/account facts and consumer records live in `text_docs/private/`, NEVER in committed files.
