# llm_utils — design record

## Founding (2026-07-23, owner-ordered)

Extracted from the owner's research repos into a standalone shared-infrastructure repo
(oikos 4th kind). The problem it solves: the package was vendored in three places and had
already diverged — updates required manual copy-paste propagation and copies silently
disagreed. Charter founding test passed: 3+ consumers concretely depend on it today, and
it owns external access (LLM provider APIs, cluster serving) behind a narrow interface.

**Seed:** the `imaging_text_attacks` copy — identified by the owner as the most-current
trunk at founding. Published as v2.1.0 (the vendored line called itself 2.0.0; the
extraction is the first standalone release).

## The three diverged copies (reconciliation map)

| Copy | Status at founding | Plan |
|---|---|---|
| `imaging_text_attacks…/src/llm_utils` | most-current trunk (owner-identified) | SEEDED here as v2.1.0. MIGRATED to the pinned dep @v2.3.0 (2026-07-23): vendored copy deleted, imports rewritten, YAML defaults wired via `set_config_loader` in the host's `src/__init__.py`, chat-template path resolves in the installed package |
| `llm_agent_security/src/llm_utils` | same ancestry, older (`base_llm_service.py` ~8.5K vs ~13K, `llm_model.py` differs) | RECONCILED at v2.2.0 (2026-07-23): full diff showed the trunk is a strict superset — zero deltas to merge (its vendored README was a stale usage doc, dropped). Its `_check_fatal_error` 404 path was a latent crash (imported `src.utils.exceptions`, which that repo never had). MIGRATED to the pinned dep @v2.3.0 (2026-07-23): vendored copy deleted, imports rewritten; no config loader wired (repo has no src/experiment) — kwargs-only defaults; the dep also brings the LLM SDKs the repo never declared |
| psyche `src/psyche/llm_utils` | DELIBERATE heavy fork: added `cost_ledger.py` + `model_policy.py`, slimmed `llm_model.py` ~35K→14K, dropped `cluster_server_manager` | RECONCILED at v2.3.0 (2026-07-23): generic improvements merged up (chat/achat + real-time Claude/Google chat, usage hook + label, OpenRouter provider+service+rows, thinking-budget quirk, Opus 4.7/4.8 no-temperature fix, API_KEY_ENV pattern, optional Pillow, new registry rows). NOT merged, by design: `cost_ledger.py` (psyche meaning — becomes a `set_usage_hook` adapter at migration), `model_policy.py` (meaning-heavy band policy, single consumer — rule of two says stay psyche-side; revisit if a second consumer wants band resolution), psyche's registry trimming (deliberate slimming, not generic), psyche's dead Gemini 3-pro/3.1-flash-lite preview rows (trunk's 2026-07-12 removal is newer verified knowledge — psyche drops them at migration). MIGRATED @v2.3.0 (2026-07-23): fork transport files deleted; `psyche.llm_utils` remains as the ADAPTER layer (cost_ledger wired via `set_usage_hook` at adapter import, model_policy over the base registry) |

Rule for all future divergence pressure: generic capability merges here; project-specific
meaning stays consumer-side as an adapter. A consumer needing a new capability requests it
here (issue / TODO), never forks the package.

## Dependency protocol (the standard consumers follow)

Canonical statement lives in CLAUDE.md (public seam, pinned uv git dep by tag, semver,
extras, no vendoring, blind-mirror vendoring exception). Design rationale:

- **Pin by tag, not branch:** consumers must never break mid-experiment because the base
  moved; an upgrade is a deliberate one-line bump the consumer makes when ready. This is
  what replaces copy-paste propagation.
- **Public seam = `__init__` exports only:** keeps refactoring freedom inside the package;
  a MAJOR bump is only needed when exported names/signatures/behavior contracts change.
- **Extras keep the core light:** API-only consumers (e.g. psyche) must not drag in
  torch/boto3. `[local]` = HF/transformers serving · `[bedrock]` = AWS. Future serving
  routes follow the same pattern (new extra, never new core deps).
- **Secrets are env vars only** (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
  AWS env credentials). The repo is public: no secret-manager references anywhere; each
  consumer fills env vars its own way.
- **Repo stays public** so public research consumers remain installable/reproducible by
  outside readers (reviewers, portfolio). Anonymous-review mirrors are the one exception:
  they vendor the source instead of carrying the dep (git URLs + uv.lock deanonymize);
  operative checklist in the owner's research-workflow skill (S12).

## Release discipline

- Tag every release `vX.Y.Z`; `CHANGELOG.md` records every release from v2.2.0 on.
- `__version__` in `src/llm_utils/__init__.py` matches `pyproject.toml` version.
- `httpx` note: migrated to `httpx2` at v2.2.0 (openai floor raised to >=2.47, the
  verified version accepting httpx2 clients as `http_client`).

## v2.2.0 decoupling decision (2026-07-23)

The last host coupling (`LLMServiceFactory._load_model_defaults` lazily importing
`src.experiment.config.load_conf`) was resolved with an **injectable config loader**
(`set_config_loader(loader)`, mirroring the existing `set_server_manager` pattern)
rather than moving the method host-side: the merge-defaults-with-kwargs behavior is
generic factory mechanics worth keeping in the seam, while WHERE defaults live
(YAML schema, file layout) is domain meaning that stays consumer-side. With no
loader registered the factory builds services from caller kwargs alone. Hosts wire
their YAML back in with one startup line — recorded in the migration plan (TODO
consumers task).
