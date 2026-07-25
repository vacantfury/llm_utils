# llm_utils — design record

## Founding (2026-07-23, owner-ordered)

Extracted from the owner's research repos into a standalone shared-infrastructure repo.
The problem it solves: the package was vendored in three places and had already
diverged — updates required manual copy-paste propagation and copies silently
disagreed. Founding test passed: 3+ consumers concretely depend on it today, and
it owns external access (LLM provider APIs, cluster serving) behind a narrow interface.

Seeded from the most-current vendored copy (owner-identified trunk). Published as
v2.1.0 (the vendored line called itself 2.0.0; the extraction is the first
standalone release).

## Reconciliation (2026-07-23)

Three diverged vendored copies were reconciled into this trunk across v2.1.0 →
v2.3.0: the trunk proved a strict superset of one copy (zero deltas), and the
generic capabilities of a deliberately forked copy were merged up (chat/achat
real-time paths, usage hook + label, OpenRouter provider, thinking-budget quirk,
no-temperature quirks, API_KEY_ENV pattern, optional Pillow, registry rows).
Consumer-specific meaning (cost-ledger destinations, routing-policy bands) stayed
consumer-side as adapters over the seam.

Rule for all future divergence pressure: generic capability merges here;
project-specific meaning stays consumer-side as an adapter. A consumer needing a
new capability requests it here (issue / TODO), never forks the package.

## Migration campaign (2026-07-24, owner-ordered)

Beyond the vendored copies, remaining repos hand-rolling direct SDK clients
adopted the package. v2.4.0 first added the capabilities those clients actually
used and the seam lacked — `chat_structured` (provider parse APIs), the
resumable Claude batch trio (`submit_batch_chat` / `batch_chat_status` /
`harvest_batch_chat`), `api_params` pass-through on OpenAI-family services, and
the SDK-default credential fallback on `ClaudeService` — then the consumers
migrated the same day.

**Standing exclusion (owner decision 2026-07-24):** repos with outside
collaborators on private hosting do not take this dep — they keep self-contained
LLM code rather than depending on a personal-account repo.

Tag bumps stay per-consumer and deliberate.

## Dependency protocol (the standard consumers follow)

Canonical statement lives in CLAUDE.md (public seam, pinned uv git dep by tag,
semver, extras, no vendoring, blind-mirror vendoring exception). Design rationale:

- **Pin by tag, not branch:** consumers must never break mid-experiment because the base
  moved; an upgrade is a deliberate one-line bump the consumer makes when ready. This is
  what replaces copy-paste propagation.
- **Public seam = `__init__` exports only:** keeps refactoring freedom inside the package;
  a MAJOR bump is only needed when exported names/signatures/behavior contracts change.
- **Extras keep the core light:** API-only consumers must not drag in
  torch/boto3. `[local]` = HF/transformers serving · `[bedrock]` = AWS. Future serving
  routes follow the same pattern (new extra, never new core deps).
- **Secrets are env vars only** (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
  AWS env credentials). The repo is public: no secret-manager references anywhere; each
  consumer fills env vars its own way.
- **Repo stays public** so public research consumers remain installable/reproducible by
  outside readers (reviewers, portfolio). Anonymous-review mirrors are the one exception:
  they vendor the source instead of carrying the dep (git URLs + uv.lock deanonymize).
- **No site-specific information in committed files** (v3.0.0, owner-ordered):
  cluster names, account details, and research-plan internals never appear in
  public files — code stays generic and config-driven; site facts live in
  gitignored private notes.

## Release discipline

- Tag every release `vX.Y.Z`; `CHANGELOG.md` records every release from v2.2.0 on.
- `__version__` in `src/llm_utils/__init__.py` matches `pyproject.toml` version.
- `httpx` note: migrated to `httpx2` at v2.2.0 (openai floor raised to >=2.47, the
  verified version accepting httpx2 clients as `http_client`).

## v2.2.0 decoupling decision (2026-07-23)

The last host coupling (`LLMServiceFactory._load_model_defaults` lazily importing
the host's config loader) was resolved with an **injectable config loader**
(`set_config_loader(loader)`, mirroring the existing `set_server_manager` pattern)
rather than moving the method host-side: the merge-defaults-with-kwargs behavior is
generic factory mechanics worth keeping in the seam, while WHERE defaults live
(YAML schema, file layout) is domain meaning that stays consumer-side. With no
loader registered the factory builds services from caller kwargs alone. Hosts wire
their YAML back in with one startup line.
