# Changelog

All notable changes to the public seam are recorded here. Versioning follows
semver: MAJOR = breaking seam change · MINOR = new capability · PATCH = fix.

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
