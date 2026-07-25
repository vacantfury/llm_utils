# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

**Visibility: public.** Open-source-safe rules are MANDATORY: no secret values, no
secret-manager references, no personal data, no hardcoded personal paths or
hostnames anywhere in committed files. All credentials reach the code as plain environment
variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, AWS credential env vars).

## What this repo is

**Shared infrastructure** (registered in the owner's private hub as an infrastructure
repo, never a portfolio card). The LLM-provider utility layer historically vendored
inside the owner's research repos, extracted 2026-07-23 (owner-ordered) into one base
repo so updates propagate by version bump instead of copy-paste.

One public seam over many serving routes: OpenAI / Anthropic / Gemini API clients, AWS
Bedrock, local HuggingFace models, and SLURM-cluster vLLM serving (generic SLURM discovery —
no site-specific hostnames).

## Scope boundary (the load-bearing line)

- **This repo owns:** provider transport and serving mechanics — model/provider registry
  (`LLMModel`, `Provider`), the service interface (`BaseLLMService`), concrete services,
  usage-stat primitives, cluster server lifecycle. Example: adding a new provider client or
  a retry policy belongs HERE.
- **Stays in consumers:** all domain meaning. Example: a private consumer's cost-ledger
  wiring and its jurisdiction-based model-routing policy are that consumer's adapters
  ON TOP of this package; research repos' attack/eval logic stays in those repos.
  Infrastructure is domain-blind (transport belongs to infrastructure, meaning belongs
  to domains).

## Dependency protocol (charter §3.7 — the standard for every consumer)

- Consumers declare a **pinned uv git dependency by tag** and import only the public seam
  (the package `__init__` exports):
  `uv add "llm_utils @ git+https://github.com/vacantfury/llm_utils@v3.0.0"`
- Upgrades are deliberate: bump the tag in the consumer when the consumer chooses.
  Never track a branch; never vendor a copy back into a consumer.
- **Semver discipline:** MAJOR = breaking change to the public seam · MINOR = new
  capability · PATCH = fix. Every release gets a git tag `vX.Y.Z`; internals behind the
  seam may change freely.
- Heavy deps are extras, not core: `llm_utils[local]` (torch/transformers),
  `llm_utils[bedrock]` (boto3). Core stays API-client-light.
- **Anonymous paper mirrors never carry this repo as a git dep** — the dep URL and
  `uv.lock` carry the owner's GitHub handle. Blind mirrors VENDOR the package source
  (operative rule: the global research-workflow skill, S12 pre-submit checklist).
- **Shared-private repos never carry this dep either** (owner-ordered 2026-07-24): a
  repo with outside collaborators on private hosting keeps self-contained LLM code
  instead of depending on a personal-account repo.

## Layout

- `src/llm_utils/` — the package (src layout, hatchling build).
- `text_docs/design.md` — founding record + reconciliation plan (three diverged copies
  are being merged; this repo seeded from the most-current trunk).
- `TODO.md` — gitignored (task text is personal); registered in the owner's task system.
- `text_docs/private/` — gitignored; site-specific cluster/account facts and consumer
  records live there, NEVER in committed files (owner order 2026-07-24).

## Task system

Repo tasks live in root `TODO.md` (gitignored) per the global task standard; finished
items move to the owner's central task archive. The repo is registered in the owner's
hub task registry.
