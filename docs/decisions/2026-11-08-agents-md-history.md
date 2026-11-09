# History taken out of AGENTS.md (2026-11-08)

The project-structure standard (project_manager `text_docs/standards/project-structure.md`, section 1) keeps history
out of rule text. When AGENTS.md was brought to that form, the dates, owner rulings and task ids that stood beside its
rules moved here, under the rule they explain. The rules themselves stayed in AGENTS.md. Removed as rules held
elsewhere: the tests line's "Every release runs it" and "a change to routing, cost math, the registry, or the
mechanism-error contract gets a test in the same commit", which project_manager's testing standard replaced (no session
runs tests before a push or a release; a fixed bug that reached another repo or a person gets one test of that
behavior, a feature or an internal fix gets none). The old "# CLAUDE.md" title and its Claude Code guidance line gave
way to the repo's name; CLAUDE.md stays a link to AGENTS.md.

## Purpose
- The LLM-provider utility layer was extracted from the owner's research repos into this base repo on 2026-07-23, on
  the owner's order.
- The SLURM serving lifecycle moved out of this package to the owner's device-management layer in v6.0.0.

## Working here: dependency protocol
- Shared-private repos may carry this dep: owner ruling 2026-07-30, superseding the 2026-07-24 exclusion of
  shared-private repos. Anonymous paper mirrors kept the vendoring rule.

## Working here: code and tests
- Site-specific cluster/account facts and consumer records in `text_docs/private/`, never in committed files: owner
  order 2026-07-24.

## Tasks
- There is no TODO.md: the generated task views were retired 2026-10-14 (psyche t920).
