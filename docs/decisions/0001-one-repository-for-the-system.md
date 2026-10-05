# 0001 The research system stays one repository

**Context.** The research paths (agent, baseline), extraction, corpus, API and UI change together through
one contract, the `Run` schema (`harness_schema`). The import layers already keep the packages apart
(`lint-imports`, AGENTS.md "Layers").

**Decision.** Keep them as packages in this repository; do not split them into repositories.

**Consequences.** A change to the run contract stays one commit. Coding agents work package by package
(0006), not repository by repository.
