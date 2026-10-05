# 0006 Each package has its own short guide for coding agents

**Context.** An agent changing extraction should not need to read the agent loop.

**Decision.** Every package has an `AGENTS.md`: what it owns, its entry points, what it may import, the one
command that tests it. The root `AGENTS.md` keeps the repository-wide rules and the map.

**Consequences.** A coding session can be pointed at one folder.
