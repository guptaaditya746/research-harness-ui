# 0004 Scoring stays in this repository for now

**Context.** The Benchmark page and `make evaluate` call `harness_evaluation` directly. Moving it to labelling
would make this repository depend on labelling while labelling depends on this one.

**Decision.** Keep `harness_evaluation` here. Revisit when scoring against evidence documents is built; that
scorer may live in labelling and write results the Benchmark page reads.

**Consequences.** No circular dependency between the repositories.
