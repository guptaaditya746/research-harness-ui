# baseline

Fast and Adaptive research paths. A fixed LangGraph workflow (retrieve → infer_schema → extract_build → query_answer), after question understanding answers fact operations from verified DBLP facts.

## Files

- `pipeline.py` — `Baseline20.run` and the graph
- `retrieval.py`, `research_steps.py`, `facts.py`, `outcomes.py` — the steps (mixins)
- `planner.py` — `QueryPlanner`, `PaperResolver`, operations and measures
- `answer_graph.py` — the connected answer graph for the UI
- `state.py` — pipeline state and early outcomes

## Depends on

harness_extraction, harness_corpus, harness_graphstore, harness_semantic, harness_prompts.

See `AGENTS.md` for the repository map and conventions.
