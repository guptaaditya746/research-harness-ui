# Few-shot examples

One JSON file per prompt and set, e.g. `v2/extraction/edc_extract.json`:

```json
[
  {"input": "passage: \"… improves success rate by 41.5% on CALVIN ABC-D …\"",
   "output": "(paper, reportsMetric, \"41.5% success-rate improvement on CALVIN ABC-D\")",
   "why": "keep the benchmark with the number; a result without its benchmark was scored as missed"}
]
```

- Only v2 templates show examples (`extraction/edc_extract`, `extraction/evidence_review`,
  `extraction/answer_composition`, `agent/paper_reader`, `agent/research_supervisor`).
- Examples change the prompt's hash, so runs record which examples they used.
- Take them from failures on a development part of the gold set (`make fewshot` lists candidates),
  never from the questions you report results on; see docs/PROMPTS.md.
