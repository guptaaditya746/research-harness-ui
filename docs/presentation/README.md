# Thesis pipeline map

`thesis_pipeline.html` is a one-page visual overview of the whole thesis for presentations: the corpus, the
research harness, evaluation and the gold standard (evidence document D), which stages call an LLM, which
stage is being worked on now, and which paper, tool or dataset each idea comes from. It is a single file
with no dependencies and no network access: the browser draws it from the data at the top of its script.

## Open it

On your own computer (after `git pull`), open the file in any browser:

```sh
xdg-open docs/presentation/thesis_pipeline.html     # Linux
open docs/presentation/thesis_pipeline.html         # macOS
start docs\presentation\thesis_pipeline.html        # Windows
```

On the cluster (no browser there), serve it and tunnel the port:

```sh
make pipeline-map                                   # serves docs/presentation on port 8099
ssh -L 8099:localhost:8099 <user>@<cluster-host>    # on your laptop, in a second terminal
# then open http://localhost:8099/thesis_pipeline.html
```

`make pipeline-map PORT=9000` uses another port. Press F11 in the browser for full screen when presenting.

## Change it

Everything shown comes from the blocks at the top of the `<script>`:

- `STAGES`: one line per box (`name`, `desc`, `lane`, `col` for the position, `llm: true` for the LLM badge,
  `people: true` for human work, `current: true` for the "currently working here" marker, `sources` and `ours`
  for the chips under the box).
- `SOURCES`: the papers, tools and datasets (chip text, full reference); the reference list under the diagram
  is built from it.
- `LANES`: the rows, top to bottom. `LINKS`: the arrows between rows.
- `LLM_ICON_SRC`: a `data:image/...` URI of the Qwen logo replaces the generic LLM icon.

Save and reload the page; there is nothing to build.
