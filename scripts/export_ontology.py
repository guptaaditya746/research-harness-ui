"""Write the harness's seed ontology (DBLP schema + harness core terms and generic parents + the CS-KG
content core) as Turtle.

    make ontology                                    # artifacts/ontology/harness_seed.ttl (or $ARTIFACT_ROOT/...)
    make ontology ARGS="--out /srv/ontology.ttl"

The external harness reads this file (its config's knowledge_base.ontology), so the External path
sees the same vocabulary every harness run starts from. Run-local invented terms are not in it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from harness_config import settings
from harness_semantic import hand_written_subschema

ROOT = Path(__file__).resolve().parents[1]


def main(out: Annotated[Path | None, typer.Option(help="Target .ttl file.")] = None) -> None:
    target = out or Path(settings().artifact_root or ROOT / "artifacts") / "ontology" / "harness_seed.ttl"
    target.parent.mkdir(parents=True, exist_ok=True)
    schema = hand_written_subschema()
    target.write_text(schema.graph.serialize(format="turtle"), encoding="utf-8")
    typer.echo(f"{target}: {len(schema.graph)} triples, {len(schema.classes)} classes, "
               f"{len(schema.properties)} properties (seed {schema.seed_schema_version}, "
               f"content core {schema.content_core})")


if __name__ == "__main__":
    typer.run(main)
