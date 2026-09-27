"""Rebuild seed/cskg_ontology_2.0_subset.ttl from the full CS-KG 2.0 ontology.

    make cskg-subset                                                     # from seed/cskg_ontology_2.0.ttl
    make cskg-subset ARGS="--check"                                      # compare, write nothing
    make cskg-subset ARGS="--source other.ttl"                           # another copy (file or URL)

The full ontology is kept unchanged in seed/cskg_ontology_2.0.ttl (downloaded from
https://scholkg.kmi.open.ac.uk/cskg/ontology). The subset holds the terms the harness names
(harness_semantic.CSKG_CLASSES and CSKG_RELATIONS: the research-entity classes, the base relations
and their inverses) with every statement the source makes about them, blank nodes included (e.g.
an owl:unionOf domain). Commit the rebuilt subset: runs record its SHA-256 (seed_manifest.json
content_core_sha256). CS-KG is CC BY 4.0; the header keeps the attribution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated

import typer
from harness_semantic import CSKG, CSKG_CLASSES, CSKG_RELATIONS, cskg_subset_path
from rdflib import DCTERMS, OWL, RDF, RDFS, BNode, Graph, Literal, URIRef

ROOT = Path(__file__).resolve().parents[1]
FULL_ONTOLOGY = ROOT / "seed" / "cskg_ontology_2.0.ttl"
HEADER = """# Subset of the Computer Science Knowledge Graph (CS-KG) Ontology 2.0: the harness content core.
#
# Source:  https://scholkg.kmi.open.ac.uk/cskg/ontology  (namespace https://w3id.org/cskg/ontology#)
#          read from {source}, SHA-256 {digest}
# Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), (c) the CS-KG authors,
#          Knowledge Media Institute, The Open University. Terms and statements are copied unchanged.
#
# Written by scripts/cskg_subset.py (make cskg-subset): every statement the source makes about the
# research-entity classes and the base relations the harness names (harness_semantic), and their
# inverses. The harness names these terms through its own vocabulary (hrn:Method owl:equivalentClass
# cskg-ont:Method, hrn:uses owl:equivalentProperty cskg-ont:uses).

"""


def wanted_terms() -> list[URIRef]:
    terms = [CSKG.ResearchEntity, *CSKG_CLASSES.values()]
    for official, inverse, _range, _definition in CSKG_RELATIONS.values():
        terms += [official, inverse]
    return list(dict.fromkeys(terms))


def subset(source: Graph) -> tuple[Graph, list[URIRef]]:
    """The statements about the wanted terms (and the blank nodes they use), and the terms not found."""
    out = Graph()
    out.bind("owl", OWL)
    out.bind("cskg-ont", CSKG)
    out.bind("dct", DCTERMS)
    missing = []
    for term in wanted_terms():
        if not any(source.triples((term, RDF.type, None))):
            missing.append(term)
            continue
        todo: list[object] = [term]
        seen: set[object] = set()
        while todo:
            node = todo.pop()
            if node in seen:
                continue
            seen.add(node)
            for predicate, obj in source.predicate_objects(node):
                out.add((node, predicate, obj))
                if isinstance(obj, BNode):
                    todo.append(obj)
    header = URIRef("https://harness.local/ontology/cskg-2.0-subset")
    out.add((header, RDF.type, OWL.Ontology))
    out.add((header, RDFS.label, Literal("CS-KG Ontology 2.0, harness subset")))
    out.add((header, DCTERMS.source, URIRef(str(CSKG))))
    out.add((header, DCTERMS.license, URIRef("https://creativecommons.org/licenses/by/4.0/")))
    # CS-KG declares its ontology as <https://w3id.org/cskg/ontology#>; take the version from whichever
    # owl:Ontology the source declares.
    version = next((value for ontology in source.subjects(RDF.type, OWL.Ontology)
                    for value in source.objects(ontology, OWL.versionInfo)), None)
    if version is not None:
        out.add((header, OWL.versionInfo, version))
    return out, missing


def read_source(source: str) -> bytes:
    if source.startswith(("http://", "https://")):
        import httpx

        response = httpx.get(source, headers={"Accept": "text/turtle"}, follow_redirects=True, timeout=60)
        response.raise_for_status()
        return response.content
    return Path(source).expanduser().read_bytes()


def main(
    source: Annotated[str, typer.Option(help="The full CS-KG ontology: a Turtle file or its URL.")] = "",
    out: Annotated[Path | None, typer.Option(help="Target file (default: the subset the harness loads).")] = None,
    check: Annotated[bool, typer.Option(help="Only compare with the current subset; write nothing.")] = False,
) -> None:
    source = source or str(FULL_ONTOLOGY.relative_to(ROOT))
    data = read_source(source if source.startswith(("http://", "https://")) or Path(source).is_absolute()
                       else str(ROOT / source))
    graph = Graph()
    graph.parse(data=data, format="turtle")
    chosen, missing = subset(graph)
    if missing:
        typer.echo("not in the source: " + ", ".join(str(term).rsplit("#", 1)[-1] for term in missing), err=True)
        raise typer.Exit(1)
    target = out or cskg_subset_path()
    text = HEADER.format(source=source, digest=hashlib.sha256(data).hexdigest()) + chosen.serialize(format="turtle")
    if check:
        from rdflib.compare import graph_diff, to_isomorphic

        current = Graph().parse(target, format="turtle") if target.exists() else Graph()
        # Canonical blank-node labels, so an unchanged owl:unionOf domain is not reported as a change.
        _, removed, added = graph_diff(to_isomorphic(current), to_isomorphic(chosen))
        typer.echo(f"{target}: {len(added)} statements would be added, {len(removed)} removed")
        for label, rows in (("+", added), ("-", removed)):
            for s, p, o in sorted(rows, key=str)[:40]:
                typer.echo(f"  {label} {s.n3(graph.namespace_manager)} {p.n3(graph.namespace_manager)} "
                           f"{o.n3(graph.namespace_manager)}")
        return
    target.write_text(text, encoding="utf-8")
    typer.echo(f"{target}: {len(chosen)} triples for {len(wanted_terms())} terms")


if __name__ == "__main__":
    typer.run(main)
