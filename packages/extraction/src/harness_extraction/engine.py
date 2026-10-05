"""The extraction engine: schema inference, triple extraction, review and answers.

One interface shared by the baseline paths and the agentic path. Every model
call renders a named prompt from ``harness_prompts`` (see the prompt name at
each call) and goes through the ``ModelClient`` protocol. The stages are mixins:
``stage_schema``, ``stage_extract``, ``stage_review``, ``stage_answer``; this
module holds the engine's schema (registration of run-local terms) and signatures.
"""

from __future__ import annotations

import re
import threading
from dataclasses import replace

from harness_semantic import (
    DBLP,
    HRN,
    TOP_ATTRIBUTE,
    TOP_CLASS,
    TOP_RELATION,
    Subschema,
    is_datatype_range,
    property_kind,
)
from rdflib import OWL, RDF, RDFS, XSD, Literal, URIRef

from .citations import citation_ids, drop_unknown_citations
from .model_client import ModelClient
from .naming import _literal_value, local_name, prompt_shape, relation_name
from .schemas import SchemaInference
from .stage_answer import AnswerStage
from .stage_extract import ExtractionStage
from .stage_review import ReviewStage
from .stage_schema import SchemaStage

# DBLP properties offered as parents of invented properties (e.g. firstAuthor under dblp:authoredBy).
DBLP_PARENTS = ("authoredBy", "createdBy", "publishedIn", "publishedInStream", "publishedAsPartOf",
                "isVersionOf", "yearOfPublication", "title", "affiliation")



class ExtractionEngine(SchemaStage, ExtractionStage, ReviewStage, AnswerStage):
    """One interface shared by baseline and the later agentic path."""

    def __init__(
        self,
        model: ModelClient,
        schema: Subschema,
        schema_model: ModelClient | None = None,
        edc_model: ModelClient | None = None,
    ) -> None:
        self.model = model
        self.schema_model = schema_model or model
        self.edc_model = edc_model or model
        self.schema = schema
        self.class_names = {local_name(iri): iri for iri in schema.classes}
        self.property_names = {local_name(iri): iri for iri in schema.properties}
        # The generic parent class is usable as a class (and is where "off" mode maps new classes).
        self.class_names.setdefault(local_name(TOP_CLASS), TOP_CLASS)
        self._local_schema = False
        # Extraction batches run in parallel and may each add a relation to the schema.
        self._schema_lock = threading.RLock()
        # Relations added for open facts that mapped onto no subschema term (HARNESS_UNMAPPED_RELATIONS=keep):
        # name -> the open wording it came from. The pipeline adds them to the run's subschema.
        self.kept_relations: dict[str, str] = {}

    def parent_properties(self) -> dict[str, URIRef]:
        """Properties an invented property may specialise: the harness terms, the two generic parents
        and a few DBLP properties (prefixed ``dblp:``)."""
        parents = {**self.property_names, local_name(TOP_RELATION): TOP_RELATION,
                   local_name(TOP_ATTRIBUTE): TOP_ATTRIBUTE}
        for name in DBLP_PARENTS:
            iri = URIRef(DBLP[name])
            if property_kind(self.schema.graph, iri) is not None:
                parents[f"dblp:{name}"] = iri
        return parents

    def _resolve_parent_property(self, parent: str | None, kind: str, own: URIRef) -> URIRef:
        """The named parent when it exists and has the same kind (object/datatype), else the generic one."""
        fallback = TOP_ATTRIBUTE if kind == "datatype" else TOP_RELATION
        # Accept the name as listed ("dblp:authoredBy (object)") or with an hrn: prefix.
        name = re.sub(r"\s*\((object|datatype)\)\s*$", "", str(parent or "").strip()).removeprefix("hrn:")
        iri = self.parent_properties().get(name)
        if iri is None or iri == own or property_kind(self.schema.graph, iri) != kind:
            return fallback
        return iri

    def _resolve_parent_class(self, parent: str | None, own: URIRef) -> URIRef:
        name = str(parent or "").strip()
        iri = self.class_names.get(name)
        if iri is None and name.startswith("dblp:"):
            candidate = URIRef(DBLP[name.removeprefix("dblp:")])
            iri = candidate if (candidate, RDF.type, OWL.Class) in self.schema.graph else None
        return TOP_CLASS if iri is None or iri == own else iri

    def _ensure_local_schema(self) -> None:
        if self._local_schema:
            return
        graph = self.schema.graph.__class__()
        for prefix, namespace in self.schema.graph.namespaces():
            graph.bind(prefix, namespace)
        for triple in self.schema.graph:
            graph.add(triple)
        self.schema = replace(self.schema, graph=graph)
        self._local_schema = True

    def register_local_class(self, name: str, definition: str, parent: str | None = None) -> URIRef:
        """Introduce one query-local OWL 2 class, a subclass of an existing class, without changing the seed."""
        if not re.fullmatch(r"[A-Z][A-Za-z0-9_]{0,63}", name):
            raise ValueError(f"invalid proposed class name: {name!r}")
        with self._schema_lock:
            return self._register_class(name, definition, parent)

    def _register_class(self, name: str, definition: str, parent: str | None) -> URIRef:
        if name in self.class_names:
            return self.class_names[name]
        self._ensure_local_schema()
        iri = URIRef(HRN[name])
        self.schema.graph.add((iri, RDF.type, OWL.Class))
        self.schema.graph.add((iri, RDFS.label, Literal(name)))
        self.schema.graph.add((iri, RDFS.comment, Literal(definition)))
        self.schema.graph.add((iri, RDFS.subClassOf, self._resolve_parent_class(parent, iri)))
        self.schema = replace(self.schema, classes=self.schema.classes | {iri})
        self.class_names[name] = iri
        return iri

    def register_local_relation(
        self, name: str, definition: str, *, domain_name: str = "Paper",
        range_name: str = "xsd:string", parent: str | None = None,
    ) -> URIRef:
        """Add a query-local OWL 2 property: object or datatype by its range, with one domain, one
        range and a parent property (the named one when it fits, else the generic one)."""
        if not re.fullmatch(r"[a-z][A-Za-z0-9_]{0,63}", name):
            raise ValueError(f"invalid proposed relation name: {name!r}")
        with self._schema_lock:
            return self._register_relation(name, definition, domain_name, range_name, parent)

    def _register_relation(self, name: str, definition: str, domain_name: str, range_name: str,
                           parent: str | None) -> URIRef:
        iri = URIRef(HRN[name])
        if iri in self.schema.properties:
            return iri
        if property_kind(self.schema.graph, iri) is not None:
            # Already declared (e.g. a generic parent): never declare a property twice.
            return iri
        self._ensure_local_schema()
        graph = self.schema.graph
        range_ = self.class_names.get(range_name, XSD.string)
        kind = "datatype" if is_datatype_range(range_) else "object"
        graph.add((iri, RDF.type, OWL.DatatypeProperty if kind == "datatype" else OWL.ObjectProperty))
        graph.add((iri, RDFS.label, Literal(name)))
        graph.add((iri, RDFS.comment, Literal(definition)))
        graph.add((iri, RDFS.domain, self.class_names.get(domain_name, HRN.Paper)))
        graph.add((iri, RDFS.range, range_))
        graph.add((iri, RDFS.subPropertyOf, self._resolve_parent_property(parent, kind, iri)))
        self.schema = replace(self.schema, properties=self.schema.properties | {iri})
        self.property_names[name] = iri
        return iri

    def signatures(self, names: list[str]) -> list[str]:
        """"name: Domain -> Range. definition" for each property (v2 prompts), so the models see which
        properties are about a paper and which link two research entities (the CS-KG relations)."""
        rows = []
        for name in names:
            iri = self.property_names.get(name)
            if iri is None:
                continue
            graph = self.schema.graph
            domain, range_ = graph.value(iri, RDFS.domain), graph.value(iri, RDFS.range)
            if not isinstance(domain, URIRef) or not isinstance(range_, URIRef):
                continue
            range_name = f"xsd:{local_name(range_)}" if str(range_).startswith(str(XSD)) else local_name(range_)
            comment = " ".join(str(graph.value(iri, RDFS.comment) or "").split())
            rows.append(f"{name}: {local_name(domain)} -> {range_name}" + (f". {comment}" if comment else ""))
        return rows

    def axioms(self, inferred: SchemaInference) -> list[tuple[str, str, str]]:
        rows = []
        for name in inferred.properties:
            iri = self.property_names[name]
            domain = self.schema.graph.value(iri, RDFS.domain)
            range_ = self.schema.graph.value(iri, RDFS.range)
            if isinstance(domain, URIRef) and isinstance(range_, URIRef):
                range_name = (
                    f"xsd:{local_name(range_)}"
                    if str(range_).startswith("http://www.w3.org/2001/XMLSchema#")
                    else local_name(range_)
                )
                rows.append((name, local_name(domain), range_name))
        return rows


__all__ = ["ExtractionEngine", "_literal_value", "citation_ids", "drop_unknown_citations", "local_name",
           "prompt_shape", "relation_name"]
