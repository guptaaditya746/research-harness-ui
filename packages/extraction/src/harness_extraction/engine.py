"""The extraction engine: schema inference, triple extraction, review and answers.

One interface shared by the baseline paths and the agentic path. Every model
call renders a named prompt from ``harness_prompts`` (see the prompt name at
each call) and goes through the ``ModelClient`` protocol.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any

from harness_config import settings
from harness_prompts import render
from harness_semantic import (
    DBLP,
    HRN,
    TOP_ATTRIBUTE,
    TOP_CLASS,
    TOP_RELATION,
    Subschema,
    content_core_on,
    is_datatype_range,
    property_kind,
)
from rdflib import OWL, RDF, RDFS, XSD, Literal, URIRef

from .call_control import parallel_map
from .errors import ModelFailure, UnsupportedAnswer
from .events import emit_model_event, labelled
from .model_client import ModelClient, complete
from .schemas import (
    AnswerResult,
    EDCDefinition,
    EDCDefinitionResult,
    EDCOpenTriple,
    ExtractedTriple,
    ExtractionResult,
    SchemaInference,
)


def local_name(value: URIRef) -> str:
    return str(value).rsplit("#", 1)[-1]


# DBLP properties offered as parents of invented properties (e.g. firstAuthor under dblp:authoredBy).
DBLP_PARENTS = ("authoredBy", "createdBy", "publishedIn", "publishedInStream", "publishedAsPartOf",
                "isVersionOf", "yearOfPublication", "title", "affiliation")


# Source ids are lowercase slugs with a letter and a digit ("p4-abstract", "p12-s3-p2", "2501-01234-metadata");
# research_notes is part of the composer's input but not a source. Other brackets are prose ("[0.5-0.7]",
# "[CLS]", "[GPT-4]", "[h_t]").
_ID_LIKE = re.compile(r"(?=[a-z0-9_-]*[a-z])(?=[a-z0-9_-]*\d)[a-z0-9]+(?:[-_][a-z0-9]+)*")
_NOT_SOURCES = {"research_notes"}
# A sentence ends at . ! or ? followed by a space, but not after "et al.", "e.g.", "i.e.", "vs.", "Fig." and the like.
_SENTENCE_END = re.compile(r"(?<!\bet al\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<!\bFig\.)(?<!\bSec\.)"
                           r"(?<!\bEq\.)(?<!\bNo\.)(?<!\bU\.S\.)(?<=[.!?])\s+")
_BRACKET = re.compile(r"\[([^\]]+)\]")


def _drop_reason(value: object) -> str:
    """Evidence review's reason for dropping a triple, as one of the prompt's words."""
    text = str(value or "").strip().casefold()
    return next((word for word in ("unsupported", "irrelevant", "malformed") if word in text),
                "no reason given" if not text else "other")


def _looks_like_source(value: str) -> bool:
    return value in _NOT_SOURCES or bool(_ID_LIKE.fullmatch(value))


def citation_ids(group: str, sources: set[str]) -> list[str] | None:
    """The ids of one bracket group, or None when the group is prose. A group is a citation when one of
    its parts is a known source ("[p4-abstract, Table 2]" cites p4-abstract) or every part looks like a
    source id ("[research_notes]", "[p9-abstract]")."""
    parts = [part.strip() for part in re.split(r"[,;]", group) if part.strip()]
    if any(part in sources for part in parts):
        return [part for part in parts if part in sources or _looks_like_source(part)]
    return parts if parts and all(_looks_like_source(part) for part in parts) else None


def drop_unknown_citations(answer: str, cited: list[str], sources: set[str]) -> tuple[str, list[str], list[str], int]:
    """The answer without citations that are no triple's source.

    A citation after the full stop ("A holds. [p4-abstract]") belongs to the sentence before it. A sentence
    with a citation that names no known source is dropped, even when it also cites a real one (its other
    claim has no evidence); an unknown id next to a known one in the same brackets is removed. Sentences
    without citations stay. Returns the text, the valid cited ids (listed or in brackets), every unknown id
    and the number of sentences dropped.
    """
    def attach(match: re.Match[str]) -> str:
        return (f" [{match.group(2)}]{match.group(1)}" if citation_ids(match.group(2), sources) is not None
                else match.group(0))

    def known_only(match: re.Match[str]) -> str:
        ids = citation_ids(match.group(1), sources)
        return match.group(0) if ids is None else f" [{', '.join(item for item in ids if item in sources)}]"

    listed = [str(item).strip().strip("[]").strip() for item in cited]
    valid = [item for item in listed if item in sources]
    unknown = [item for item in listed if item and item not in sources]
    text = re.sub(r"([.!?])\s+\[([^\]]+)\](?=\s|$)", attach, str(answer).strip())
    kept: list[str] = []
    dropped = 0
    for sentence in _SENTENCE_END.split(text):
        groups = [ids for ids in (citation_ids(group, sources) for group in _BRACKET.findall(sentence)) if ids is not None]
        unknown.extend(item for ids in groups for item in ids if item not in sources)
        if any(not any(item in sources for item in ids) for ids in groups):
            dropped += 1
            continue
        valid.extend(item for ids in groups for item in ids if item in sources)
        kept.append(re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\s*\[([^\]]+)\]", known_only, sentence)) if groups
                    else sentence)
    return " ".join(kept).strip(), list(dict.fromkeys(valid)), list(dict.fromkeys(unknown)), dropped


class ExtractionEngine:
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

    @labelled("schema inference")
    def infer_schema(self, question: str, passages: list[dict[str, Any]],
                     notes: list[str] | None = None) -> SchemaInference:
        """``notes``: the agent's unverified findings (agentic runs), sent as their own field so they
        never become part of a definition or the question text."""
        allowed_classes = sorted(self.class_names)
        allowed_properties = sorted(self.property_names)
        parents = self.parent_properties()
        payload: dict[str, Any] = {
            "question": question,
            "allowed_classes": allowed_classes,
            "allowed_properties": allowed_properties,
            "parent_properties": [f"{name} ({property_kind(self.schema.graph, iri) or 'object'})"
                                  for name, iri in sorted(parents.items())],
            "passages": [
                {"source_id": row["passage_id"], "text": row["text"][:1200]}
                for row in passages
            ],
        }
        core = content_core_on()
        if core and settings().prompt_set != "v1":
            payload["property_signatures"] = self.signatures(allowed_properties)
        if notes:
            payload["research_notes"] = list(notes)
        result = SchemaInference.model_validate(complete(self.schema_model, render(
            "extraction/schema_inference", payload, content_core=core,
        )))
        if not result.classes or not result.properties:
            raise ValueError("schema inference returned no classes or properties")
        if settings().schema_new_classes == "off":
            result = self._without_new_classes(result)
        # The generic parents are abstract: they are parents, not predicates to extract.
        abstract = {local_name(TOP_RELATION), local_name(TOP_ATTRIBUTE)}
        if abstract & set(result.properties):
            kept = [name for name in result.properties if name not in abstract]
            if not kept:
                raise ValueError("schema inference returned only abstract parent properties")
            result = result.model_copy(update={"properties": kept})
        class_defs = {
            item["name"]: item for item in result.class_definitions
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        property_defs = {
            item["name"]: item for item in result.property_definitions
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        for name in result.classes:
            if name not in self.class_names:
                definition = str(class_defs.get(name, {}).get("definition") or
                                 f"Run-local research concept {name} for: {question}")
                self.register_local_class(name, definition, parent=class_defs.get(name, {}).get("parent"))
        for name in result.properties:
            if name in self.property_names:
                continue
            spec = property_defs.get(name, {})
            range_name = str(spec.get("range") or "")
            if range_name not in self.class_names:
                range_name = next(
                    (class_name for class_name in sorted(result.classes, key=len, reverse=True)
                     if name.casefold().endswith(class_name.casefold())),
                    "xsd:string",
                )
            definition = str(spec.get("definition") or
                             f"Run-local research relation {name} for: {question}")
            self.register_local_relation(
                name, definition, domain_name=str(spec.get("domain") or "Paper"),
                range_name=range_name, parent=spec.get("parent"),
            )
        new_classes = set(result.classes) - set(allowed_classes)
        new_properties = set(result.properties) - set(allowed_properties)
        if new_classes or new_properties:
            emit_model_event(
                "Schema extension proposed",
                f"classes: {', '.join(sorted(new_classes)) or 'none'}; "
                f"properties: {', '.join(sorted(new_properties)) or 'none'}",
            )
        return result

    def _without_new_classes(self, result: SchemaInference) -> SchemaInference:
        """HARNESS_SCHEMA_NEW_CLASSES=off: an invented class becomes the generic ResearchEntity."""
        generic = local_name(TOP_CLASS)
        rename = {name: generic for name in result.classes if name not in self.class_names}
        if not rename:
            return result
        properties = [
            {**item, **{key: rename.get(str(item.get(key)), item.get(key)) for key in ("domain", "range")
                        if key in item}}
            for item in result.property_definitions if isinstance(item, dict)
        ]
        return result.model_copy(update={
            "classes": list(dict.fromkeys(rename.get(name, name) for name in result.classes)),
            "class_definitions": [],
            "property_definitions": properties,
        })

    @labelled("triple extraction")
    def extract(
        self, question: str, inferred: SchemaInference, passages: list[dict[str, Any]]
    ) -> ExtractionResult:
        core = content_core_on()
        signatures = ({"property_signatures": self.signatures(list(inferred.properties))}
                      if core and settings().prompt_set != "v1" else {})
        result = ExtractionResult.model_validate(complete(self.model, render(
            "extraction/triple_extraction",
            {
                "question": question,
                "classes": inferred.classes,
                "properties": inferred.properties,
                **signatures,
                "passages": [
                    {
                        "passage_id": row["passage_id"],
                        "paper_id": row["paper_id"],
                        "text": row["text"][:1800],
                    }
                    for row in passages
                ],
            }, content_core=core,
        )))
        passage_ids = {row["passage_id"] for row in passages}
        # Models sometimes return a useful supported triple alongside a
        # natural-language relation or an invalid passage id. Keep the valid
        # evidence instead of rejecting the whole extraction response.
        result.triples = [
            triple
            for triple in result.triples
            if triple.predicate in inferred.properties and triple.source_id in passage_ids
        ]
        return result

    @labelled("open extraction and relation mapping")
    def extract_edc(
        self, question: str, inferred: SchemaInference, passages: list[dict[str, Any]]
    ) -> tuple[ExtractionResult, list[dict[str, Any]]]:
        """Run EDC-style Extract -> Define -> Canonicalize over passages.

        Extraction keeps predicates open and ties every proposition to evidence.
        Canonicalization then maps only high-confidence open predicates to the
        inferred ontology; unsupported relations remain unresolved instead of
        being forced into a schema property.
        """
        allowed_ids = {row["passage_id"] for row in passages}
        paper_by_source = {str(row["passage_id"]): str(row["paper_id"]) for row in passages}
        payload = complete(self.edc_model, render("extraction/edc_extract", {
            "question": question,
            "passages": [{"source_id": row["passage_id"], "paper_id": row["paper_id"],
                          "text": row["text"][:1800]} for row in passages],
        }))
        raw = payload.get("triples", []) if isinstance(payload, dict) else []
        open_triples: list[EDCOpenTriple] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source_id") or "").strip() or None
            ids = [str(x) for x in item.get("evidence_ids", []) if str(x) in allowed_ids]
            if source in allowed_ids and source not in ids:
                ids.append(source)
            if not ids:
                continue
            try:
                open_triples.append(EDCOpenTriple.model_validate({
                    **item,
                    "source_id": source,
                    "evidence_ids": ids,
                    "evidence": item.get("evidence") or item.get("evidence_span") or "",
                }))
            except (TypeError, ValueError):
                continue

        relations = list(inferred.properties)
        definitions: list[EDCDefinition] = []
        distinct = sorted({item.predicate.strip() for item in open_triples if item.predicate.strip()})
        if distinct:
            defined = complete(self.edc_model, render("extraction/edc_define", {"relations": distinct}))
            try:
                definitions = EDCDefinitionResult.model_validate(defined).definitions
            except (TypeError, ValueError):
                definitions = []
        definition_map = {self._norm(item.relation): item.definition for item in definitions}
        selected_relations = [
            self._canonical_property(
                item.predicate, definition_map.get(self._norm(item.predicate), ""), relations
            )
            for item in open_triples
        ]
        unresolved = [
            {"id": index, "subject": item.subject, "predicate": item.predicate,
             "object": item.object, "evidence": item.evidence}
            for index, (item, selection) in enumerate(zip(open_triples, selected_relations, strict=True))
            if selection[0] is None
        ]
        llm_mapping: dict[int, tuple[str, float]] = {}
        if unresolved and relations:
            property_catalog = []
            for relation in relations:
                iri = self.property_names.get(relation)
                domain = self.schema.graph.value(iri, RDFS.domain) if iri else None
                range_ = self.schema.graph.value(iri, RDFS.range) if iri else None
                entry = {
                    "property": relation,
                    "domain": local_name(domain) if isinstance(domain, URIRef) else None,
                    "range": local_name(range_) if isinstance(range_, URIRef) else None,
                }
                if content_core_on() and settings().prompt_set != "v1" and iri is not None:
                    # With the content core, the model tells paper-level from entity-level relations
                    # (addresses vs solves) by their definitions too.
                    entry["definition"] = " ".join(str(self.schema.graph.value(iri, RDFS.comment) or "").split())
                property_catalog.append(entry)
            mapped = complete(self.edc_model, render(
                "extraction/edc_map",
                {"question": question, "unresolved": unresolved,
                 "allowed_properties": property_catalog},
                ensure_ascii=False,
            ))
            for row in mapped.get("mappings", []) if isinstance(mapped, dict) else []:
                if not isinstance(row, dict):
                    continue
                index = row.get("id")
                property_name = row.get("property")
                try:
                    confidence = float(row.get("confidence", 0))
                except (TypeError, ValueError):
                    continue
                if type(index) is int and 0 <= index < len(open_triples) and property_name in relations and confidence >= 0.7:
                    llm_mapping[index] = (property_name, confidence)

        canonical: list[ExtractedTriple] = []
        decisions: list[dict[str, Any]] = []
        for index, item in enumerate(open_triples):
            selected, score = selected_relations[index]
            method = "lexical"
            if selected is None and index in llm_mapping:
                selected, score = llm_mapping[index]
                method = "ontology_review"
            decisions.append({"open": item.predicate, "selected": selected, "score": round(score, 3), "method": method})
            if selected is None:
                continue
            property_iri = self.property_names.get(selected)
            domain = self.schema.graph.value(property_iri, RDFS.domain) if property_iri else None
            subject = (
                paper_by_source.get(item.evidence_ids[0], item.subject)
                if domain == HRN.Paper
                else item.subject
            )
            canonical.append(ExtractedTriple(
                subject=subject,
                predicate=selected,
                object=item.object,
                source_id=item.evidence_ids[0],
                evidence=item.evidence or f"Evidence from {item.evidence_ids[0]}.",
            ))
        return ExtractionResult(triples=canonical), [
            {"stage": "extract", "open_triples": len(open_triples)},
            {"stage": "define", "definitions": len(definitions)},
            {"stage": "ontology_review", "unresolved": len(unresolved), "mapped": len(llm_mapping)},
            {"stage": "canonicalize", "decisions": decisions},
        ]

    @staticmethod
    def _norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    @classmethod
    def _canonical_property(cls, predicate: str, definition: str, relations: list[str]) -> tuple[str | None, float]:
        source = set(cls._norm(f"{predicate} {definition}").split())
        ranked = []
        for relation in relations:
            target = set(cls._norm(relation).split())
            overlap = len(source & target) / max(1, len(source | target))
            ratio = SequenceMatcher(None, cls._norm(predicate), cls._norm(relation)).ratio()
            ranked.append((0.55 * ratio + 0.45 * overlap, relation))
        if not ranked:
            return None, 0.0
        score, relation = max(ranked)
        return (relation, score) if score >= 0.42 else (None, score)

    def extract_batches(
        self,
        question: str,
        inferred: SchemaInference,
        passages: list[dict[str, Any]],
        batch_size: int = 4,
        text_limit: int = 2500,
    ) -> tuple[list[ExtractedTriple], list[dict[str, Any]]]:
        """Extract from bounded passage batches and deduplicate with provenance."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        merged: dict[tuple[str, str, str], ExtractedTriple] = {}
        reports: list[dict[str, Any]] = []
        batches = [passages[start : start + batch_size] for start in range(0, len(passages), batch_size)]

        def run_batch(batch: list[dict[str, Any]]) -> ExtractionResult | ValueError:
            try:
                return self.extract(question, inferred, [{**row, "text": row["text"][:text_limit]} for row in batch])
            except ValueError as exc:
                return exc

        outcomes = parallel_map(run_batch, batches, settings().extraction_concurrency)
        for batch_no, (batch, result) in enumerate(zip(batches, outcomes, strict=True), start=1):
            if isinstance(result, ValueError):
                reports.append({"batch": batch_no, "passages": [p["passage_id"] for p in batch], "error": str(result)})
                continue
            reports.append({"batch": batch_no, "passages": [p["passage_id"] for p in batch], "triples": len(result.triples)})
            for triple in result.triples:
                key = (triple.subject.casefold().strip(), triple.predicate, triple.object.casefold().strip())
                previous = merged.get(key)
                if previous is None:
                    merged[key] = triple
                elif triple.source_id != previous.source_id:
                    merged[key] = previous.model_copy(update={"evidence": f"{previous.evidence} | {triple.evidence}"})
        return list(merged.values()), reports

    @labelled("evidence review")
    def refine_triples(
        self,
        question: str,
        triples: list[ExtractedTriple],
        passages: list[dict[str, Any]],
        batch_size: int = 24,
        reviewer: ModelClient | None = None,
    ) -> tuple[list[ExtractedTriple], dict[str, Any]]:
        """Second model pass: retain only atomic claims relevant to the question.

        The reviewer receives numbered existing candidates and may only return
        their IDs. It cannot add a fact or change provenance. Batches cover the
        entire candidate set, so adaptive retrieval has no output cap here.
        """
        passage_text = {str(row["passage_id"]): str(row["text"]) for row in passages}
        deduped: dict[tuple[str, str, str], ExtractedTriple] = {}
        for item in triples:
            if not all((item.subject.strip(), item.predicate.strip(), item.object.strip(), item.source_id.strip())):
                continue
            key = (item.subject.casefold().strip(), item.predicate, item.object.casefold().strip())
            deduped.setdefault(key, item)
        candidates = list(deduped.values())
        batches = [candidates[start : start + batch_size] for start in range(0, len(candidates), batch_size)]

        def review(batch: list[ExtractedTriple]) -> tuple[list[ExtractedTriple], list[dict[str, str]]]:
            payload = {
                "question": question,
                "candidates": [
                    {
                        "id": i,
                        **item.model_dump(),
                        "source_text": passage_text.get(item.source_id, "")[:1500],
                    }
                    for i, item in enumerate(batch)
                ],
            }
            decision = complete(reviewer or self.model,
                                render("extraction/evidence_review", payload, ensure_ascii=False))
            selected = decision.get("keep_ids", []) if isinstance(decision, dict) else []
            if not isinstance(selected, list):
                raise TypeError("triple reviewer must return keep_ids as a list")
            indices = {value for value in selected if type(value) is int and 0 <= value < len(batch)}
            reasons = decision.get("drop_reasons") if isinstance(decision, dict) else None
            reasons = reasons if isinstance(reasons, dict) else {}
            dropped = [{"triple": f"{item.subject} —{item.predicate}→ {item.object}", "source": item.source_id,
                        "why": _drop_reason(reasons.get(str(i)))}
                       for i, item in enumerate(batch) if i not in indices]
            return [item for i, item in enumerate(batch) if i in indices], dropped

        # Review batches run in parallel (HARNESS_EXTRACTION_CONCURRENCY); kept triples keep their order.
        results = parallel_map(review, batches, settings().extraction_concurrency)
        kept = [item for part, _ in results for item in part]
        dropped = [item for _, part in results for item in part]
        report: dict[str, Any] = {"candidates": len(triples), "unique": len(candidates), "kept": len(kept),
                                  "model_calls": len(batches)}
        if dropped:
            # Why facts are lost at this stage (the largest loss in the funnel): counts per reason and examples.
            report["dropped"] = dict(Counter(item["why"] for item in dropped).most_common())
            report["dropped_examples"] = dropped[:8]
        return kept, report

    @labelled("answer composition and verification")
    def answer(
        self,
        question: str,
        inferred: SchemaInference,
        triples: list[ExtractedTriple],
        verify: bool = False,
        verifier: ModelClient | None = None,
        notes: list[str] | None = None,
        report: dict[str, Any] | None = None,
    ) -> AnswerResult:
        """``notes`` (agentic runs) go to composition only; the verifier sees the plain question.

        One bounded repair (Phase 5.1): when the verifier removes every claim of the answer, or it cites
        only ids that are no triple's source, it is written once more from the triples alone (no
        research notes), told which statements or ids were rejected. ``report["repaired"]`` records that
        it happened; ``report["citations_dropped"]`` counts sentences dropped from the returned answer
        for citing unknown ids. A second rejection raises ``UnsupportedAnswer``; the caller decides what
        to show instead.
        """
        try:
            return self._compose(question, inferred, triples, verify, verifier, notes, report=report)
        except UnsupportedAnswer as first:
            # The v1 templates have no repair section, so a second call would repeat the first.
            if first.stage not in {"verification", "citation"} or settings().prompt_set == "v1":
                raise
            if first.stage == "citation" and not first.unknown_sources:
                raise                                      # nothing to tell the composer: same prompt again
            emit_model_event("Answer unsupported", "writing the answer once more from the triples alone")
            if report is not None:
                report["repaired"] = True
                report["notes_dropped"] = bool(notes)
            try:
                return self._compose(question, inferred, triples, verify, verifier, None,
                                     repair=first.removed or [item["text"] for item in first.sentences
                                                              if item["verdict"] == "removed"],
                                     bad_sources=first.unknown_sources, report=report)
            except UnsupportedAnswer as second:
                raise UnsupportedAnswer(f"{first}; after one repair: {second}", stage=second.stage,
                                        removed=second.removed, sentences=second.sentences,
                                        unknown_sources=second.unknown_sources) from second

    def _compose(
        self,
        question: str,
        inferred: SchemaInference,
        triples: list[ExtractedTriple],
        verify: bool,
        verifier: ModelClient | None,
        notes: list[str] | None,
        repair: list[str] | None = None,
        bad_sources: list[str] | None = None,
        report: dict[str, Any] | None = None,
    ) -> AnswerResult:
        payload: dict[str, Any] = {
            "question": question,
            "classes": inferred.classes,
            "properties": inferred.properties,
            "triples": [triple.model_dump() for triple in triples],
        }
        if notes:
            payload["research_notes"] = list(notes)
        extra: dict[str, Any] = {"repair": [str(item)[:300] for item in repair[:8]]} if repair else {}
        if bad_sources:
            extra["bad_sources"] = [str(item)[:80] for item in bad_sources[:8]]
        try:
            reply = complete(self.model, render("extraction/answer_composition", payload, **extra))
        except ModelFailure as exc:
            if "output limit" not in str(exc):
                raise
            # Cut off (a very long answer or a repetition loop): once more with explicit limits.
            emit_model_event("Answer too long", "retrying answer composition with a shorter format")
            reply = complete(self.model, render("extraction/answer_composition", payload, brief=True, **extra))
        if isinstance(reply, dict):
            reply = {key: value for key, value in reply.items() if key != "sentences"}   # set only by verification
        result = AnswerResult.model_validate(reply)
        sources = {triple.source_id for triple in triples}
        if report is not None:
            report.pop("citations_dropped", None)             # describes the attempt that is returned
        text, cited, unknown, dropped = drop_unknown_citations(result.answer, result.cited_source_ids, sources)
        if unknown and (not text or not cited):
            raise UnsupportedAnswer(
                f"no sentence of the answer cites a source of the extracted evidence (it cited {unknown[:3]}); "
                f"{len(triples)} triples available", stage="citation", unknown_sources=unknown)
        if dropped:
            # e.g. "[research_notes]" or a passage id no triple carries: those sentences have no evidence.
            emit_model_event("Answer cites unknown sources",
                             f"dropped {dropped} sentence{'s' if dropped > 1 else ''} citing {', '.join(unknown[:3])}")
            if report is not None:
                report["citations_dropped"] = {"sentences": dropped, "ids": unknown[:5]}
        if unknown:
            result = result.model_copy(update={"answer": text, "cited_source_ids": cited})
        result = result.model_copy(update={"sparql": self.answer_query(triples, result.cited_source_ids)})
        if verify:
            reviewed = complete(verifier or self.model, render(
                "extraction/answer_verification",
                {
                    "question": question,
                    "draft_answer": result.answer,
                    "triples": [triple.model_dump() for triple in triples],
                },
                ensure_ascii=False,
            ))
            removed = reviewed.get("removed_claims")
            removed = [str(item) for item in removed] if isinstance(removed, list) else []
            sentences = self._sentence_verdicts(reviewed.get("sentences"), sources)
            accepted = [item for item in sentences if item["verdict"] == "accepted"]
            if (reviewed.get("supported") is not True or not str(reviewed.get("answer", "")).strip()
                    or (sentences and not accepted)):
                removed = removed or [item["text"] for item in sentences if item["verdict"] == "removed"]
                example = f", e.g. {removed[0][:120]!r}" if removed else ""
                raise UnsupportedAnswer(
                    f"answer has no fully supported factual claim (verifier removed {len(removed)} "
                    f"claims{example}; {len(triples)} triples available"
                    f"{', research notes used by composition' if notes else ''})",
                    removed=removed, sentences=sentences)
            if accepted:
                # With per-sentence verdicts the answer is exactly the accepted sentences, in order, and
                # its sources are theirs (each already checked against the supplied triples).
                text = " ".join(item["text"] for item in accepted)
                cited = list(dict.fromkeys(source for item in accepted for source in item["sourceIds"]))
            else:
                verified_sources = reviewed.get("cited_source_ids", [])
                if not isinstance(verified_sources, list) or not verified_sources or not set(verified_sources) <= sources:
                    raise UnsupportedAnswer("verified answer lacks valid triple provenance", stage="provenance")
                text, cited = str(reviewed["answer"]), [str(source) for source in verified_sources]
            result = result.model_copy(update={
                "answer": text, "cited_source_ids": cited,
                "sparql": self.answer_query(triples, cited), "sentences": sentences,
            })
        return result

    @staticmethod
    def _sentence_verdicts(raw: object, sources: set[str]) -> list[dict[str, Any]]:
        """The verifier's per-sentence verdicts; a sentence counts as accepted only when it is marked
        supported and cites at least one source of the supplied triples."""
        def clean(value: object) -> str:
            return str(value).strip().strip("[]").strip()

        out: list[dict[str, Any]] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                continue
            text = str(item["text"]).strip()
            ids = list(dict.fromkeys(clean(value) for value in item.get("source_ids") or [] if clean(value) in sources))
            accepted = item.get("supported") is True and bool(ids)
            reason = str(item.get("reason") or "")
            if item.get("supported") is True and not ids:
                reason = reason or "cites no source of the supplied triples"
            in_text = [item for group in _BRACKET.findall(text) for item in citation_ids(group, sources) or []]
            outside = [value for value in in_text if value not in sources]
            if accepted and outside:
                accepted, reason = False, f"cites a source not among the triples: {', '.join(outside[:3])}"
            if accepted and not in_text:
                # Keep the sentence traceable in the answer text: its sources in brackets before the full stop.
                body, stop = (text[:-1], text[-1]) if text[-1] in ".!?" else (text, "")
                text = f"{body.rstrip()} [{', '.join(ids)}]{stop}"
            out.append({"text": text, "sourceIds": ids, "verdict": "accepted" if accepted else "removed",
                        "reason": reason})
        return out

    def answer_query(self, triples: list[ExtractedTriple], cited: list[str]) -> str:
        """SPARQL that selects the triples behind the answer: one pattern per predicate they use.

        Built from the cited triples (all triples when none is cited), so the query always matches
        the answer and uses only subschema terms.
        """
        chosen = [triple for triple in triples if triple.source_id in set(cited)] or list(triples)
        iris: list[str] = []
        for triple in chosen:
            iri = self.property_names.get(triple.predicate)
            if iri is not None and str(iri) not in iris:
                iris.append(str(iri))
        head = f"PREFIX hrn: <{HRN}>\n"
        if not iris:
            return head + "SELECT ?subject ?predicate ?object WHERE { ?subject ?predicate ?object . }"
        names = [f"hrn:{iri[len(str(HRN)):]}" if iri.startswith(str(HRN)) else f"<{iri}>" for iri in iris]
        branches = [f"{{ ?subject {name} ?object . BIND({name} AS ?predicate) }}" for name in names]
        return head + "SELECT ?subject ?predicate ?object WHERE {\n  " + "\n  UNION ".join(branches) + "\n}"

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


__all__ = ["ExtractionEngine", "local_name"]
