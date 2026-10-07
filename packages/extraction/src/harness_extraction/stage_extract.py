"""Triple extraction: direct (subschema predicates) and EDC (extract, define, canonicalize).

A mixin of ``ExtractionEngine``.
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any

from harness_config import settings
from harness_prompts import render
from harness_semantic import HRN, TOP_CLASS
from rdflib import RDFS, URIRef

from .call_control import call_interrupted, parallel_map
from .errors import ModelFailure
from .events import labelled
from .model_client import complete
from .naming import _literal_value, local_name, prompt_shape, relation_name
from .schemas import (
    EDCDefinition,
    EDCOpenTriple,
    ExtractedTriple,
    ExtractionResult,
    SchemaInference,
)

# Lexical candidates shown to the mapping model per open proposition, best first.
_LEXICAL_CANDIDATES = 3
# Word boundaries inside camelCase names: usesDataset -> uses Dataset, BERTLarge -> BERT Large.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Function words left out of the word-overlap term: with camelCase split, "by" alone would make
# "is outperformed by" overlap authoredBy.
_FUNCTION_WORDS = frozenset({"a", "an", "the", "is", "are", "was", "were", "be", "been", "by", "of", "on", "in",
                             "to", "for", "with", "and", "or", "as", "at", "from", "its", "it"})
# Subjects that denote the passage's own paper (after _plain): its authors speaking about their work.
# Deliberately short and exact; "our method" or "the proposed model" name a method, not the paper.
_SELF_REFERENCES = frozenset({
    "we", "us", "the authors", "authors", "this paper", "the paper", "our paper", "the present paper",
    "this work", "our work", "the present work", "this study", "our study", "the present study",
    "this article", "our article",
})


def _plain(value: str) -> str:
    """Case- and punctuation-insensitive form used to compare a subject with a paper id or title."""
    return " ".join(re.findall(r"\w+", str(value).casefold()))


class ExtractionStage:
    @labelled("triple extraction")
    def extract(
        self, question: str, inferred: SchemaInference, passages: list[dict[str, Any]]
    ) -> ExtractionResult:
        result = ExtractionResult.model_validate(complete(self.model, render(
            "extraction/triple_extraction",
            {
                "question": question,
                "classes": inferred.classes,
                "properties": inferred.properties,
                "property_signatures": self.signatures(list(inferred.properties)),
                "passages": [
                    {
                        "passage_id": row["passage_id"],
                        "paper_id": row["paper_id"],
                        "text": row["text"][:1800],
                    }
                    for row in passages
                ],
            }, **prompt_shape(),
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
        """Run EDC-style Extract -> Define -> Canonicalize over one batch of passages.

        1. Extract (``edc_extract``, one call): open propositions, each tied to a passage of the batch.
        2. Define (``edc_define``, one call when there are propositions): one definition per distinct open
           relation; a malformed entry loses only its own definition.
        3. Canonicalize: a lexical score (word overlap and string similarity of the open relation and its
           definition against each property name, camelCase split) only *proposes* the top candidates.
           Every proposition then goes to the mapping model (``edc_map``, one call for the whole batch),
           which sees the subject, object, evidence and paper id, the lexical candidates first and the
           full property catalog (domain, range, definition), and chooses a property or none. A choice is
           used when its confidence is at least ``HARNESS_MAP_MODEL_MIN``. Only when that call fails (or
           its reply addresses no proposition at all) is the best lexical candidate accepted on its own
           (score at least ``HARNESS_MAP_LEXICAL_MIN``, method ``lexical_fallback``).
        4. Checks on every mapped property of the schema (not on relations kept from open extraction):
           a property whose domain is Paper keeps the proposition only when its subject denotes the
           passage's paper (``_paper_reference``), and the subject is then written as the paper id; a
           subject that is not the paper is rejected (``domain_mismatch``). An object that denotes the paper
           under a property whose range cannot hold a paper is rejected (``inverted_direction``).
        5. A proposition left without a property is dropped and reported (``unmapped_dropped``), or with
           ``HARNESS_UNMAPPED_RELATIONS=keep`` registered under its own relation (domain ResearchEntity,
           subject unchanged).

        The ``canonicalize`` report has one decision per proposition: open relation, selected property,
        score (model confidence, or the lexical score), method (``ontology_review``, ``lexical_fallback``,
        ``kept_new_relation``, or why it stayed unmapped: ``model_rejected``, ``below_threshold``,
        ``model_invalid``, ``no_answer``, ``model_failed``, ``no_properties``, ``domain_mismatch``,
        ``inverted_direction``), the lexical candidates, and the rejected property when a check failed.
        """
        paper_by_source = {str(row["passage_id"]): str(row["paper_id"]) for row in passages}
        paper_names = self._paper_names(passages)
        open_triples = self._open_triples(question, passages)
        definitions = self._define(open_triples)
        definition_map = {self._norm(item.relation): item.definition for item in definitions}
        relations = list(inferred.properties)
        candidates = [
            [name for _, name in self._lexical_ranking(item.predicate, self._definition(item, definition_map),
                                                       relations)[:_LEXICAL_CANDIDATES]]
            for item in open_triples
        ]
        mapping: dict[int, tuple[str | None, float, str]] | None = None
        review: dict[str, Any] = {"stage": "ontology_review", "propositions": len(open_triples), "mapped": 0}
        if open_triples and relations:
            catalog = self._property_catalog(relations)
            signature = {row["property"]: {key: row[key] for key in ("property", "domain", "range")}
                         for row in catalog}
            propositions = []
            for index, item in enumerate(open_triples):
                row = {"id": index, "subject": item.subject, "predicate": item.predicate, "object": item.object,
                       "evidence": item.evidence, "paper_id": paper_by_source.get(item.evidence_ids[0], ""),
                       "candidates": [signature[name] for name in candidates[index]]}
                if definition := self._definition(item, definition_map):
                    row["definition"] = definition
                propositions.append(row)
            try:
                mapping = self._map_by_model(question, propositions, relations, catalog)
            except (ModelFailure, TypeError, ValueError) as exc:
                if call_interrupted():
                    raise                                   # Stop or the run deadline: never fall back
                review["error"] = f"{type(exc).__name__}: {exc}"[:200]
            if mapping is None:
                review.setdefault("error", "the reply addresses no proposition")
            else:
                review["mapped"] = sum(1 for chosen, _, _ in mapping.values() if chosen)
        canonical, decisions, unmapped = self._canonicalize(
            open_triples, candidates, mapping, bool(relations), definition_map, paper_by_source, paper_names,
            relations)
        reports: list[dict[str, Any]] = [
            {"stage": "extract", "open_triples": len(open_triples)},
            {"stage": "define", "definitions": len(definitions)},
            review,
            {"stage": "canonicalize", "decisions": decisions},
        ]
        if unmapped:
            # Facts lost because their relation fits no subschema term (HARNESS_UNMAPPED_RELATIONS=drop).
            reports.append({"stage": "unmapped_dropped", "count": len(unmapped), "triples": unmapped})
        return ExtractionResult(triples=canonical), reports

    def _open_triples(self, question: str, passages: list[dict[str, Any]]) -> list[EDCOpenTriple]:
        """Extract: open propositions, each tied to at least one of the given passages."""
        allowed_ids = {row["passage_id"] for row in passages}
        payload = complete(self.edc_model, render("extraction/edc_extract", {
            "question": question,
            "passages": [{"source_id": row["passage_id"], "paper_id": row["paper_id"],
                          "text": row["text"][:1800]} for row in passages],
        }, **prompt_shape()))
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
        return open_triples

    def _define(self, open_triples: list[EDCOpenTriple]) -> list[EDCDefinition]:
        """Define: one definition per distinct open relation. Entries are validated one by one, so a
        malformed entry loses only its own definition."""
        distinct = sorted({item.predicate.strip() for item in open_triples if item.predicate.strip()})
        if not distinct:
            return []
        defined = complete(self.edc_model, render("extraction/edc_define", {"relations": distinct}))
        raw = defined.get("definitions") if isinstance(defined, dict) else None
        definitions: list[EDCDefinition] = []
        for entry in raw if isinstance(raw, list) else []:
            try:
                definitions.append(EDCDefinition.model_validate(entry))
            except (TypeError, ValueError):
                continue
        return definitions

    @classmethod
    def _definition(cls, item: EDCOpenTriple, definition_map: dict[str, str]) -> str:
        return definition_map.get(cls._norm(item.predicate), "")

    def _property_catalog(self, relations: list[str]) -> list[dict[str, Any]]:
        catalog = []
        for relation in relations:
            iri = self.property_names.get(relation)
            domain = self.schema.graph.value(iri, RDFS.domain) if iri else None
            range_ = self.schema.graph.value(iri, RDFS.range) if iri else None
            entry = {
                "property": relation,
                "domain": local_name(domain) if isinstance(domain, URIRef) else None,
                "range": local_name(range_) if isinstance(range_, URIRef) else None,
            }
            if iri is not None:
                # The model tells paper-level from entity-level relations (addresses vs solves)
                # by their definitions too.
                entry["definition"] = " ".join(str(self.schema.graph.value(iri, RDFS.comment) or "").split())
            catalog.append(entry)
        return catalog

    def _map_by_model(self, question: str, propositions: list[dict[str, Any]], relations: list[str],
                      catalog: list[dict[str, Any]]) -> dict[int, tuple[str | None, float, str]] | None:
        """Canonicalize: the mapping model's verdict for every proposition of the batch (one call).

        Returns ``id -> (property or None, confidence, outcome)``, outcome ``ontology_review`` (used),
        ``model_rejected`` (no property fits), ``below_threshold`` or ``model_invalid`` (unknown property,
        confidence outside 0..1). Ids may come as integers or integer strings; ids the reply leaves out are
        absent. None when the reply addresses no proposition at all (the caller falls back to lexical).
        """
        mapped = complete(self.edc_model, render(
            "extraction/edc_map",
            {"question": question, "propositions": propositions, "allowed_properties": catalog},
            ensure_ascii=False, min_confidence=settings().map_model_min,
        ))
        rows = mapped.get("mappings") if isinstance(mapped, dict) else None
        if not isinstance(rows, list):
            return None
        threshold = settings().map_model_min
        mapping: dict[int, tuple[str | None, float, str]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            index = row.get("id")
            if isinstance(index, str) and index.strip().isdigit():
                index = int(index.strip())
            if type(index) is not int or not 0 <= index < len(propositions) or index in mapping:
                continue
            property_name = row.get("property")
            if isinstance(property_name, str):
                property_name = property_name.strip().removeprefix("hrn:") or None
            try:
                confidence = float(row.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                confidence = math.nan
            if property_name is None or (isinstance(property_name, str) and property_name.lower() == "null"):
                mapping[index] = (None, confidence if 0 <= confidence <= 1 else 0.0, "model_rejected")
            elif property_name not in relations or not 0 <= confidence <= 1:   # NaN fails the range too
                mapping[index] = (None, 0.0, "model_invalid")
            elif confidence < threshold:
                mapping[index] = (None, confidence, "below_threshold")
            else:
                mapping[index] = (property_name, confidence, "ontology_review")
        return mapping or None

    def _canonicalize(self, open_triples: list[EDCOpenTriple], candidates: list[list[str]],
                      mapping: dict[int, tuple[str | None, float, str]] | None, model_called: bool,
                      definition_map: dict[str, str], paper_by_source: dict[str, str],
                      paper_names: dict[str, str], relations: list[str],
                      ) -> tuple[list[ExtractedTriple], list[dict[str, Any]], list[dict[str, str]]]:
        """Each open fact under its chosen property (the model's choice; lexical only when the model call
        failed; else kept as a new relation), after the paper-domain and direction checks; returns the
        triples, one decision per open fact, and the facts left unmapped."""
        canonical: list[ExtractedTriple] = []
        decisions: list[dict[str, Any]] = []
        keep = settings().unmapped_relations == "keep"
        unmapped: list[dict[str, str]] = []
        for index, item in enumerate(open_triples):
            definition = self._definition(item, definition_map)
            source = item.evidence_ids[0]
            selected: str | None
            if not model_called:
                selected, score, method = None, 0.0, "no_properties"
            elif mapping is None:
                selected, score = self._canonical_property(item.predicate, definition, relations)
                method = "lexical_fallback" if selected else "model_failed"
            else:
                selected, score, method = mapping.get(index, (None, 0.0, "no_answer"))
            decision: dict[str, Any] = {"open": item.predicate, "candidates": candidates[index]}
            subject = item.subject
            if selected is not None:
                subject, rejection = self._check_mapping(item, selected, paper_by_source.get(source, ""),
                                                         paper_names)
                if rejection:
                    decision["rejected"] = selected
                    selected, method = None, rejection
            if selected is None and keep:
                kept = self._keep_relation(item, definition)
                if kept is not None:
                    subject, rejection = self._check_mapping(item, kept, paper_by_source.get(source, ""),
                                                             paper_names)
                    if rejection is None:
                        decision["unmapped_because"] = method
                        selected, method = kept, "kept_new_relation"
            decisions.append({**decision, "selected": selected, "score": round(score, 3), "method": method})
            if selected is None:
                unmapped.append({"subject": item.subject, "predicate": item.predicate, "object": item.object,
                                 "source_id": source, "reason": method})
                continue
            canonical.append(ExtractedTriple(
                subject=subject,
                predicate=selected,
                object=item.object,
                source_id=source,
                evidence=item.evidence or f"Evidence from {source}.",
            ))
        return canonical, decisions, unmapped

    def _check_mapping(self, item: EDCOpenTriple, selected: str, passage_paper: str,
                       paper_names: dict[str, str]) -> tuple[str, str | None]:
        """The subject to write under ``selected`` and why the mapping is rejected (None when it holds).

        Relations kept from open extraction keep the open subject unchanged and are not checked (their
        domain and range are the generic ones). For a schema property: a domain of Paper needs a subject
        that denotes a paper (written as that paper's id), else ``domain_mismatch``; an object that denotes
        a paper under a range that cannot hold a paper is ``inverted_direction``.
        """
        with self._schema_lock:
            if selected in self.kept_relations:
                return item.subject, None
        iri = self.property_names.get(selected)
        if iri is None:
            return item.subject, None
        graph = self.schema.graph
        domain, range_ = graph.value(iri, RDFS.domain), graph.value(iri, RDFS.range)
        subject_paper = self._paper_reference(item.subject, passage_paper, paper_names)
        object_paper = self._paper_reference(item.object, passage_paper, paper_names)
        if object_paper and isinstance(range_, URIRef) and \
                range_ not in set(graph.transitive_objects(HRN.Paper, RDFS.subClassOf)):
            return item.subject, "inverted_direction"
        if domain == HRN.Paper:
            if subject_paper is None:
                return item.subject, "domain_mismatch"
            return subject_paper, None
        return item.subject, None

    @staticmethod
    def _paper_names(passages: list[dict[str, Any]]) -> dict[str, str]:
        """Normalised paper ids and titles (when the passage rows carry one) -> paper id."""
        names: dict[str, str] = {}
        for row in passages:
            paper = str(row.get("paper_id") or "")
            if not paper:
                continue
            for name in (paper, row.get("title"), row.get("paper_title")):
                if key := _plain(str(name or "")):
                    names.setdefault(key, paper)
        return names

    @staticmethod
    def _paper_reference(value: str, passage_paper: str, paper_names: dict[str, str]) -> str | None:
        """The paper id ``value`` denotes, or None. Conservative, exact matches only: a paper id of the
        batch or a paper's full title (case and punctuation ignored), or a self-reference of the passage's
        own paper ("we", "this paper", "the authors", ... in ``_SELF_REFERENCES``). Anything else, a method
        name included, is not taken for the paper."""
        key = _plain(value)
        if key in paper_names:
            return paper_names[key]
        if key in _SELF_REFERENCES and passage_paper:
            return passage_paper
        return None

    def _keep_relation(self, item: EDCOpenTriple, definition: str) -> str | None:
        """Register the open fact's own relation (HARNESS_UNMAPPED_RELATIONS=keep) and return its name.

        The relation is an OWL 2 sub-property of the generic researchRelation (object range) or
        researchAttribute (a value), so the graph stays valid and the scorer's entailed level can
        relate it to other terms. Its domain is always the generic ResearchEntity: it does not depend
        on which batch registered it first, and no subject is rewritten for it. A wording whose name
        is already a property returns that property.
        """
        name = relation_name(item.predicate)
        if name is None:
            return None
        if name in self.property_names:
            return name
        try:
            self.register_local_relation(
                name,
                definition or f"Relation kept from open extraction, worded '{item.predicate}'.",
                domain_name=local_name(TOP_CLASS),
                range_name="xsd:string" if _literal_value(item.object) else local_name(TOP_CLASS),
            )
        except ValueError:
            return None
        with self._schema_lock:
            self.kept_relations.setdefault(name, item.predicate)
        return name

    @staticmethod
    def _norm(value: str) -> str:
        """Lower-case words of a relation or property name: an IRI or prefixed name (``hrn:usesDataset``)
        becomes its local name and camelCase is split (``usesDataset`` -> ``uses dataset``)."""
        text = str(value).strip()
        if re.fullmatch(r"\S+", text) and re.search(r"[#/:]", text):
            text = re.split(r"[#/:]", text.rstrip("#/:"))[-1]
        text = _CAMEL_BOUNDARY.sub(" ", text)
        return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()

    @classmethod
    def _lexical_ranking(cls, predicate: str, definition: str, relations: list[str]) -> list[tuple[float, str]]:
        """Every property with its lexical score against the open relation (and its definition), best first."""
        source = set(cls._norm(f"{predicate} {definition}").split()) - _FUNCTION_WORDS
        ranked = []
        for relation in relations:
            target = set(cls._norm(relation).split()) - _FUNCTION_WORDS
            overlap = len(source & target) / max(1, len(source | target))
            ratio = SequenceMatcher(None, cls._norm(predicate), cls._norm(relation)).ratio()
            ranked.append((0.55 * ratio + 0.45 * overlap, relation))
        return sorted(ranked, reverse=True)

    @classmethod
    def _canonical_property(cls, predicate: str, definition: str, relations: list[str]) -> tuple[str | None, float]:
        """The best lexical candidate when it scores at least HARNESS_MAP_LEXICAL_MIN (the fallback when the
        mapping model fails), with its score."""
        ranked = cls._lexical_ranking(predicate, definition, relations)
        if not ranked:
            return None, 0.0
        score, relation = ranked[0]
        return (relation, score) if score >= settings().map_lexical_min else (None, score)

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
