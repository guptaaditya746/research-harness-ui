"""Shared structured model calls for schema inference, extraction, and answers."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import replace
from difflib import SequenceMatcher
from functools import wraps
from typing import Any, Protocol

import httpx
from harness_semantic import HRN, Subschema
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from rdflib import RDF, RDFS, XSD, Literal, URIRef


class ModelFailure(RuntimeError):
    """Raised after the configured provider retries are exhausted."""


# ─── Run call budget ─────────────────────────────────────────────────────────
# A run's wall-clock deadline and stop flag, visible to every model call made
# on its behalf (context variables follow the run's thread and asyncio tasks).
_CALL_BUDGET: ContextVar[tuple[float, Callable[[], bool]] | None] = ContextVar(
    "harness_call_budget", default=None
)


@contextmanager
def call_budget(deadline: float, stopped: Callable[[], bool]) -> Iterator[None]:
    """Bound every model call inside the block by ``deadline`` (time.monotonic)."""
    token = _CALL_BUDGET.set((deadline, stopped))
    try:
        yield
    finally:
        _CALL_BUDGET.reset(token)


def bounded_timeout(timeout_s: float) -> float:
    """The per-call timeout, shortened to the time the run has left.

    Raises ModelFailure when the run was stopped or its deadline has passed,
    so no new model call starts after cancellation.
    """
    budget = _CALL_BUDGET.get()
    if budget is None:
        return timeout_s
    deadline, stopped = budget
    if stopped():
        raise ModelFailure("run stopped")
    remaining = deadline - time.monotonic()
    if remaining <= 1.0:
        raise ModelFailure("run deadline reached")
    return min(timeout_s, remaining)


def call_interrupted() -> str | None:
    """Why the current run's model calls must stop, or None."""
    budget = _CALL_BUDGET.get()
    if budget is None:
        return None
    deadline, stopped = budget
    if stopped():
        return "run stopped"
    if time.monotonic() > deadline:
        return "run deadline reached"
    return None


_CALL_POOL = ThreadPoolExecutor(
    max_workers=int(os.getenv("HARNESS_MODEL_CALL_THREADS", "16")), thread_name_prefix="model-call"
)


def run_interruptible(call: Callable[[], Any]) -> Any:
    """Run a blocking provider call that Stop and the run deadline can abandon.

    The HTTP request runs on a worker thread; the run's own thread waits in
    short intervals and gives up as soon as the run is stopped or out of time,
    so the run (and its model slot) is released without waiting for the
    provider. An abandoned request finishes in the background and is ignored.
    """
    if _CALL_BUDGET.get() is None:
        return call()
    future = _CALL_POOL.submit(copy_context().run, call)
    while True:
        try:
            return future.result(timeout=0.25)
        except FutureTimeout:
            reason = call_interrupted()
            if reason:
                future.cancel()
                raise ModelFailure(reason) from None


# ─── Model-call observability ───────────────────────────────────────────────
# What the current call is for (set by callers) and where start/finish events
# go (set by the runtime for each run). Both follow the run's thread/tasks.
_MODEL_PURPOSE: ContextVar[str] = ContextVar("harness_model_purpose", default="model call")
_MODEL_EVENTS: ContextVar[Callable[[str, str], None] | None] = ContextVar(
    "harness_model_events", default=None
)


@contextmanager
def model_purpose(purpose: str) -> Iterator[None]:
    """Label the model calls made inside the block (e.g. "schema inference")."""
    token = _MODEL_PURPOSE.set(purpose)
    try:
        yield
    finally:
        _MODEL_PURPOSE.reset(token)


@contextmanager
def model_events(callback: Callable[[str, str], None]) -> Iterator[None]:
    """Receive (action, detail) events for every model call inside the block."""
    token = _MODEL_EVENTS.set(callback)
    try:
        yield
    finally:
        _MODEL_EVENTS.reset(token)


def labelled(purpose: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: label every model call made by the wrapped function."""
    def wrap(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def inner(*args: Any, **kwargs: Any) -> Any:
            with model_purpose(purpose):
                return function(*args, **kwargs)
        return inner
    return wrap


def emit_model_event(action: str, detail: str) -> None:
    callback = _MODEL_EVENTS.get()
    if callback is not None:
        callback(action, detail)


def current_purpose() -> str:
    return _MODEL_PURPOSE.get()


def model_options(alias: str) -> dict[str, Any]:
    """Per-alias request options from HARNESS_MODEL_OPTIONS (JSON), if configured.

    Example: {"heavy-model": {"max_tokens": 6144,
              "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}}
    """
    raw = os.getenv("HARNESS_MODEL_OPTIONS", "").strip()
    if not raw:
        return {}
    try:
        options = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    value = options.get(alias) if isinstance(options, dict) else None
    return value if isinstance(value, dict) else {}


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove reasoning blocks a thinking model may emit before its answer."""
    text = _THINK_BLOCK.sub("", text)
    # An unterminated block means the answer never started.
    if re.search(r"<think>", text, re.IGNORECASE):
        text = re.split(r"<think>", text, flags=re.IGNORECASE)[0]
    return text.strip()


class ModelClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


def _json_object(text: str) -> dict[str, Any]:
    text = strip_reasoning(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise TypeError("model response must be a JSON object")
    return value


class OpenAICompatibleModel:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 30,
        retries: int | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1")
        ).rstrip("/")
        self.model = model or os.getenv("HARNESS_MODEL_SMALL", "lite-model")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "local")
        self.timeout_s = timeout_s
        self.retries = retries or int(os.getenv("PROVIDER_MAX_RETRIES", "3"))

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """One structured call, bounded in time and output, with visible progress.

        Transient transport failures (connection errors, 429, 5xx) are
        retried. A read timeout is not: repeating the same long generation
        only multiplies the wait. A reply cut off by the output limit is
        reported as such rather than as malformed JSON.
        """
        purpose = current_purpose()
        options = model_options(self.model)
        max_tokens = int(options.get("max_tokens") or os.getenv("HARNESS_MAX_OUTPUT_TOKENS", "4096"))
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **(options.get("extra_body") or {}),
        }
        error: str = "no attempt made"
        for attempt in range(self.retries):
            timeout_s = bounded_timeout(self.timeout_s)
            started = time.monotonic()
            emit_model_event(
                "Model call started",
                f"{self.model} · {purpose}" + (f" · attempt {attempt + 1}" if attempt else ""),
            )

            def request(timeout_s: float = timeout_s) -> dict[str, Any]:
                with httpx.Client(timeout=timeout_s) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=body,
                    )
                    if response.status_code >= 400:
                        raise _ProviderStatus(response.status_code, response.text[:400])
                    return response.json()

            try:
                payload = run_interruptible(request)
            except httpx.TimeoutException:
                took = time.monotonic() - started
                error = f"timed out after {took:.0f}s ({self.model} · {purpose})"
                emit_model_event("Model call failed", error)
                raise ModelFailure(error) from None
            except _ProviderStatus as exc:
                error = f"HTTP {exc.status}: {exc.body}"
                emit_model_event("Model call failed", f"{self.model} · {purpose} · {error}")
                if exc.status == 429 or exc.status >= 500:
                    self._backoff(attempt)
                    continue
                raise ModelFailure(error) from None
            except httpx.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
                emit_model_event("Model call failed", f"{self.model} · {purpose} · {error}")
                self._backoff(attempt)
                continue
            took = time.monotonic() - started
            try:
                choice = payload["choices"][0]
                content = str(choice["message"].get("content") or "")
                usage = payload.get("usage") or {}
                tokens = usage.get("completion_tokens")
                if choice.get("finish_reason") == "length":
                    raise ValueError(
                        f"reply hit the {max_tokens}-token output limit before finishing"
                        + (" (a thinking model spent it on reasoning)"
                           if choice["message"].get("reasoning_content") or "<think>" in content else "")
                    )
                value = _json_object(content)
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                error = f"unusable reply: {exc}"[:300]
                emit_model_event("Model call failed", f"{self.model} · {purpose} · {error} · {took:.1f}s")
                # A deterministic model repeats the same reply; do not retry.
                raise ModelFailure(error) from None
            emit_model_event(
                "Model call finished",
                f"{self.model} · {purpose} · {took:.1f}s" + (f" · {tokens} tokens" if tokens else ""),
            )
            return value
        raise ModelFailure(f"provider failed after {self.retries} attempts: {error}")

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(0.25 * (2**attempt))


class _ProviderStatus(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body


class SchemaInference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    gloss: str
    classes: list[str]
    properties: list[str]
    class_definitions: list[dict[str, Any]] = Field(default_factory=list)
    property_definitions: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_catalogs(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for key in ("classes", "properties"):
            item = normalized.get(key)
            if isinstance(item, dict):
                normalized[key] = list(item)
        return normalized


class ExtractedTriple(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    predicate: str
    object: str
    source_id: str
    evidence: str = Field(validation_alias=AliasChoices("evidence", "evidence_span"))


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    triples: list[ExtractedTriple] = Field(default_factory=list)


class EDCOpenTriple(BaseModel):
    """Evidence-linked open proposition produced by EDC Extract."""

    model_config = ConfigDict(extra="ignore")
    subject: str
    predicate: str
    object: str
    source_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evidence: str = ""


class EDCDefinition(BaseModel):
    relation: str
    definition: str


class EDCDefinitionResult(BaseModel):
    definitions: list[EDCDefinition] = Field(default_factory=list)


class AnswerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sparql: str
    answer: str
    cited_source_ids: list[str]


def local_name(value: URIRef) -> str:
    return str(value).rsplit("#", 1)[-1]


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
        self._local_schema = False

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

    def register_local_class(self, name: str, definition: str) -> URIRef:
        """Introduce one query-local class without changing the seed ontology."""
        if not re.fullmatch(r"[A-Z][A-Za-z0-9_]{0,63}", name):
            raise ValueError(f"invalid proposed class name: {name!r}")
        if name in self.class_names:
            return self.class_names[name]
        self._ensure_local_schema()
        iri = URIRef(HRN[name])
        self.schema.graph.add((iri, RDF.type, RDFS.Class))
        self.schema.graph.add((iri, RDFS.label, Literal(name)))
        self.schema.graph.add((iri, RDFS.comment, Literal(definition)))
        self.schema = replace(self.schema, classes=self.schema.classes | {iri})
        self.class_names[name] = iri
        return iri

    def register_local_relation(
        self, name: str, definition: str, *, domain_name: str = "Paper",
        range_name: str = "xsd:string",
    ) -> URIRef:
        """Add a query-local relation with an explicit domain and range."""
        if not re.fullmatch(r"[a-z][A-Za-z0-9_]{0,63}", name):
            raise ValueError(f"invalid proposed relation name: {name!r}")
        iri = URIRef(HRN[name])
        if iri in self.schema.properties:
            return iri
        self._ensure_local_schema()
        graph = self.schema.graph
        graph.add((iri, RDF.type, RDF.Property))
        graph.add((iri, RDFS.label, Literal(name)))
        graph.add((iri, RDFS.comment, Literal(definition)))
        graph.add((iri, RDFS.domain, self.class_names.get(domain_name, HRN.Paper)))
        graph.add((iri, RDFS.range, self.class_names.get(range_name, XSD.string)))
        self.schema = replace(self.schema, properties=self.schema.properties | {iri})
        self.property_names[name] = iri
        return iri

    @labelled("schema inference")
    def infer_schema(self, question: str, passages: list[dict[str, Any]]) -> SchemaInference:
        allowed_classes = sorted(self.class_names)
        allowed_properties = sorted(self.property_names)
        result = SchemaInference.model_validate(self.schema_model.complete_json(
            "Use the supplied ontology as a seed. Reuse its classes and properties whenever "
            "they express the question. When no seed term is suitable, invent a concise "
            "UpperCamelCase class or lowerCamelCase property. Return strict JSON with label, "
            "gloss, nonempty classes and properties. For invented terms, also return "
            "class_definitions [{name, definition}] and property_definitions "
            "[{name, definition, domain, range}], using class names for domain and range "
            "or xsd:string for a literal range. Prefer reusable concept types over making "
            "each named task, method, or benchmark its own class; those names are usually "
            "instance values. Keep proposed terms grounded in the question and passages; "
            "do not invent facts.",
            json.dumps({
                "question": question,
                "allowed_classes": allowed_classes,
                "allowed_properties": allowed_properties,
                "passages": [
                    {"source_id": row["passage_id"], "text": row["text"][:1200]}
                    for row in passages
                ],
            }),
        ))
        if not result.classes or not result.properties:
            raise ValueError("schema inference returned no classes or properties")
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
                self.register_local_class(name, definition)
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
                range_name=range_name,
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

    @labelled("triple extraction")
    def extract(
        self, question: str, inferred: SchemaInference, passages: list[dict[str, Any]]
    ) -> ExtractionResult:
        result = ExtractionResult.model_validate(
            self.model.complete_json(
                "Extract only supported triples. Use a supplied passage_id as source_id and copy a short evidence span. Return strict JSON with triples.",
                json.dumps(
                    {
                        "question": question,
                        "classes": inferred.classes,
                        "properties": inferred.properties,
                        "passages": [
                            {
                                "passage_id": row["passage_id"],
                                "paper_id": row["paper_id"],
                                "text": row["text"][:1800],
                            }
                            for row in passages
                        ],
                    }
                ),
            )
        )
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
        payload = self.edc_model.complete_json(
            "Extract evidence-backed atomic propositions relevant to the question. Use the supplied paper_id as subject for paper-level claims. Keep predicates open natural-language relations. "
            "Return strict JSON with triples; each triple must include source_id (a supplied passage_id), subject, predicate, object, and evidence.",
            json.dumps({
                "question": question,
                "passages": [{"source_id": row["passage_id"], "paper_id": row["paper_id"], "text": row["text"][:1800]} for row in passages],
            }),
        )
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
            defined = self.edc_model.complete_json(
                "Define each open predicate in subject-to-object terms. Return strict JSON with definitions, each containing relation and definition.",
                json.dumps({"relations": distinct}),
            )
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
                property_catalog.append({
                    "property": relation,
                    "domain": local_name(domain) if isinstance(domain, URIRef) else None,
                    "range": local_name(range_) if isinstance(range_, URIRef) else None,
                })
            mapped = self.edc_model.complete_json(
                "Map each open relation to one allowed ontology property only when the evidence supports the same subject-to-object claim. Return strict JSON with mappings: [{id, property, confidence}]. Use null property when none fits. Confidence must be between 0 and 1. Never add a fact or source.",
                json.dumps({"question": question, "unresolved": unresolved,
                            "allowed_properties": property_catalog}, ensure_ascii=False),
            )
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
        for batch_no, start in enumerate(range(0, len(passages), batch_size), start=1):
            batch = passages[start : start + batch_size]
            try:
                result = self.extract(question, inferred, [
                    {**row, "text": row["text"][:text_limit]} for row in batch
                ])
            except ValueError as exc:
                reports.append({"batch": batch_no, "passages": [p["passage_id"] for p in batch], "error": str(exc)})
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
    ) -> tuple[list[ExtractedTriple], dict[str, int]]:
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
        kept: list[ExtractedTriple] = []
        calls = 0
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
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
            decision = (reviewer or self.model).complete_json(
                "You review candidate atomic knowledge-graph triples. Select only IDs whose exact subject-predicate-object claim answers or directly supports the question AND whose source text supports the claim. Never add triples. Return strict JSON {\"keep_ids\": [integer IDs]}. If none qualify, return an empty list.",
                json.dumps(payload, ensure_ascii=False),
            )
            calls += 1
            selected = decision.get("keep_ids", []) if isinstance(decision, dict) else []
            if not isinstance(selected, list):
                raise TypeError("triple reviewer must return keep_ids as a list")
            indices = {value for value in selected if type(value) is int and 0 <= value < len(batch)}
            kept.extend(item for i, item in enumerate(batch) if i in indices)
        return kept, {"candidates": len(triples), "unique": len(candidates), "kept": len(kept), "model_calls": calls}

    @labelled("answer composition and verification")
    def answer(
        self,
        question: str,
        inferred: SchemaInference,
        triples: list[ExtractedTriple],
        verify: bool = False,
        verifier: ModelClient | None = None,
    ) -> AnswerResult:
        result = AnswerResult.model_validate(
            self.model.complete_json(
                "Compose a concise answer supported only by the supplied triples. Return exactly one JSON object with only these three keys and field types: "
                "sparql is a string containing executable SPARQL over prefix hrn: <https://harness.local/schema#>; "
                "answer is one plain-language string; cited_source_ids is an array of source_id strings copied from the supplied triples. "
                "The answer field must always be a string, never an array or object. If the answer has several themes or points, "
                "write them as sentences within that single string rather than returning structured items. "
                'Example shape: {"sparql":"SELECT ?s WHERE { ?s ?p ?o }",'
                '"answer":"The evidence connects theme A with theme B.","cited_source_ids":["actual-source-id-from-input"]}. '
                "Replace all example values with a query, answer, and source IDs grounded in the supplied triples; do not copy placeholders.",
                json.dumps(
                    {
                        "question": question,
                        "classes": inferred.classes,
                        "properties": inferred.properties,
                        "triples": [triple.model_dump() for triple in triples],
                    }
                ),
            )
        )
        sources = {triple.source_id for triple in triples}
        if not set(result.cited_source_ids) <= sources:
            raise ValueError("answer cites a source outside the extracted evidence")
        if verify:
            reviewed = (verifier or self.model).complete_json(
                "Verify every factual claim in the draft answer against the supplied atomic triples and their evidence. Revise the answer to remove unsupported claims. Return strict JSON with supported (boolean), answer (string), and cited_source_ids (list). Set supported=false if no factual answer remains. Never add a source or fact.",
                json.dumps({
                    "question": question,
                    "draft_answer": result.answer,
                    "triples": [triple.model_dump() for triple in triples],
                }, ensure_ascii=False),
            )
            if reviewed.get("supported") is not True or not str(reviewed.get("answer", "")).strip():
                raise ValueError("answer has no fully supported factual claim")
            verified_sources = reviewed.get("cited_source_ids", [])
            if not isinstance(verified_sources, list) or not verified_sources or not set(verified_sources) <= sources:
                raise ValueError("verified answer lacks valid triple provenance")
            result = result.model_copy(update={
                "answer": str(reviewed["answer"]),
                "cited_source_ids": [str(source) for source in verified_sources],
            })
        return result

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


__all__ = [
    "AnswerResult",
    "ExtractedTriple",
    "ExtractionEngine",
    "ExtractionResult",
    "ModelClient",
    "ModelFailure",
    "OpenAICompatibleModel",
    "SchemaInference",
]
