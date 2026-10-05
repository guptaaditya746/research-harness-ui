"""Schema inference: the subschema a question needs, its invented terms registered in the engine.

A mixin of ``ExtractionEngine``.
"""

from __future__ import annotations

from typing import Any

from harness_config import settings
from harness_prompts import render
from harness_semantic import TOP_ATTRIBUTE, TOP_CLASS, TOP_RELATION, property_kind

from .events import emit_model_event, labelled
from .model_client import complete
from .naming import local_name, prompt_shape
from .schemas import SchemaInference


class SchemaStage:
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
        payload["property_signatures"] = self.signatures(allowed_properties)
        if notes:
            payload["research_notes"] = list(notes)
        result = SchemaInference.model_validate(complete(self.schema_model, render(
            "extraction/schema_inference", payload, **prompt_shape(),
        )))
        return self.apply_schema(result, question)

    def apply_schema(self, result: SchemaInference, question: str) -> SchemaInference:
        """Validate a proposed subschema and register its invented terms in this engine's schema.

        Separate from :meth:`infer_schema` so a stored proposal (e.g. a resumed evidence-document
        build) can be applied again without a model call.
        """
        allowed_classes = sorted(self.class_names)
        allowed_properties = sorted(self.property_names)
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
