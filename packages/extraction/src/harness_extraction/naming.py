"""Names shared by the extraction stages: IRI local names, relation names, prompt shape."""

from __future__ import annotations

import re

from harness_config import settings
from rdflib import URIRef


def local_name(value: URIRef) -> str:
    return str(value).rsplit("#", 1)[-1]


def relation_name(text: str) -> str | None:
    """A lowerCamelCase property name from an open relation's wording ("is evaluated on" ->
    "isEvaluatedOn"), or None when the wording has no usable word."""
    words = re.findall(r"[A-Za-z0-9]+", str(text))
    if not words or not words[0][0].isalpha():
        return None
    name = words[0][:1].lower() + words[0][1:] + "".join(word[:1].upper() + word[1:] for word in words[1:])
    name = name[:64]
    return name if re.fullmatch(r"[a-z][A-Za-z0-9_]{0,63}", name) else None


def _literal_value(value: str) -> bool:
    """An object that is a value (a number or measurement, a long phrase) rather than a thing other papers
    could share. Names with digits ("GPT-4", "Llama 3") stay things."""
    text = str(value).strip()
    return bool(re.match(r"[-+~≈<>]?\s*\d", text) or "%" in text) or len(text.split()) > 6


def prompt_shape() -> dict[str, str]:
    """Template variables that set how broad the subschema is and how linked the extracted graph is."""
    config = settings()
    return {"schema_scope": config.schema_scope, "graph_linking": config.graph_linking}


