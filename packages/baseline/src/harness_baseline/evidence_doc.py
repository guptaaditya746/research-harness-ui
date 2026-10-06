"""The question's evidence document D as a run's input (``HARNESS_RUN_INPUT=evidence_doc``, docs/RUN_INPUT.md).

labelling builds one D per question (``<root>/<question id>/doc.json``): every triple the builder found,
each with the passage(s) that state it, one section per interpretation. In this mode a path does not
search the corpus: it gets D's triples, picks the ones that answer the question and writes the answer,
so its score measures choosing and answering, not finding.

A triple is given with its first source that is a corpus passage (the run's provenance only accepts
corpus ids); a triple with none is left out and counted.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from harness_extraction.schemas import ExtractedTriple


class EvidenceDocMissing(LookupError):
    """No evidence document for the question (or it cannot be read)."""


@dataclass(frozen=True)
class DocTriple:
    id: str                 # D's triple id ("t00042")
    reading: str            # D's interpretation id ("i1")
    s: str
    p: str
    o: str
    paper_id: str | None
    source_id: str
    evidence: str

    def extracted(self) -> ExtractedTriple:
        return ExtractedTriple(subject=self.s, predicate=self.p, object=self.o, source_id=self.source_id,
                               evidence=self.evidence or f"Stated in {self.source_id}.")

    def row(self) -> dict[str, Any]:
        """The triple as the agents' D tools show it."""
        return {"id": self.id, "subject": self.s, "predicate": self.p, "object": self.o,
                "paper_id": self.paper_id, "source_id": self.source_id, "evidence": self.evidence[:300]}


@dataclass(frozen=True)
class EvidenceDoc:
    question_id: str
    question: str
    path: Path
    sha256: str
    interpretations: list[dict[str, str]]
    triples: list[DocTriple]
    left_out: int = 0                       # triples with no corpus passage as source
    reading: str | None = None              # set when restricted to one interpretation
    chosen: int | None = field(default=None)  # set when restricted to the triples an agent chose

    def for_reading(self, reading: str | None) -> EvidenceDoc:
        """Only one interpretation's triples (an unknown or empty id keeps them all)."""
        if not reading or reading not in {item["id"] for item in self.interpretations}:
            return self
        return replace(self, triples=[t for t in self.triples if t.reading == reading], reading=reading)

    def subset(self, ids: Iterable[str]) -> EvidenceDoc:
        wanted = set(ids)
        kept = [t for t in self.triples if t.id in wanted]
        return replace(self, triples=kept, chosen=len(kept))

    def extracted(self) -> list[ExtractedTriple]:
        """One ``ExtractedTriple`` per distinct fact and source (the same fact under two readings once)."""
        seen: dict[tuple[str, str, str, str], ExtractedTriple] = {}
        for triple in self.triples:
            seen.setdefault((triple.s.casefold(), triple.p, triple.o.casefold(), triple.source_id), triple.extracted())
        return list(seen.values())

    def source_ids(self) -> list[str]:
        return list(dict.fromkeys(t.source_id for t in self.triples))

    def summary(self) -> dict[str, Any]:
        """What the run records about its input."""
        return {"question_id": self.question_id, "file": str(self.path), "sha256": self.sha256,
                "triples": len(self.triples), "left_out": self.left_out,
                "interpretations": [item["id"] for item in self.interpretations],
                **({"reading": self.reading} if self.reading else {}),
                **({"chosen_by_agent": self.chosen} if self.chosen is not None else {})}


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def find(root: Path, question: str) -> Path | None:
    """The ``doc.json`` whose question text equals ``question`` (case, spacing and punctuation aside)."""
    wanted = _norm(question)
    for path in sorted(Path(root).expanduser().glob("*/doc.json")):
        try:
            text = json.loads(path.read_text(encoding="utf-8")).get("question", {}).get("question", "")
        except (OSError, ValueError):
            continue
        if _norm(text) == wanted:
            return path
    return None


def load(root: Path, question: str, passages: dict[str, Any]) -> EvidenceDoc:
    """D for ``question`` from ``root``; ``passages`` is the corpus's passage index (``by_passage``)."""
    path = find(root, question)
    if path is None:
        raise EvidenceDocMissing(f"no evidence document for this question in {root} "
                                 "(build it with `labelling evidence-docs`, or set HARNESS_EVIDENCE_DOC_ROOT)")
    raw = path.read_bytes()
    doc = json.loads(raw)
    triples: list[DocTriple] = []
    left_out = 0
    interpretations = []
    for section in doc.get("interpretations") or []:
        reading = str(section.get("id") or "i1")
        interpretations.append({"id": reading, "label": str(section.get("label") or ""),
                                "gloss": str(section.get("gloss") or "")})
        for item in section.get("triples") or []:
            sources = [source for source in item.get("sources") or [] if str(source.get("source_id")) in passages]
            if not sources or not all(str(item.get(key) or "").strip() for key in ("s", "p", "o")):
                left_out += 1
                continue
            source = sources[0]
            triples.append(DocTriple(
                id=str(item.get("id")), reading=reading, s=str(item["s"]), p=str(item["p"]), o=str(item["o"]),
                paper_id=str(item["paper_id"]) if item.get("paper_id") else None,
                source_id=str(source["source_id"]), evidence=str(source.get("evidence") or "")))
    return EvidenceDoc(question_id=str(doc.get("question", {}).get("id") or path.parent.name),
                       question=question, path=path, sha256=hashlib.sha256(raw).hexdigest(),
                       interpretations=interpretations, triples=triples, left_out=left_out)


__all__ = ["DocTriple", "EvidenceDoc", "EvidenceDocMissing", "find", "load"]
