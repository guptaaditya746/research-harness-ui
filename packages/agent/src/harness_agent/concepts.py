"""Agent-proposed concept readings shown before every agentic research run.

``HARNESS_CONCEPT_GROUNDING``:

- ``retrieve_first`` (default): retrieve the question's top passages and keyphrases first
  (``CorpusStore.hybrid_rank``), show them to the model with the question, and let each reading
  cite the passages that support it. Each reading is then grounded by ranked retrieval: a passage
  supports it when it contains every content word of one of its search terms or its label
  (plural and suffix tolerant), or when the reading cited it.
- ``legacy``: propose from the question alone and count exact-phrase matches (which often
  found 0 passages for multi-word terms: the "0 related paragraphs" bug).

``HARNESS_HIDE_UNGROUNDED_READINGS`` (default true, ``retrieve_first`` only) drops readings without
any supporting passage, unless none has support. Hiding runs after the minimum-count check, so
fewer than three readings can be shown: better than readings the corpus cannot answer.
``evidence_hits`` counts supporting passages among the top ``GROUNDING_DEPTH`` ranked ones plus the
cited ones, not all matches in the corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from harness_config import settings
from harness_extraction import model_purpose, using_prompt
from harness_prompts import render

MIN_CONCEPTS = 3
MAX_CONCEPTS = 4
EVIDENCE_SHOWN = 8          # passages shown to the model before it proposes readings
KEYPHRASES_SHOWN = 12
GROUNDING_DEPTH = 50        # ranked passages inspected per reading
STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "these", "those", "into", "over", "under",
    "about", "which", "what", "when", "where", "who", "whom", "whose", "how", "why", "are", "was",
    "were", "been", "being", "have", "has", "had", "does", "did", "their", "there", "they", "them",
    "than", "then", "such", "also", "most", "more", "less", "each", "any", "all", "our", "its",
    # Short words whose plural forms are other words ("new"/"news", "not"/"notes").
    "new", "not", "now", "one", "two", "can", "may", "per", "via",
}

class ConceptProposalError(RuntimeError):
    """The agent could not produce a valid set of concept readings."""


@dataclass
class Concept:
    label: str
    gloss: str
    classes: list[str]
    search_terms: list[str]
    evidence_hits: int = 0
    sample_sources: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "gloss": self.gloss,
            "classes": self.classes,
            "search_terms": self.search_terms,
            "evidence_hits": self.evidence_hits,
            "sample_sources": self.sample_sources,
            "evidence_ids": self.evidence_ids,
        }


@dataclass
class ConceptProposal:
    """The readings, and the evidence the model saw before proposing them."""

    concepts: list[Concept]
    evidence: list[dict[str, Any]] = field(default_factory=list)
    keyphrases: list[str] = field(default_factory=list)
    hidden: list[str] = field(default_factory=list)


def _clean(text: object, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def _parse(payload: dict[str, Any], allowed_classes: set[str]) -> list[Concept]:
    raw = payload.get("concepts") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    concepts: list[Concept] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = _clean(item.get("label"), 80)
        gloss = _clean(item.get("gloss"), 300)
        key = label.casefold()
        if not label or not gloss or key in seen:
            continue
        seen.add(key)
        classes = [
            str(name) for name in item.get("classes") or []
            if isinstance(name, str) and name in allowed_classes
        ]
        terms = [
            _clean(term, 60) for term in item.get("search_terms") or []
            if isinstance(term, str) and _clean(term, 60)
        ][:5]
        cited = [str(value) for value in item.get("evidence_ids") or [] if isinstance(value, str | int)][:8]
        concepts.append(Concept(label, gloss, classes, terms, evidence_ids=cited))
    return concepts


def _content_tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.casefold()) if len(word) >= 3 and word not in STOP_WORDS]


def _supports(tokens: list[str], words: set[str]) -> bool:
    """Every token appears as a word, a plural or singular of one ("llm" matches "llms", "tools"
    matches "tool"), or, for longer words, the stem of one ("attack" matches "attacked")."""
    def present(token: str) -> bool:
        variants = {token, token + "s", token + "es"}
        if token.endswith("s") and len(token) >= 4:
            variants.add(token[:-1])
        if variants & words:
            return True
        stem = token[: max(4, len(token) - 2)]
        return len(token) >= 5 and any(word.startswith(stem) for word in words)

    return bool(tokens) and all(present(token) for token in tokens)


def gather_evidence(corpus: object | None, question: str) -> tuple[list[dict[str, Any]], list[str]]:
    """The question's top-ranked passages (short excerpts) and the keyphrases of their papers."""
    rank = getattr(corpus, "hybrid_rank", None)
    if rank is None:
        return [], []
    try:
        items = rank(question, limit=EVIDENCE_SHOWN)["items"]
    except Exception:  # noqa: BLE001 - evidence is advisory; proposal works without it
        return [], []
    title = getattr(corpus, "paper_title", lambda _pid: None)
    evidence = [
        {"passage_id": str(row["passage_id"]), "paper_id": str(row["paper_id"]),
         "title": title(str(row["paper_id"])), "text": _clean(row.get("text"), 400)}
        for row in items
    ]
    phrases: list[str] = []
    keyphrases = getattr(corpus, "keyphrases", None)
    papers = sorted({row["paper_id"] for row in evidence})
    if keyphrases is not None and papers:
        try:
            phrases = [str(row["phrase"]) for row in
                       keyphrases(scope={"paper_ids": papers}, limit=KEYPHRASES_SHOWN)["items"]]
        except Exception:  # noqa: BLE001
            phrases = []
    return evidence, phrases


def _ground_ranked(concept: Concept, corpus: object, shown: set[str]) -> None:
    """Supporting passages among the reading's top-ranked ones, plus the shown passages it cited."""
    phrases = [*concept.search_terms, concept.label]
    token_sets = [tokens for tokens in (_content_tokens(phrase) for phrase in phrases) if tokens]
    supporting: list[str] = []
    try:
        items = corpus.hybrid_rank(" ".join(phrases), limit=GROUNDING_DEPTH)["items"]  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - grounding is advisory, never fatal
        items = []
    for row in items:
        words = set(re.findall(r"[a-z0-9]+", str(row.get("text", "")).casefold()))
        if any(_supports(tokens, words) for tokens in token_sets):
            supporting.append(str(row["passage_id"]))
    cited = [pid for pid in concept.evidence_ids if pid in shown]
    concept.evidence_ids = cited
    concept.evidence_hits = len(set(supporting) | set(cited))
    concept.sample_sources = list(dict.fromkeys([*cited, *supporting]))[:5]


def _ground(concept: Concept, corpus: object | None) -> None:
    """Legacy: count corpus passages matching any of the concept's search phrases exactly."""
    search = getattr(corpus, "search", None)
    if search is None:
        return
    sources: list[str] = []
    total = 0
    for term in concept.search_terms:
        words = re.findall(r"\w+", term)
        if not words:
            continue
        query = '"' + " ".join(words) + '"' if len(words) > 1 else words[0]
        try:
            result = search(query, limit=3)
        except Exception:  # noqa: BLE001, S112 - grounding is advisory, never fatal
            continue
        total += int(result.get("total", 0))
        sources.extend(
            str(row.get("passage_id") or row.get("source_id"))
            for row in result.get("items", [])
            if row.get("passage_id") or row.get("source_id")
        )
    concept.evidence_hits = total
    concept.sample_sources = list(dict.fromkeys(sources))[:5]


def propose_grounded(
    model: object,
    question: str,
    allowed_classes: set[str],
    corpus: object | None = None,
    attempts: int = 2,
) -> ConceptProposal:
    """Ask the model for 3-4 concept readings and ground them in the corpus (see module doc)."""
    complete = getattr(model, "complete_json", None)
    if complete is None:
        raise ConceptProposalError("agent model cannot produce structured output")
    config = settings()
    minimum = max(MIN_CONCEPTS, config.readings_min)
    maximum = min(MAX_CONCEPTS, max(config.readings_max, MIN_CONCEPTS))
    retrieve_first = config.concept_grounding == "retrieve_first" and hasattr(corpus, "hybrid_rank")
    evidence, phrases = gather_evidence(corpus, question) if retrieve_first else ([], [])
    shown = {row["passage_id"] for row in evidence}
    last_error = "no concepts returned"
    request: dict[str, Any] = {"question": question, "allowed_classes": sorted(allowed_classes)}
    if retrieve_first:
        request["passages"] = evidence
        request["keyphrases"] = phrases
    for attempt in range(attempts):
        if attempt:
            # Tell the model what was wrong instead of repeating an identical
            # deterministic request.
            request["previous_attempt_problem"] = last_error
        prompt = render("agent/concept_proposal", dict(request), minimum=minimum, maximum=maximum)
        try:
            with model_purpose("concept proposal"), using_prompt(prompt.ref):
                concepts = _parse(complete(prompt.system, prompt.user), allowed_classes)
        except Exception as exc:  # noqa: BLE001 - provider errors are reported below
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if len(concepts) < minimum:
            last_error = f"agent proposed {len(concepts)} concepts; {minimum} required"
            continue
        for concept in concepts:
            if retrieve_first:
                _ground_ranked(concept, corpus, shown)
            else:
                _ground(concept, corpus)
        concepts.sort(key=lambda item: item.evidence_hits == 0)
        hidden: list[str] = []
        # Hiding needs ranked grounding; legacy mode keeps every reading, as before.
        if retrieve_first and config.hide_ungrounded_readings and any(item.evidence_hits for item in concepts):
            hidden = [item.label for item in concepts if not item.evidence_hits]
            concepts = [item for item in concepts if item.evidence_hits]
        return ConceptProposal(concepts[:maximum], evidence, phrases, hidden)
    raise ConceptProposalError(last_error)


def propose_concepts(
    model: object,
    question: str,
    allowed_classes: set[str],
    corpus: object | None = None,
    attempts: int = 2,
) -> list[Concept]:
    """The readings only (see ``propose_grounded``)."""
    return propose_grounded(model, question, allowed_classes, corpus, attempts).concepts


__all__ = ["Concept", "ConceptProposal", "ConceptProposalError", "gather_evidence", "propose_concepts",
           "propose_grounded"]
