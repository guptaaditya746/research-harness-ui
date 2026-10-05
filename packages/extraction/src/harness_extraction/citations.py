"""Citations in composed answers: which bracket groups cite sources, and dropping unknown ones."""

from __future__ import annotations

import re

# Source ids are lowercase slugs with a letter and a digit ("p4-abstract", "p12-s3-p2", "2501-01234-metadata");
# research_notes is part of the composer's input but not a source. Other brackets are prose ("[0.5-0.7]",
# "[CLS]", "[GPT-4]", "[h_t]").
_ID_LIKE = re.compile(r"(?=[a-z0-9_-]*[a-z])(?=[a-z0-9_-]*\d)[a-z0-9]+(?:[-_][a-z0-9]+)*")
_NOT_SOURCES = {"research_notes"}
# A sentence ends at . ! or ? followed by a space, but not after "et al.", "e.g.", "i.e.", "vs.", "Fig." and the like.
_SENTENCE_END = re.compile(r"(?<!\bet al\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<!\bFig\.)(?<!\bSec\.)"
                           r"(?<!\bEq\.)(?<!\bNo\.)(?<!\bU\.S\.)(?<=[.!?])\s+")
_BRACKET = re.compile(r"\[([^\]]+)\]")


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
