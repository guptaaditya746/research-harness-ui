"""The exact search language (phrases, AND/OR/NOT, NEAR/n) and time-bounded model-written regexes.

Words, not substrings: text and query are split into lowercase ``\\w+`` runs, a term matches a whole
word and a quoted phrase matches consecutive words. This is the BM25 tokenizer's split (bm25s:
``lower()``, ``\\b\\w\\w+\\b``) without its two-character minimum and stopword removal, so an exact
query can still say ``"GPT 4"`` or ``"state of the art"``, while every hit on a term that has a
non-stopword word of two or more characters shares that token with the BM25 index, so it ranks with a
positive score (before, ``graph`` matched ``paragraph`` as a substring and then scored 0). There is no
stemming in either: ``graph`` does not match ``graphs``.
"""

from __future__ import annotations

import re
import time

import regex as safe_regex
from bm25s.stopwords import STOPWORDS_EN

PATTERN_MAX_LENGTH = 300
PATTERN_BUDGET_S = 2.0
WORD = re.compile(r"\w+")
BM25_TOKEN = re.compile(r"(?u)\b\w\w+\b")      # bm25s' default token_pattern
BM25_STOPWORDS = frozenset(STOPWORDS_EN)       # bm25s.tokenize(..., stopwords="en")


def words(text: str) -> list[str]:
    """The exact search language's words: lowercase ``\\w+`` runs."""
    return WORD.findall(text.lower())


def bm25_terms(text: str) -> list[str]:
    """The tokens BM25 sees for ``text``: exactly ``bm25s.tokenize(text, stopwords="en")`` (lowercase,
    two or more word characters, English stopwords removed, no stemming)."""
    return [token for token in BM25_TOKEN.findall(text.lower()) if token not in BM25_STOPWORDS]


class BoundedPattern:
    """A model-written regex with a total time budget per tool call.

    Tool callers are language models; a pathological pattern (catastrophic
    backtracking) would otherwise block the single-threaded tool server for
    every later run. The ``regex`` engine supports per-search timeouts.
    """

    def __init__(self, pattern: str, budget_s: float = PATTERN_BUDGET_S) -> None:
        if len(pattern) > PATTERN_MAX_LENGTH:
            raise ValueError(f"pattern is longer than {PATTERN_MAX_LENGTH} characters; simplify it")
        try:
            self.compiled = safe_regex.compile(pattern, safe_regex.IGNORECASE)
        except safe_regex.error as exc:
            raise ValueError(f"invalid pattern: {exc}") from exc
        self.deadline = time.monotonic() + budget_s

    def search(self, text: str):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("pattern search exceeded its time budget; use a simpler pattern")
        try:
            return self.compiled.search(text, timeout=remaining)
        except TimeoutError as exc:
            raise ValueError("pattern search exceeded its time budget; use a simpler pattern") from exc


def lex(dsl: str) -> list[str]:
    tokens = re.findall(
        r'"[^"\n]+"|\(|\)|\bNEAR/\d+\b|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+', dsl, re.IGNORECASE
    )
    if not tokens:
        raise ValueError("empty search")
    return tokens


def matches(dsl: str, text: str) -> bool:
    """Evaluate phrase, boolean, and proximity terms against one passage (whole words, see above)."""
    tokens = lex(dsl)
    text_words = words(text)
    vocabulary = set(text_words)
    pos = 0

    def positions(sequence: list[str]) -> list[int]:
        if not sequence or not set(sequence) <= vocabulary:
            return []
        size = len(sequence)
        return [i for i in range(len(text_words) - size + 1) if text_words[i : i + size] == sequence]

    def term(value: str) -> bool:
        sequence = words(value)
        if len(sequence) == 1:
            return sequence[0] in vocabulary
        return bool(positions(sequence))

    def near(left: str, right: str, distance: int) -> bool:
        a, b = words(left), words(right)
        left_pos, right_pos = positions(a), positions(b)
        return any(0 <= j - (i + len(a)) <= distance for i in left_pos for j in right_pos)

    def atom() -> bool:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("incomplete search expression")
        token = tokens[pos]
        pos += 1
        if token.upper() == "NOT":
            return not atom()
        if token == "(":
            value = expression()
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError("unclosed parenthesis")
            pos += 1
            return value
        if token == ")":
            raise ValueError("unexpected parenthesis")
        if pos + 1 < len(tokens) and re.fullmatch(r"NEAR/\d+", tokens[pos], re.IGNORECASE):
            distance = int(tokens[pos].split("/")[1])
            other = tokens[pos + 1]
            pos += 2
            return near(token, other, distance)
        return term(token)

    def conjunction() -> bool:
        nonlocal pos
        value = atom()
        while pos < len(tokens) and tokens[pos] != ")" and tokens[pos].upper() != "OR":
            if tokens[pos].upper() == "AND":
                pos += 1
            other = atom()
            value = value and other
        return value

    def expression() -> bool:
        nonlocal pos
        value = conjunction()
        while pos < len(tokens) and tokens[pos].upper() == "OR":
            pos += 1
            other = conjunction()
            value = value or other
        return value

    result = expression()
    if pos != len(tokens):
        raise ValueError("unexpected search token")
    return result
