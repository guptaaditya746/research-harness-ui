"""The exact search language (phrases, AND/OR/NOT, NEAR/n) and time-bounded model-written regexes."""

from __future__ import annotations

import re
import time

import regex as safe_regex

PATTERN_MAX_LENGTH = 300
PATTERN_BUDGET_S = 2.0



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
    """Evaluate phrase, boolean, and proximity terms against one passage."""
    tokens = lex(dsl)
    words = re.findall(r"\w+", text.casefold())
    haystack = " ".join(words)
    pos = 0

    def term(value: str) -> bool:
        phrase = " ".join(re.findall(r"\w+", value.casefold()))
        return bool(phrase) and phrase in haystack

    def near(left: str, right: str, distance: int) -> bool:
        a = re.findall(r"\w+", left.casefold())
        b = re.findall(r"\w+", right.casefold())
        if not a or not b:
            return False
        left_pos = [i for i in range(len(words)) if words[i : i + len(a)] == a]
        right_pos = [i for i in range(len(words)) if words[i : i + len(b)] == b]
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
