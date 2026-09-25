"""The soft-match judge: "do these two triples state the same fact?", with a cache.

The evaluation package never calls a model itself. A caller that wants live
judgements passes ``complete``: a function that sends a rendered prompt and
returns the parsed JSON object (``harness_extraction.complete`` bound to a
model does exactly that). Without ``complete`` the judge only answers from its
cache and returns ``None`` for pairs it has not seen, which the scorer reports
as ``judge.pending``. The Benchmark page uses the cache-only mode, so opening
it never starts model calls; ``make evaluate ARGS="--judge"`` fills the cache.

Every decision is stored with the prompt reference (``name@set:hash``), so a
changed prompt never reuses old answers, and re-scoring a run gives the same
number.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from harness_config import settings
from harness_prompts import Prompt, render

PROMPT_NAME = "evaluation/triple_match"
LOGGER = logging.getLogger(__name__)


def default_cache_path() -> Path:
    configured = settings().judge_cache_file.strip()
    if configured:
        return Path(configured).expanduser()
    root = settings().artifact_root or Path("artifacts")
    return Path(root) / "judge_cache.jsonl"


def _fact(item: Mapping[str, Any]) -> dict[str, str]:
    return {"subject": str(item.get("s", "")), "relation": str(item.get("p", "")), "object": str(item.get("o", ""))}


def parse_verdict(answer: Mapping[str, Any] | None) -> bool | None:
    if not isinstance(answer, Mapping):
        return None
    value = answer.get("same_fact")
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().casefold() in {"true", "yes", "false", "no"}:
        return value.strip().casefold() in {"true", "yes"}
    return None


class CachedJudge:
    """Callable ``(gold, run) -> bool | None`` backed by a JSONL cache."""

    def __init__(self, cache_path: Path | None = None,
                 complete: Callable[[Prompt], Mapping[str, Any]] | None = None,
                 *, prompt_set: str | None = None) -> None:
        self.cache_path = cache_path or default_cache_path()
        self.complete = complete
        self.prompt_set = prompt_set
        self.lock = threading.Lock()
        self.calls = 0
        self._cache: dict[str, bool] = {}
        if self.cache_path.exists():
            for number, line in enumerate(self.cache_path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    self._cache[str(row["key"])] = bool(row["same_fact"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    LOGGER.warning("Skipping unreadable line %d in %s", number, self.cache_path)

    def prompt(self, gold: Mapping[str, Any], run: Mapping[str, Any]) -> Prompt:
        return render(PROMPT_NAME, {"reference": _fact(gold), "candidate": _fact(run)},
                      prompt_set=self.prompt_set)

    def __call__(self, gold: Mapping[str, Any], run: Mapping[str, Any]) -> bool | None:
        prompt = self.prompt(gold, run)
        key = hashlib.sha256(f"{prompt.ref}\x00{prompt.user}".encode()).hexdigest()
        with self.lock:
            if key in self._cache:
                return self._cache[key]
        if self.complete is None:
            return None
        try:
            answer = self.complete(prompt)
        except Exception as exc:  # noqa: BLE001 - a failed judgement is "pending", never a lost run
            LOGGER.warning("Judge call failed (%s: %s); the pair stays pending", type(exc).__name__, exc)
            return None
        verdict = parse_verdict(answer)
        if verdict is None:
            return None
        row = {"key": key, "same_fact": verdict, "prompt": prompt.ref, "gold": _fact(gold), "run": _fact(run),
               "reason": str((answer or {}).get("reason", ""))[:300]}
        with self.lock:
            self.calls += 1
            self._cache[key] = verdict
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return verdict


__all__ = ["PROMPT_NAME", "CachedJudge", "default_cache_path", "parse_verdict"]
