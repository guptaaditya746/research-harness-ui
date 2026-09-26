"""Errors raised by model calls."""

from __future__ import annotations


class ModelFailure(RuntimeError):
    """A model call failed: provider error, timeout, stop request or unusable reply."""


class UnsupportedAnswer(ValueError):
    """The composed answer is not backed by the extracted triples (run reason ``answer_unsupported``).

    Raised when the answer cites a source outside the evidence (``stage="citation"``), the verifier
    removed every claim (``"verification"``) or the verified answer has no valid provenance
    (``"provenance"``). Before Phase 4 this surfaced as ``schema_rejected``.
    """

    def __init__(self, message: str, stage: str = "verification", removed: list[str] | None = None,
                 sentences: list[dict] | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.removed = list(removed or [])          # the claims the verifier removed
        self.sentences = list(sentences or [])      # its per-sentence verdicts, when it gave them


__all__ = ["ModelFailure", "UnsupportedAnswer"]
