"""Errors raised by model calls."""

from __future__ import annotations


class ModelFailure(RuntimeError):
    """A model call failed: provider error, timeout, stop request or unusable reply."""


class UnsupportedAnswer(ValueError):
    """The composed answer is not backed by the extracted triples (run reason ``answer_unsupported``).

    Raised when the answer cites a source outside the evidence, or the verifier removed every claim,
    or the verified answer has no valid provenance. Before Phase 4 this surfaced as ``schema_rejected``.
    """


__all__ = ["ModelFailure", "UnsupportedAnswer"]
