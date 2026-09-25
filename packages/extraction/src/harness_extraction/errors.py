"""Errors raised by model calls."""

from __future__ import annotations


class ModelFailure(RuntimeError):
    """A model call failed: provider error, timeout, stop request or unusable reply."""


__all__ = ["ModelFailure"]
