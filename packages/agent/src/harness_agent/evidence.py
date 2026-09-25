"""Evidence handles found in tool results and run records."""

from __future__ import annotations


def evidence_ids(value: object) -> set[str]:
    """Every ``*_id`` / ``*_ids`` value nested anywhere in ``value``."""
    if isinstance(value, dict):
        found = {
            str(item) for key, item in value.items()
            if key.endswith("_id") and isinstance(item, (str, int))
        }
        for key, item in value.items():
            if key.endswith("_ids") and isinstance(item, list):
                found.update(str(entry) for entry in item if isinstance(entry, (str, int)))
        for item in value.values():
            found.update(evidence_ids(item))
        return found
    if isinstance(value, list):
        return set().union(*(evidence_ids(item) for item in value)) if value else set()
    return set()


__all__ = ["evidence_ids"]
