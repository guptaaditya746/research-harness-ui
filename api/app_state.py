"""The process-wide run runtime. Routes read ``app_state.runtime`` at call time,
so tests can replace it with ``monkeypatch.setattr(app_state, "runtime", ...)``."""

from __future__ import annotations

from paths import artifact_root, datasets_root
from runtime import RunRuntime

runtime = RunRuntime(artifact_root(), datasets_root())
