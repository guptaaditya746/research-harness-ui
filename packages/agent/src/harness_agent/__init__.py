"""Agentic research path: an MCP tool-using agent over the shared evidence pipeline.

Modules:
- ``research``: ``AgenticResearch``, one run split into named steps.
- ``proposal``: concept proposal before research (the run pauses for the user's pick).
- ``concepts``: proposing and grounding concept readings.
- ``mcp``: the MCP client and the agent's tool loop (``MCPToolClient``).
- ``context``: fitting the agent conversation into the model's context window.
- ``research_state``: the research ledger (digests, repeat detection, loop hints).
- ``verification``: the independent fresh-context verifier.
- ``load``: Heavy-model load from the thesis-llm-setup status API.
- ``skills``: small research playbooks loaded into the supervisor prompt.
"""

from __future__ import annotations

from .mcp import MCPToolClient, MCPUnavailable
from .proposal import DEFAULT_CLASSES
from .research import NON_CORPUS_RECORDS, AgenticResearch

__all__ = ["DEFAULT_CLASSES", "NON_CORPUS_RECORDS", "AgenticResearch", "MCPToolClient", "MCPUnavailable"]
