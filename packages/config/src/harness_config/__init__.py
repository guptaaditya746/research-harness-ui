"""Every setting the harness reads from the environment, in one typed place.

Call :func:`settings` where a value is needed. It reads the current process
environment each time (no caching), so a changed variable takes effect on
the next call and tests can use ``monkeypatch.setenv``. On import, the
nearest ``.env`` (searched from the working directory upwards) is loaded
into the environment without overriding variables that are already set, so
every entry point (API, MCP server, scripts) sees the same configuration.

Adding a setting: add a field below with its default, document it in
``.env.example``, and read it through ``settings().<field>``. The test in
``tests/test_config.py`` fails when ``.env.example`` names a variable that
has no field here.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import AliasChoices, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(find_dotenv(usecwd=True), override=False)


def _env(name: str, *more: str) -> Any:
    """Field alias: the environment variable name(s) a field is read from."""
    return AliasChoices(name, *more)


class Settings(BaseSettings):
    """Harness settings. Field names are Python-friendly; aliases are the variables."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=True, env_ignore_empty=True, frozen=True)

    # ─── Models (aliases served by LiteLLM, see thesis-llm-setup) ────────────
    model_small: str = Field("heavy-model", validation_alias=_env("HARNESS_MODEL_SMALL"))
    model_edc: str | None = Field(None, validation_alias=_env("HARNESS_MODEL_EDC"))
    model_agent: str = Field("lite-model", validation_alias=_env("HARNESS_MODEL_AGENT"))
    model_high: str = Field("heavy-model", validation_alias=_env("HARNESS_MODEL_HIGH"))
    model_judge: str = Field("heavy-model", validation_alias=_env("HARNESS_MODEL_JUDGE"))
    model_embedding: str = Field("qwen-embed", validation_alias=_env("HARNESS_MODEL_EMBEDDING"))
    model_agent_url: str | None = Field(None, validation_alias=_env("HARNESS_MODEL_AGENT_URL"))
    model_options_json: str = Field("", validation_alias=_env("HARNESS_MODEL_OPTIONS"))
    max_output_tokens: int = Field(4096, validation_alias=_env("HARNESS_MAX_OUTPUT_TOKENS"))
    model_call_threads: int = Field(16, validation_alias=_env("HARNESS_MODEL_CALL_THREADS"))
    model_stream: bool = Field(False, validation_alias=_env("HARNESS_MODEL_STREAM"))
    provider_max_retries: int = Field(3, validation_alias=_env("PROVIDER_MAX_RETRIES"))
    prompt_set: str = Field("v2", validation_alias=_env("HARNESS_PROMPT_SET"))

    # ─── Endpoints ───────────────────────────────────────────────────────────
    litellm_base_url: str = Field("http://127.0.0.1:4000/v1", validation_alias=_env("LITELLM_BASE_URL"))
    api_key: str = Field("local", validation_alias=_env("VLLM_API_KEY"))
    llm_status_url: str = Field(
        "http://127.0.0.1:8010/v1/status", validation_alias=_env("LLM_STATUS_URL")
    )
    vllm_heavy_api_base: str = Field("", validation_alias=_env("VLLM_HEAVY_API_BASE"))
    mcp_corpus_url: str = Field("http://127.0.0.1:8765/mcp", validation_alias=_env("MCP_CORPUS_URL"))
    mcp_corpus_host: str = Field("127.0.0.1", validation_alias=_env("MCP_CORPUS_HOST"))
    mcp_corpus_port: int = Field(8765, validation_alias=_env("MCP_CORPUS_PORT"))
    mcp_corpus_transport: str = Field("streamable-http", validation_alias=_env("MCP_CORPUS_TRANSPORT"))
    grobid_url: str = Field("http://127.0.0.1:8070", validation_alias=_env("GROBID_URL"))
    database_url: str | None = Field(None, validation_alias=_env("DATABASE_URL"))
    api_cors_origins: str = Field("http://localhost:5173", validation_alias=_env("API_CORS_ORIGINS"))
    sse_keepalive_seconds: int = Field(15, validation_alias=_env("SSE_KEEPALIVE_SECONDS"))

    # ─── Budgets and limits ──────────────────────────────────────────────────
    budget_baseline_fast_seconds: float = Field(
        60, validation_alias=_env("BUDGET_BASELINE_FAST_SECONDS"))
    budget_baseline_slow_seconds: float = Field(
        90, validation_alias=_env("BUDGET_BASELINE_SLOW_SECONDS"))
    budget_agentic_fast_seconds: float = Field(
        45, validation_alias=_env("BUDGET_AGENTIC_FAST_SECONDS"))
    budget_agentic_slow_seconds: float = Field(
        120, validation_alias=_env("BUDGET_AGENTIC_SLOW_SECONDS"))
    max_concurrent_runs: int = Field(2, validation_alias=_env("HARNESS_MAX_CONCURRENT_RUNS"))
    max_concurrent_heavy: int = Field(1, validation_alias=_env("HARNESS_MAX_CONCURRENT_HEAVY"))
    max_concurrent_public_runs: int = Field(
        1, validation_alias=_env("HARNESS_MAX_CONCURRENT_PUBLIC_RUNS"))
    heavy_queue_threshold: int = Field(8, validation_alias=_env("HARNESS_HEAVY_QUEUE_THRESHOLD"))
    baseline_retrieve_k: int = Field(20, validation_alias=_env("BASELINE_RETRIEVE_K"))
    adaptive_max_passages: int = Field(16, validation_alias=_env("HARNESS_ADAPTIVE_MAX_PASSAGES"))
    agentic_max_passages: int = Field(12, validation_alias=_env("HARNESS_AGENTIC_MAX_PASSAGES"))
    agentic_max_mcp_seeds: int = Field(6, validation_alias=_env("HARNESS_AGENTIC_MAX_MCP_SEEDS"))

    # ─── Agent loop ──────────────────────────────────────────────────────────
    agent_context_tokens: int | None = Field(
        None, validation_alias=_env("HARNESS_AGENT_CONTEXT_TOKENS"))
    agent_reply_tokens: int = Field(2048, validation_alias=_env("HARNESS_AGENT_REPLY_TOKENS"))
    agent_reflections: int = Field(1, validation_alias=_env("HARNESS_AGENT_REFLECTIONS"))
    agent_first_tool_choice: Literal["auto", "required"] = Field(
        "required", validation_alias=_env("HARNESS_AGENT_FIRST_TOOL_CHOICE"))
    mcp_agent_timeout_s: float = Field(90, validation_alias=_env("HARNESS_MCP_AGENT_TIMEOUT_S"))
    readings_min: int = Field(3, validation_alias=_env("READINGS_MIN"))
    readings_max: int = Field(4, validation_alias=_env("READINGS_MAX"))

    # ─── Paths and datasets (None = the repository default) ──────────────────
    corpus_root: Path | None = Field(None, validation_alias=_env("CORPUS_ROOT"))
    artifact_root: Path | None = Field(None, validation_alias=_env("ARTIFACT_ROOT"))
    datasets_root: Path | None = Field(None, validation_alias=_env("DATASETS_ROOT"))
    fixtures_dir: Path | None = Field(None, validation_alias=_env("FIXTURES_DIR"))
    dblp_schema_path: str = Field("", validation_alias=_env("DBLP_SCHEMA_PATH"))
    question_sets: str = Field(
        "question_matrix_20.json,question_library_2025.json,corpus_10.json",
        validation_alias=_env("HARNESS_QUESTION_SETS"),
    )
    include_fixtures: bool = Field(True, validation_alias=_env("HARNESS_INCLUDE_FIXTURES"))
    labelling_normalization_file: str | None = Field(
        None, validation_alias=_env("LABELLING_NORMALIZATION_FILE"))
    labelling_vocabulary_file: str | None = Field(
        None, validation_alias=_env("LABELLING_VOCABULARY_FILE"))
    # Evaluation: rules for the "entailed" level and the soft-match judge's cache.
    predicate_relations_file: str = Field("", validation_alias=_env("HARNESS_PREDICATE_RELATIONS_FILE"))
    judge_cache_file: str = Field("", validation_alias=_env("HARNESS_JUDGE_CACHE_FILE"))

    # ─── Evaluation snapshot (must match labelling's gold rows) ──────────────
    corpus_fingerprint: str = Field("unknown", validation_alias=_env("CORPUS_FINGERPRINT"))
    corpus_snapshot_id: str | None = Field(None, validation_alias=_env("CORPUS_SNAPSHOT_ID"))
    ontology_version: str = Field("unknown", validation_alias=_env("ONTOLOGY_VERSION"))
    ontology_hash: str = Field("unknown", validation_alias=_env("ONTOLOGY_HASH"))
    retrieval_index_version: str = Field("unknown", validation_alias=_env("RETRIEVAL_INDEX_VERSION"))

    # ─── Observability ───────────────────────────────────────────────────────
    harness_env: str = Field("development", validation_alias=_env("HARNESS_ENV"))
    langfuse_base_url: str = Field("", validation_alias=_env("LANGFUSE_BASE_URL"))
    langfuse_public_key: str = Field("", validation_alias=_env("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str = Field("", validation_alias=_env("LANGFUSE_SECRET_KEY"))

    @field_validator(
        "model_edc", "model_agent_url", "agent_context_tokens", "corpus_root", "artifact_root",
        "datasets_root", "fixtures_dir", "database_url", "corpus_snapshot_id",
        "labelling_normalization_file", "labelling_vocabulary_file", mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: Any) -> Any:
        """``VAR=`` in .env means "not set", not an empty string or zero."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("include_fixtures", mode="before")
    @classmethod
    def _on_unless_disabled(cls, value: Any) -> Any:
        """On unless explicitly 0/false/no (the API's historical rule)."""
        if isinstance(value, str):
            return value.strip().casefold() not in {"0", "false", "no"}
        return value

    @field_validator("model_stream", mode="before")
    @classmethod
    def _off_unless_enabled(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().casefold() in {"1", "true", "yes", "on"}
        return value

    def model_options(self, alias: str) -> dict[str, Any]:
        """Per-alias request options from HARNESS_MODEL_OPTIONS (JSON); {} when absent or invalid."""
        raw = self.model_options_json.strip()
        if not raw:
            return {}
        try:
            options = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        value = options.get(alias) if isinstance(options, dict) else None
        return value if isinstance(value, dict) else {}

    def run_deadline_seconds(self, path: str, budget: str) -> float:
        """Hard wall-clock limit for one run.

        Most specific first: HARNESS_RUN_DEADLINE_<PATH>_<BUDGET>_S, then
        HARNESS_RUN_DEADLINE_<BUDGET>_S, then the built-in default. Read
        dynamically because the variable name depends on the path.
        """
        default = 300.0 if (path == "agentic" and budget == "slow") else {"fast": 300.0, "slow": 900.0}[budget]
        path_key = f"HARNESS_RUN_DEADLINE_{path.upper().replace('-', '_')}_{budget.upper()}_S"
        return float(os.environ.get(path_key, os.environ.get(f"HARNESS_RUN_DEADLINE_{budget.upper()}_S", default)))


_CACHE: tuple[dict[str, str], Settings] | None = None
_CACHE_LOCK = threading.Lock()


def settings() -> Settings:
    """The current settings, read from the process environment.

    Rebuilt whenever the environment changes (so ``monkeypatch.setenv`` and
    runtime changes take effect), otherwise reused: building ``Settings``
    costs about a millisecond, comparing the environment a few microseconds.
    An invalid value raises ``ValueError`` naming the variable.
    """
    global _CACHE
    environment = dict(os.environ)
    cached = _CACHE
    if cached is not None and cached[0] == environment:
        return cached[1]
    try:
        current = Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{error['loc'][0] if error['loc'] else '?'}: {error['msg']}" for error in exc.errors())
        raise ValueError(f"invalid harness setting ({problems}); check .env and .env.example") from None
    with _CACHE_LOCK:
        _CACHE = (environment, current)
    return current


# Variables that are read outside this class on purpose.
DYNAMIC_VARIABLES = ("HARNESS_RUN_DEADLINE_",)


__all__ = ["DYNAMIC_VARIABLES", "Settings", "settings"]
