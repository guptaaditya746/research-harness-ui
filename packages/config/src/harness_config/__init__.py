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

from dotenv import dotenv_values, find_dotenv
from pydantic import AliasChoices, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _clean(value: str | None) -> str | None:
    """``KEY=   # comment`` is an empty value (that is how the shell reads it)."""
    if value is None:
        return None
    return "" if value.lstrip().startswith("#") else value


def profile_file(name: str, base: Path | None = None) -> Path:
    """The file of a settings profile: a path, or a name under ``profiles/`` (``exhaustive`` ->
    ``profiles/exhaustive.env``) next to the ``.env`` in use, else under the working directory."""
    candidate = Path(name)
    if candidate.suffix == ".env" or len(candidate.parts) > 1:
        return candidate if candidate.is_absolute() else (base or Path.cwd()) / candidate
    return (base or Path.cwd()) / "profiles" / f"{name}.env"


def load_env_file(path: str | os.PathLike[str] | None = None) -> None:
    """Load a ``.env`` into ``os.environ`` without overriding variables that are already set.

    ``HARNESS_PROFILE`` (from the environment or the ``.env``) names a settings profile, e.g.
    ``exhaustive``; its values apply before the ``.env``'s. Precedence, highest first: the process
    environment, the profile, the ``.env``, the defaults in :class:`Settings`.
    """
    chosen = path if path is not None else find_dotenv(usecwd=True)
    values = {key: _clean(value) for key, value in dotenv_values(chosen).items()} if chosen else {}
    profile = (os.environ.get("HARNESS_PROFILE") or values.get("HARNESS_PROFILE") or "").strip()
    layers: list[dict[str, str | None]] = []
    if profile:
        file = profile_file(profile, Path(chosen).resolve().parent if chosen else None)
        if not file.is_file():
            raise ValueError(f"HARNESS_PROFILE={profile!r}: no profile file at {file}")
        layers.append({key: _clean(value) for key, value in dotenv_values(file).items()})
    layers.append(values)
    for layer in layers:
        for key, value in layer.items():
            if value is None or key in os.environ:
                continue
            os.environ[key] = value


load_env_file()


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
    # Optional served reranker alias (LiteLLM /rerank); empty = fused order only.
    model_rerank: str = Field("", validation_alias=_env("HARNESS_MODEL_RERANK"))
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
    # Ranked retrieval everywhere (pipeline, concept evidence, search_evidence, readers): lexical = BM25
    # (the pipeline also keeps its boolean candidates and relative threshold); hybrid = BM25 + dense, fused.
    retrieval_mode: Literal["lexical", "hybrid"] = Field("lexical", validation_alias=_env("HARNESS_RETRIEVAL_MODE"))
    # Reuse claims accepted in earlier runs as retrieval candidates (off: every run starts from the corpus).
    claim_reuse: bool = Field(False, validation_alias=_env("HARNESS_CLAIM_REUSE"))
    agentic_max_passages: int = Field(24, validation_alias=_env("HARNESS_AGENTIC_MAX_PASSAGES"))
    agentic_max_mcp_seeds: int = Field(12, validation_alias=_env("HARNESS_AGENTIC_MAX_MCP_SEEDS"))
    # Extraction and evidence-review batches sent to the model at the same time (1 = one after another).
    extraction_concurrency: int = Field(3, ge=1, le=16, validation_alias=_env("HARNESS_EXTRACTION_CONCURRENCY"))
    # What the agent hands to the pipeline: its findings (statements + evidence ids) or only ids.
    agent_handoff: Literal["findings", "ids"] = Field("findings", validation_alias=_env("HARNESS_AGENT_HANDOFF"))

    # ─── Agent loop ──────────────────────────────────────────────────────────
    agent_context_tokens: int | None = Field(
        None, validation_alias=_env("HARNESS_AGENT_CONTEXT_TOKENS"))
    agent_reply_tokens: int = Field(2048, validation_alias=_env("HARNESS_AGENT_REPLY_TOKENS"))
    agent_reflections: int = Field(1, validation_alias=_env("HARNESS_AGENT_REFLECTIONS"))
    # Context and tools of the agent's tool loop (Phase 3c).
    agent_toolset: Literal["curated", "all"] = Field("curated", validation_alias=_env("HARNESS_AGENT_TOOLSET"))
    agent_max_tool_calls: int = Field(12, validation_alias=_env("HARNESS_AGENT_MAX_TOOL_CALLS"))
    agent_keep_results: int = Field(3, validation_alias=_env("HARNESS_AGENT_KEEP_RESULTS"))
    agent_summary_compaction: bool = Field(True, validation_alias=_env("HARNESS_AGENT_SUMMARY_COMPACTION"))
    # Parallel paper readers after the supervisor's research (Phase 3d); 0 = off.
    subagent_readers: int = Field(3, validation_alias=_env("HARNESS_SUBAGENT_READERS"))
    subagent_passages: int = Field(6, validation_alias=_env("HARNESS_SUBAGENT_PASSAGES"))
    agent_first_tool_choice: Literal["auto", "required"] = Field(
        "required", validation_alias=_env("HARNESS_AGENT_FIRST_TOOL_CHOICE"))
    mcp_agent_timeout_s: float = Field(90, validation_alias=_env("HARNESS_MCP_AGENT_TIMEOUT_S"))
    readings_min: int = Field(3, validation_alias=_env("READINGS_MIN"))
    readings_max: int = Field(4, validation_alias=_env("READINGS_MAX"))
    concept_grounding: Literal["retrieve_first", "legacy"] = Field(
        "retrieve_first", validation_alias=_env("HARNESS_CONCEPT_GROUNDING"))
    hide_ungrounded_readings: bool = Field(True, validation_alias=_env("HARNESS_HIDE_UNGROUNDED_READINGS"))
    # Phase 4: invented classes are subclasses of an existing class ("subclass"), or not allowed at
    # all ("off": they become hrn:ResearchEntity). Invented properties always get a parent property.
    schema_new_classes: Literal["subclass", "off"] = Field(
        "subclass", validation_alias=_env("HARNESS_SCHEMA_NEW_CLASSES"))
    # Phase 4: the content core. "cskg": the CS-KG 2.0 subset (seed/cskg_ontology_2.0_subset.ttl) and
    # the harness terms declared equivalent to it (Method, Task, Material, OtherEntity, uses, solves, ...)
    # plus the paper-level addresses and evaluatesOn. "none": the seed before CS-KG (for comparisons).
    content_core: Literal["cskg", "none"] = Field("cskg", validation_alias=_env("HARNESS_CONTENT_CORE"))
    cskg_subset_path: str = Field("", validation_alias=_env("CSKG_SUBSET_PATH"))
    # Phase 4: add the DBLP backbone (authors, venue, citations among them) of the answer's papers
    # to the answer graph and export only the answer's subgraph in triples.ttl.
    dblp_backbone: bool = Field(True, validation_alias=_env("HARNESS_DBLP_BACKBONE"))
    # How much vocabulary schema inference proposes: "focused" (only what the answer needs) or "broad"
    # (everything the passages say about the question's topic, more invented terms).
    schema_scope: Literal["focused", "broad"] = Field("focused", validation_alias=_env("HARNESS_SCHEMA_SCOPE"))
    # How extraction shapes the graph: "standard", or "dense" (consistent entity names, relations
    # between research entities, compound statements split into facts that share entities).
    graph_linking: Literal["standard", "dense"] = Field("standard", validation_alias=_env("HARNESS_GRAPH_LINKING"))
    # An openly extracted fact whose relation maps onto no subschema property: "drop" it, or "keep" it
    # under a new relation (a sub-property of researchRelation/researchAttribute) named from its wording.
    unmapped_relations: Literal["drop", "keep"] = Field("drop", validation_alias=_env("HARNESS_UNMAPPED_RELATIONS"))
    # Relation mapping thresholds: lexical similarity, then the mapping model's stated confidence.
    map_lexical_min: float = Field(0.42, ge=0, le=1, validation_alias=_env("HARNESS_MAP_LEXICAL_MIN"))
    map_model_min: float = Field(0.7, ge=0, le=1, validation_alias=_env("HARNESS_MAP_MODEL_MIN"))
    # Passages per open-extraction call (smaller = more facts per passage, more calls).
    edc_batch_passages: int = Field(2, ge=1, le=16, validation_alias=_env("HARNESS_EDC_BATCH_PASSAGES"))
    # The External path: a separate repo (external-harness) run as a subprocess per question, e.g.
    # "uv run --project /srv/external-harness xh", with its YAML config. Empty = the path is off.
    external_command: str = Field("", validation_alias=_env("HARNESS_EXTERNAL_COMMAND"))
    external_config: str = Field("", validation_alias=_env("HARNESS_EXTERNAL_CONFIG"))

    # ─── Paths and datasets (None = the repository default) ──────────────────
    corpus_root: Path | None = Field(None, validation_alias=_env("CORPUS_ROOT"))
    artifact_root: Path | None = Field(None, validation_alias=_env("ARTIFACT_ROOT"))
    datasets_root: Path | None = Field(None, validation_alias=_env("DATASETS_ROOT"))
    fixtures_dir: Path | None = Field(None, validation_alias=_env("FIXTURES_DIR"))
    dblp_schema_path: str = Field("", validation_alias=_env("DBLP_SCHEMA_PATH"))
    # The question library in the UI. Default: the evaluation set, two categories only: corpus-wide
    # (nlp32-47) and ambiguous (nlp30, nlp48-53, nlp62-66). The earlier libraries
    # (question_library_2025.json, question_library_2025_b.json) also hold metadata questions.
    question_sets: str = Field("question_set_d.json", validation_alias=_env("HARNESS_QUESTION_SETS"))
    # Also offer gold rows and recent runs in the library (off: only the question sets above).
    question_extras: bool = Field(False, validation_alias=_env("HARNESS_QUESTION_EXTRAS"))
    include_fixtures: bool = Field(True, validation_alias=_env("HARNESS_INCLUDE_FIXTURES"))
    labelling_normalization_file: str | None = Field(
        None, validation_alias=_env("LABELLING_NORMALIZATION_FILE"))
    labelling_vocabulary_file: str | None = Field(
        None, validation_alias=_env("LABELLING_VOCABULARY_FILE"))
    # Evaluation: rules for the "entailed" level and the soft-match judge's cache.
    predicate_relations_file: str = Field("", validation_alias=_env("HARNESS_PREDICATE_RELATIONS_FILE"))
    judge_cache_file: str = Field("", validation_alias=_env("HARNESS_JUDGE_CACHE_FILE"))

    # ─── Evidence documents (D): one exhaustive, reviewed document per question ─
    # Built once per question set by `make evidence-docs`; see docs/EVIDENCE_DOCS.md.
    doc_question_set: str = Field("question_set_d.json", validation_alias=_env("HARNESS_DOC_QUESTION_SET"))
    doc_root: Path | None = Field(None, validation_alias=_env("HARNESS_DOC_ROOT"))
    # Papers screened per question: 0 = every indexed paper.
    doc_max_papers: int = Field(0, ge=0, validation_alias=_env("HARNESS_DOC_MAX_PAPERS"))
    # Passages of one paper the screening reader sees (highest-ranked for the question first; a paper
    # without a ranked passage is read from its first passages, so every paper is actually read).
    doc_screen_passages: int = Field(8, ge=1, validation_alias=_env("HARNESS_DOC_SCREEN_PASSAGES"))
    # Passages of a relevant paper that go to extraction: 0 = all of the paper's passages.
    doc_extract_passages: int = Field(0, ge=0, validation_alias=_env("HARNESS_DOC_EXTRACT_PASSAGES"))
    # Passages schema inference sees for one interpretation.
    doc_schema_passages: int = Field(24, ge=1, validation_alias=_env("HARNESS_DOC_SCHEMA_PASSAGES"))
    # Papers screened or extracted at the same time.
    doc_concurrency: int = Field(4, ge=1, le=32, validation_alias=_env("HARNESS_DOC_CONCURRENCY"))
    # Model call timeout for document building (no run deadline applies; building is offline).
    doc_call_timeout_s: float = Field(300, gt=0, validation_alias=_env("HARNESS_DOC_CALL_TIMEOUT_S"))

    # ─── Evaluation snapshot (must match labelling's gold rows) ──────────────
    corpus_fingerprint: str = Field("unknown", validation_alias=_env("CORPUS_FINGERPRINT"))
    corpus_snapshot_id: str | None = Field(None, validation_alias=_env("CORPUS_SNAPSHOT_ID"))
    ontology_version: str = Field("unknown", validation_alias=_env("ONTOLOGY_VERSION"))
    ontology_hash: str = Field("unknown", validation_alias=_env("ONTOLOGY_HASH"))
    retrieval_index_version: str = Field("unknown", validation_alias=_env("RETRIEVAL_INDEX_VERSION"))

    # The settings profile in use (applied by load_env_file); empty = none.
    profile: str = Field("", validation_alias=_env("HARNESS_PROFILE"))

    # ─── Observability ───────────────────────────────────────────────────────
    harness_env: str = Field("development", validation_alias=_env("HARNESS_ENV"))
    langfuse_base_url: str = Field("", validation_alias=_env("LANGFUSE_BASE_URL"))
    langfuse_public_key: str = Field("", validation_alias=_env("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str = Field("", validation_alias=_env("LANGFUSE_SECRET_KEY"))

    @field_validator(
        "model_edc", "model_agent_url", "agent_context_tokens", "corpus_root", "artifact_root",
        "datasets_root", "fixtures_dir", "database_url", "corpus_snapshot_id",
        "labelling_normalization_file", "labelling_vocabulary_file", "doc_root", mode="before",
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
        dynamically because the variable name depends on the path. The External
        path skips the budget-wide variable: its limit is a safety net above the
        agent's own wall time (external config, 900 s by default), and a shared
        HARNESS_RUN_DEADLINE_SLOW_S=900 would stop it before the agent writes its result.
        """
        default = 300.0 if (path == "agentic" and budget == "slow") else {"fast": 300.0, "slow": 900.0}[budget]
        path_key = f"HARNESS_RUN_DEADLINE_{path.upper().replace('-', '_')}_{budget.upper()}_S"
        if path == "external":
            return float(os.environ.get(path_key, "960"))
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


__all__ = ["DYNAMIC_VARIABLES", "Settings", "load_env_file", "profile_file", "settings"]
