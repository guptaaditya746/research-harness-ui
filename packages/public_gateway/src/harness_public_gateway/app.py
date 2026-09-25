from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import threading
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PublicStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=8, max_length=500)
    path: str = "fast"
    turnstileToken: str = Field(min_length=1, max_length=2048)

    @field_validator("question")
    @classmethod
    def meaningful(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 8:
            raise ValueError("question must contain at least 8 characters")
        return value


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PublicStartResponse(PublicModel):
    publicRunId: str
    state: Literal[
        "queued", "running", "awaiting_reading", "answered", "incomplete", "stopped", "rejected"
    ]
    queuePosition: int


class PublicProgress(PublicModel):
    atSeconds: float
    message: str


class PublicField(PublicModel):
    value: str


class PublicClaim(PublicModel):
    kind: str
    fields: dict[str, PublicField]
    evidence: list[str]
    evidenceAnchors: list[dict[str, str]]


class PublicTriple(PublicModel):
    s: str
    p: str
    o: str
    source: str
    evidence: str
    support: str | None = None


class PublicGraphNode(PublicModel):
    id: str
    label: str
    kind: str


class PublicGraphEdge(PublicModel):
    source: str
    target: str
    label: str
    kind: str
    evidence: str | None = None


class PublicGraph(PublicModel):
    nodes: list[PublicGraphNode]
    edges: list[PublicGraphEdge]


class PublicReading(PublicModel):
    label: str | None = None
    gloss: str | None = None


class PublicRunResponse(PublicModel):
    publicRunId: str
    question: str
    path: Literal["fast", "adaptive", "agentic"]
    state: Literal[
        "queued", "running", "awaiting_reading", "answered", "incomplete", "stopped", "rejected"
    ]
    progress: list[PublicProgress]
    answer: str | None = None
    evidence: list[PublicClaim]
    triples: list[PublicTriple]
    graph: PublicGraph
    reading: PublicReading | None = None
    failure: Literal["service busy", "run timed out"] | None = None


class PublicQueue(PublicModel):
    waiting: int
    capacity: int


class PublicPaths(PublicModel):
    fast: bool
    adaptive: bool
    agentic: bool


class PublicStatusResponse(PublicModel):
    state: Literal["available", "busy", "saturated", "unavailable", "recovering"]
    acceptingRuns: bool
    queue: PublicQueue
    paths: PublicPaths
    message: str


class PublicExample(PublicModel):
    question: str
    answer: str
    evidence: list[PublicClaim]
    triples: list[PublicTriple]
    graph: PublicGraph


class HumanVerifier(Protocol):
    def verify(self, token: str, remote_ip: str | None = None) -> bool: ...


class TurnstileVerifier:
    def __init__(self, secret: str, enabled: bool = True, local_bypass: bool = False):
        if enabled and local_bypass:
            raise ValueError(
                "development verifier bypass is forbidden when verification is enabled"
            )
        self.secret, self.enabled, self.local_bypass = secret, enabled, local_bypass

    def verify(self, token: str, remote_ip: str | None = None) -> bool:
        if not self.enabled:
            return self.local_bypass
        if not self.secret:
            return False
        try:
            response = httpx.post(
                "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                data={"secret": self.secret, "response": token},
                timeout=4.0,
            )
            return response.status_code == 200 and response.json().get("success") is True
        except (httpx.HTTPError, ValueError):
            return False


class PrivateClient(Protocol):
    def reserve(self) -> str: ...
    def start(
        self, question: str, path: str, budget: str, deadline_s: int, reservation: str
    ) -> str: ...
    def get(self, run_id: str) -> dict[str, Any]: ...
    def can_start(self) -> bool: ...
    def ready(self) -> bool: ...


class LoopbackPrivateClient:
    """Exact-path client. IDs are URL-encoded and no caller path is accepted."""

    def __init__(self, origin: str):
        parsed = urllib.parse.urlparse(origin)
        try:
            is_loopback = bool(
                parsed.hostname and ipaddress.ip_address(parsed.hostname).is_loopback
            )
        except ValueError:
            is_loopback = False
        if (
            parsed.scheme != "http"
            or not is_loopback
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("private origin must be an HTTP loopback origin")
        self.origin = origin.rstrip("/")

    def reserve(self) -> str:
        response = httpx.post(self.origin + "/internal/public/reservations", timeout=2.0)
        response.raise_for_status()
        return str(response.json()["reservation"])

    def start(
        self, question: str, path: str, budget: str, deadline_s: int, reservation: str
    ) -> str:
        response = httpx.post(
            self.origin + "/runs",
            json={
                "question": question,
                "path": path,
                "budget": budget,
                "origin": "public",
                "deadline_seconds": deadline_s,
            },
            headers={"X-Public-Reservation": reservation},
            timeout=5.0,
        )
        response.raise_for_status()
        return str(response.json()["id"])

    def get(self, run_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"r-[a-f0-9]{8}", run_id):
            raise ValueError("invalid private run identifier")
        return httpx.get(
            self.origin + "/runs/" + urllib.parse.quote(run_id, safe=""), timeout=5.0
        ).json()

    def can_start(self) -> bool:
        response = httpx.get(self.origin + "/internal/public/capacity", timeout=2.0)
        response.raise_for_status()
        return response.json().get("accepting") is True

    def ready(self) -> bool:
        response = httpx.get(self.origin + "/internal/public/capacity", timeout=2.0)
        return response.status_code == 200 and response.json().get("available") is True


class StatusProvider(Protocol):
    def available(self) -> bool: ...
    def capacity_available(self) -> bool: ...


class BasicStatus:
    def __init__(self, private: PrivateClient):
        self.private = private

    def available(self) -> bool:
        try:
            return self.private.ready()
        except Exception:  # noqa: BLE001 - status is deliberately generic
            return False

    def capacity_available(self) -> bool:
        try:
            return self.private.can_start()
        except Exception:  # noqa: BLE001 - status is deliberately generic
            return False


@dataclass
class Entry:
    public_id: str
    client_key: str
    question: str
    path: str
    state: str = "queued"
    private_id: str | None = None
    created: float = field(default_factory=time.time)
    progress: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    failure: str | None = None


class Admission:
    """Single-process bounded admission/store; restart invalidates all mappings."""

    def __init__(
        self,
        private: PrivateClient,
        max_queue=10,
        per_day=3,
        active_per_client=1,
        retention_s=86400,
        max_identities=10000,
        fast_deadline_s=90,
        adaptive_deadline_s=180,
    ):
        self.private, self.max_queue, self.per_day, self.active_per_client = (
            private,
            max_queue,
            per_day,
            active_per_client,
        )
        self.retention_s = retention_s
        self.max_identities = max_identities
        self.fast_deadline_s = fast_deadline_s
        self.adaptive_deadline_s = adaptive_deadline_s
        self.lock = threading.RLock()
        self.entries: dict[str, Entry] = {}
        self.waiting: deque[str] = deque()
        self.starts: dict[str, deque[float]] = {}
        self.busy = False
        self.accepting = True
        self.stop = threading.Event()
        threading.Thread(target=self._work, daemon=True, name="public-gateway-worker").start()

    def enqueue(self, question: str, path: str, client: str) -> Entry:
        now = time.time()
        with self.lock:
            self._prune(now)
            if not self.accepting:
                raise HTTPException(
                    503, detail={"code": "unavailable", "message": "Service is unavailable."}
                )
            if client not in self.starts and len(self.starts) >= self.max_identities:
                raise HTTPException(
                    503,
                    detail={
                        "code": "service_busy",
                        "message": "Service is busy. Please try again later.",
                    },
                )
            history = self.starts.setdefault(client, deque())
            while history and history[0] < now - 86400:
                history.popleft()
            active = sum(
                e.client_key == client and e.state in {"queued", "running", "awaiting_reading"}
                for e in self.entries.values()
            )
            if active >= self.active_per_client:
                raise HTTPException(
                    429,
                    detail={
                        "code": "client_busy",
                        "message": "You already have a public request in progress.",
                    },
                )
            if len(history) >= self.per_day:
                raise HTTPException(
                    429, detail={"code": "quota", "message": "Daily public request limit reached."}
                )
            if len(self.waiting) >= self.max_queue:
                raise HTTPException(
                    429,
                    detail={
                        "code": "queue_full",
                        "message": "The public queue is full. Please try again later.",
                    },
                )
            history.append(now)
            entry = Entry(secrets.token_urlsafe(24), client, question, path)
            self.entries[entry.public_id] = entry
            self.waiting.append(entry.public_id)
            return entry

    def _prune(self, now: float):
        expired = [
            key
            for key, entry in self.entries.items()
            if now - entry.created > self.retention_s
            and entry.state not in {"queued", "running", "awaiting_reading"}
        ]
        for key in expired:
            self.entries.pop(key, None)
        for client, history in list(self.starts.items()):
            while history and history[0] < now - 86400:
                history.popleft()
            if not history:
                self.starts.pop(client, None)

    def lookup(self, public_id: str) -> Entry | None:
        with self.lock:
            self._prune(time.time())
            return self.entries.get(public_id)

    def _work(self):
        while not self.stop.is_set():
            with self.lock:
                public_id = self.waiting.popleft() if self.waiting else None
                if public_id:
                    self.busy = True
            if not public_id:
                self.stop.wait(0.2)
                continue
            entry = self.entries.get(public_id)
            if entry is None:
                continue
            entry.state = "running"
            try:
                if hasattr(self.private, "can_start") and not self.private.can_start():
                    entry.state, entry.failure = "rejected", "service busy"
                    continue
                deadline = (
                    self.fast_deadline_s if entry.path == "fast" else self.adaptive_deadline_s
                )
                try:
                    reservation = self.private.reserve()
                except Exception:  # noqa: BLE001 - admission failure is generic
                    entry.state, entry.failure = "rejected", "service busy"
                    continue
                try:
                    entry.private_id = self.private.start(
                        entry.question,
                        "baseline" if entry.path == "fast" else "baseline-enhanced",
                        "fast" if entry.path == "fast" else "slow",
                        deadline,
                        reservation,
                    )
                except Exception:  # noqa: BLE001 - do not surface private API details
                    entry.state, entry.failure = "rejected", "service busy"
                    continue
                end = time.monotonic() + deadline
                while time.monotonic() < end:
                    run = self.private.get(entry.private_id)
                    status = run.get("status")
                    if status in {
                        "answered",
                        "incomplete",
                        "stopped",
                        "awaiting_reading",
                        "corpus_absent",
                        "no_evidence_found",
                    }:
                        entry.result = project_run(run)
                        entry.state = entry.result["state"]
                        break
                    entry.progress.append(
                        {
                            "atSeconds": len(entry.progress) * 3,
                            "message": "Research is in progress.",
                        }
                    )
                    time.sleep(3)
                else:
                    entry.state, entry.failure = "incomplete", "run timed out"
            except Exception:  # noqa: BLE001 - sanitize all private-client failures at this boundary
                entry.state, entry.failure = "incomplete", "service busy"
            finally:
                with self.lock:
                    self.busy = False


def project_run(run: dict[str, Any]) -> dict[str, Any]:
    status = run.get("status")
    if status == "answered":
        idx = run.get("chosenReading")
        readings = run.get("readings") or []
        if not isinstance(idx, int) or not 0 <= idx < len(readings):
            return {
                "state": "incomplete",
                "answer": None,
                "evidence": [],
                "graph": {"nodes": [], "edges": []},
            }
        reading = readings[idx]
        claims = reading.get("claims") or []
        triples = reading.get("triples") or []
        answer_claim_ids = reading.get("answerClaimIds") or []
        # Fail closed: the answer must cite at least one accepted claim, and
        # every public factual edge must tie to validated source evidence.
        safe_triples = []
        safe_claim_ids = set()
        safe_sources = set()
        public_sources: dict[str, str] = {}

        def public_source(value: str) -> str:
            if value not in public_sources:
                public_sources[value] = f"source-{len(public_sources) + 1}"
            return public_sources[value]

        def valid_claim(claim: Any) -> bool:
            if not isinstance(claim, dict) or claim.get("status", "accepted") != "accepted":
                return False
            claim_id = claim.get("id")
            sources = claim.get("sourceIds")
            fields = claim.get("fields")
            evidence = claim.get("evidence")
            anchors = claim.get("evidenceAnchors")
            if not isinstance(claim_id, str) or not claim_id:
                return False
            if (
                not isinstance(sources, list)
                or not sources
                or not all(isinstance(x, str) and x for x in sources)
            ):
                return False
            if (
                not isinstance(fields, dict)
                or not fields
                or not isinstance(evidence, dict)
                or not evidence
            ):
                return False
            if not set(sources) <= set(evidence) or not all(
                isinstance(evidence[x], str) and evidence[x].strip() for x in sources
            ):
                return False
            if not isinstance(anchors, list) or not anchors:
                return False
            anchor_ids = {
                a.get("id") for a in anchors if isinstance(a, dict) and isinstance(a.get("id"), str)
            }
            if not anchor_ids:
                return False
            for field in fields.values():
                if (
                    not isinstance(field, dict)
                    or not isinstance(field.get("value"), str)
                    or not field["value"].strip()
                ):
                    return False
                field_sources = field.get("sourceIds")
                if (
                    not isinstance(field_sources, list)
                    or not field_sources
                    or not set(field_sources) <= set(sources)
                ):
                    return False
                evidence_ids = field.get("evidenceIds", [])
                if not isinstance(evidence_ids, list) or not set(evidence_ids) <= anchor_ids:
                    return False
            return claim.get("kind") != "SynthesisClaim" or bool(claim.get("supportedBy"))

        valid_claims = [claim for claim in claims if valid_claim(claim)]
        for claim in valid_claims:
            safe_claim_ids.add(claim["id"])
            safe_sources.update(claim["sourceIds"])
        for triple in triples:
            if not all(
                isinstance(triple.get(k), str) and triple[k].strip()
                for k in ("s", "p", "o", "source", "evidence", "support")
            ):
                continue
            matching_claims = [
                claim
                for claim in valid_claims
                if triple["source"] in claim["sourceIds"]
                and claim["evidence"].get(triple["source"]) == triple["evidence"]
            ]
            if not matching_claims:
                continue
            if triple.get("claimId") and triple["claimId"] not in {
                c["id"] for c in matching_claims
            }:
                continue
            projected = {k: triple[k] for k in ("s", "p", "o", "evidence", "support")}
            projected["source"] = public_source(triple["source"])
            safe_triples.append(projected)
        if (
            not safe_triples
            or not valid_claims
            or not answer_claim_ids
            or not isinstance(answer_claim_ids, list)
            or not set(answer_claim_ids) <= safe_claim_ids
        ):
            return {
                "state": "incomplete",
                "answer": None,
                "evidence": [],
                "graph": {"nodes": [], "edges": []},
            }
        safe_claims = []
        for c in valid_claims:
            safe_claims.append(
                {
                    "kind": c.get("kind", "claim"),
                    "fields": {
                        name: {"value": field.get("value")}
                        for name, field in c["fields"].items()
                        if isinstance(field, dict) and isinstance(field.get("value"), str)
                    },
                    "evidence": [v for v in c["evidence"].values() if isinstance(v, str)],
                    "evidenceAnchors": [
                        {"kind": a.get("kind"), "text": a.get("text")}
                        for a in c["evidenceAnchors"]
                        if isinstance(a, dict)
                        and a.get("kind")
                        in {
                            "passage",
                            "table_cell",
                            "figure_annotation",
                            "dblp_fact",
                            "citation_fact",
                        }
                        and isinstance(a.get("text"), str)
                    ],
                }
            )
        if not safe_claims:
            return {
                "state": "incomplete",
                "answer": None,
                "evidence": [],
                "graph": {"nodes": [], "edges": []},
            }
        graph = reading.get("graph") or {}
        raw_nodes = graph.get("nodes", [])
        node_map = {
            node.get("id"): f"entity-{i + 1}"
            for i, node in enumerate(raw_nodes)
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }
        safe_nodes = [
            {
                "id": node_map[node["id"]],
                "label": node.get("label", ""),
                "kind": node.get("kind", "entity"),
            }
            for node in raw_nodes
            if isinstance(node, dict) and node.get("id") in node_map
        ]
        safe_edges = [
            {
                "source": node_map[edge["source"]],
                "target": node_map[edge["target"]],
                "label": edge.get("label", ""),
                "kind": edge.get("kind", "fact"),
                "evidence": edge.get("evidence"),
            }
            for edge in graph.get("edges", [])
            if isinstance(edge, dict)
            and edge.get("source") in node_map
            and edge.get("target") in node_map
            and edge.get("kind") in {"fact", "claim", "evidence", "derived"}
        ]
        answer = reading.get("answer")
        if (
            not isinstance(answer, str)
            or not answer.strip()
            or any(
                marker in answer.casefold()
                for marker in ("localhost", "127.0.0.1", "traceback", "api_key", "token=")
            )
        ):
            return {
                "state": "incomplete",
                "answer": None,
                "evidence": [],
                "triples": [],
                "graph": {"nodes": [], "edges": []},
            }
        return {
            "state": "answered",
            "answer": answer,
            "evidence": safe_claims,
            "triples": safe_triples,
            "graph": {"nodes": safe_nodes, "edges": safe_edges},
            "reading": {"label": reading.get("label"), "gloss": reading.get("gloss")},
        }
    state = (
        "awaiting_reading"
        if status == "awaiting_reading"
        else "stopped"
        if status == "stopped"
        else "incomplete"
    )
    return {
        "state": state,
        "answer": None,
        "evidence": [],
        "triples": [],
        "graph": {"nodes": [], "edges": []},
    }


class Config(BaseModel):
    private_origin: str = "http://127.0.0.1:8000"
    origins: list[str] = []
    identity_salt: str = ""
    verify_turnstile: bool = True
    turnstile_secret: str = ""
    dev_bypass: bool = False
    fast_enabled: bool = True
    adaptive_enabled: bool = True
    trust_client_header: bool = False
    environment: Literal["development", "production"] = "production"
    max_queue: int = Field(default=10, ge=1, le=10)
    starts_per_day: int = Field(default=3, ge=1, le=3)
    active_per_client: int = Field(default=1, ge=1, le=5)
    max_identities: int = Field(default=10000, ge=100, le=10000)
    max_body_bytes: int = Field(default=4096, ge=512, le=4096)
    min_question_chars: int = Field(default=8, ge=8, le=500)
    max_question_chars: int = Field(default=500, ge=8, le=500)
    fast_deadline_s: int = Field(default=90, ge=1, le=90)
    adaptive_deadline_s: int = Field(default=180, ge=1, le=180)

    @model_validator(mode="after")
    def question_bounds(self):
        if self.min_question_chars > self.max_question_chars:
            raise ValueError("minimum question length exceeds maximum")
        return self

    @field_validator("origins")
    @classmethod
    def exact_origins_only(cls, value: list[str]) -> list[str]:
        if any(origin.strip() == "*" for origin in value):
            raise ValueError("wildcard CORS origins are forbidden")
        return [origin.strip().rstrip("/") for origin in value if origin.strip()]


def create_app(
    private: PrivateClient | None = None,
    verifier: HumanVerifier | None = None,
    config: Config | None = None,
) -> FastAPI:
    config = config or Config(
        private_origin=os.getenv("PUBLIC_PRIVATE_ORIGIN", "http://127.0.0.1:8000"),
        origins=[x for x in os.getenv("PUBLIC_ALLOWED_ORIGINS", "").split(",") if x],
        identity_salt=os.getenv("PUBLIC_IDENTITY_SALT") or secrets.token_hex(32),
        verify_turnstile=os.getenv("PUBLIC_TURNSTILE_ENABLED", "true").lower() == "true",
        turnstile_secret=os.getenv("TURNSTILE_SECRET_KEY", ""),
        dev_bypass=os.getenv("PUBLIC_DEV_BYPASS", "false").lower() == "true",
        fast_enabled=os.getenv("PUBLIC_FAST_ENABLED", "true").lower() == "true",
        adaptive_enabled=os.getenv("PUBLIC_ADAPTIVE_ENABLED", "true").lower() == "true",
        trust_client_header=os.getenv("PUBLIC_TRUST_CLIENT_ID", "false").lower() == "true",
        environment=os.getenv("PUBLIC_ENV", "production").lower(),
        max_queue=int(os.getenv("PUBLIC_MAX_QUEUE", "10")),
        starts_per_day=int(os.getenv("PUBLIC_MAX_STARTS_PER_DAY", "3")),
        active_per_client=int(os.getenv("PUBLIC_MAX_ACTIVE_PER_CLIENT", "1")),
        max_identities=int(os.getenv("PUBLIC_MAX_IDENTITIES", "10000")),
        max_body_bytes=int(os.getenv("PUBLIC_MAX_BODY_BYTES", "4096")),
        min_question_chars=int(os.getenv("PUBLIC_QUESTION_MIN_CHARS", "8")),
        max_question_chars=int(os.getenv("PUBLIC_QUESTION_MAX_CHARS", "500")),
        fast_deadline_s=int(os.getenv("PUBLIC_FAST_DEADLINE_SECONDS", "90")),
        adaptive_deadline_s=int(os.getenv("PUBLIC_ADAPTIVE_DEADLINE_SECONDS", "180")),
    )
    if config.dev_bypass and config.environment != "development":
        raise ValueError("public development bypass is forbidden outside development")
    private = private or LoopbackPrivateClient(config.private_origin)
    verifier = verifier or TurnstileVerifier(
        config.turnstile_secret, config.verify_turnstile, config.dev_bypass
    )
    admission = Admission(
        private,
        max_queue=config.max_queue,
        per_day=config.starts_per_day,
        active_per_client=config.active_per_client,
        max_identities=config.max_identities,
        fast_deadline_s=config.fast_deadline_s,
        adaptive_deadline_s=config.adaptive_deadline_s,
    )
    app = FastAPI(
        title="Research Harness Public Gateway", docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.admission = admission
    app.state.status_provider = BasicStatus(private)

    @app.exception_handler(RequestValidationError)
    async def safe_validation_error(_request: Request, _error: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": "Check the question, path, and verification token format.",
                }
            },
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    def identity(request: Request) -> str:
        explicit = request.headers.get("x-public-client-id", "")
        trusted_peer = request.client and request.client.host in {"127.0.0.1", "::1"}
        if (
            config.trust_client_header
            and trusted_peer
            and explicit
            and len(explicit) <= 128
            and all(c.isalnum() or c in "-_." for c in explicit)
        ):
            return (
                "deployment:"
                + hmac.new(
                    config.identity_salt.encode(), explicit.encode(), hashlib.sha256
                ).hexdigest()
            )
        # CF-Connecting-IP is trusted only when network ingress is restricted to Cloudflare Tunnel.
        ip = (
            request.headers.get("cf-connecting-ip")
            if request.client and request.client.host in {"127.0.0.1", "::1"}
            else (request.client.host if request.client else "unknown")
        )
        return hmac.new(
            config.identity_salt.encode(), (ip or "unknown").encode(), hashlib.sha256
        ).hexdigest()

    @app.get("/v1/public/status", response_model=PublicStatusResponse)
    def status():
        q = len(admission.waiting)
        available = admission.accepting
        private_available = app.state.status_provider.available()
        model_capacity = app.state.status_provider.capacity_available()
        state = (
            "unavailable"
            if not available or not private_available
            else "busy"
            if admission.busy
            else "saturated"
            if q >= admission.max_queue or not model_capacity
            else "available"
        )
        return {
            "state": state,
            "acceptingRuns": available
            and private_available
            and q < admission.max_queue
            and (model_capacity or admission.busy),
            "queue": {"waiting": q, "capacity": admission.max_queue},
            "paths": {
                "fast": config.fast_enabled,
                "adaptive": config.adaptive_enabled,
                "agentic": False,
            },
            "message": f"{q} public requests are waiting."
            if q
            else "Public research is available.",
        }

    @app.get("/v1/public/examples", response_model=list[PublicExample])
    def examples():
        path = Path(__file__).resolve().parents[4] / "public_examples" / "examples.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    @app.post("/v1/public/runs", status_code=202, response_model=PublicStartResponse)
    async def start(body: PublicStart, request: Request):
        if body.path not in {"fast", "adaptive", "agentic"}:
            raise HTTPException(
                422,
                detail={"code": "invalid_path", "message": "Choose an available research path."},
            )
        if (body.path == "fast" and not config.fast_enabled) or (
            body.path == "adaptive" and not config.adaptive_enabled
        ):
            raise HTTPException(
                503,
                detail={
                    "code": "path_unavailable",
                    "message": "This research path is temporarily unavailable.",
                },
            )
        if body.path == "agentic":
            raise HTTPException(
                503,
                detail={
                    "code": "path_unavailable",
                    "message": "This research path is temporarily unavailable.",
                },
            )
        if (
            len(body.question) < config.min_question_chars
            or len(body.question) > config.max_question_chars
        ):
            raise HTTPException(
                422,
                detail={
                    "code": "invalid_question",
                    "message": "Question length is outside the allowed range.",
                },
            )
        if len(await request.body()) > config.max_body_bytes:
            raise HTTPException(
                413, detail={"code": "too_large", "message": "Request is too large."}
            )
        if not verifier.verify(body.turnstileToken):
            raise HTTPException(
                400,
                detail={
                    "code": "human_check",
                    "message": "Please complete the human verification and try again.",
                },
            )
        if hasattr(private, "ready"):
            try:
                ready = private.ready()
            except Exception:  # noqa: BLE001 - never return private-client diagnostics
                ready = False
            if not ready:
                raise HTTPException(
                    503,
                    detail={
                        "code": "unavailable",
                        "message": "Research service is temporarily unavailable.",
                    },
                )
        entry = admission.enqueue(body.question, body.path, identity(request))
        return {
            "publicRunId": entry.public_id,
            "state": entry.state,
            "queuePosition": len(admission.waiting),
        }

    @app.get("/v1/public/runs/{public_run_id}", response_model=PublicRunResponse)
    def get_run(public_run_id: str):
        entry = admission.lookup(public_run_id)
        if not entry:
            raise HTTPException(
                404, detail={"code": "not_found", "message": "Public run is unavailable."}
            )
        return {
            "publicRunId": entry.public_id,
            "question": entry.question,
            "path": entry.path,
            "state": entry.state,
            "progress": entry.progress[-20:],
            "answer": (entry.result or {}).get("answer"),
            "evidence": (entry.result or {}).get("evidence", []),
            "triples": (entry.result or {}).get("triples", []),
            "graph": (entry.result or {}).get("graph", {"nodes": [], "edges": []}),
            "reading": (entry.result or {}).get("reading"),
            "failure": entry.failure,
        }

    @app.get("/v1/public/runs/{public_run_id}/events")
    async def events(public_run_id: str):
        if admission.lookup(public_run_id) is None:
            raise HTTPException(
                404, detail={"code": "not_found", "message": "Public run is unavailable."}
            )

        async def stream():
            last = -1
            while True:
                entry = admission.lookup(public_run_id)
                if not entry:
                    break
                payload = {
                    "publicRunId": entry.public_id,
                    "state": entry.state,
                    "progress": entry.progress[-20:],
                    "answer": (entry.result or {}).get("answer"),
                }
                if len(entry.progress) != last or entry.state in {
                    "answered",
                    "incomplete",
                    "stopped",
                    "rejected",
                }:
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                    last = len(entry.progress)
                if entry.state in {"answered", "incomplete", "stopped", "rejected"}:
                    break
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    # Optional same-origin preview mode: the Tunnel targets this gateway only,
    # which serves the public bundle and its narrowly scoped API together.
    # Private API/MCP ports remain loopback-only and are never mounted here.
    if os.getenv("PUBLIC_SERVE_FRONTEND", "false").lower() == "true":
        frontend = Path(__file__).resolve().parents[4] / "web" / "dist"
        if not (frontend / "index.html").is_file():
            raise RuntimeError("public frontend build is missing")
        app.mount("/", StaticFiles(directory=frontend, html=True), name="public-frontend")

        @app.middleware("http")
        async def public_security_headers(request: Request, call_next):
            response = await call_next(request)
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' https://challenges.cloudflare.com; "
                "frame-src https://challenges.cloudflare.com; connect-src 'self' "
                "https://challenges.cloudflare.com; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; font-src 'self'; object-src 'none'; base-uri 'self'; "
                "form-action 'self'; frame-ancestors 'none'"
            )
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
            return response

    return app


app = create_app()
