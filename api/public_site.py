"""The full harness UI on one port, behind a password, for hosting through a free tunnel.

    SITE_PASSWORD=... uv run uvicorn public_site:create_app --factory --app-dir api --host 127.0.0.1 --port 8080

``scripts/public-full.sh`` starts everything (corpus server, API, this site) and a tunnel in front.

- Serves the built UI (``web/dist``, from ``corepack pnpm --dir web build``) and forwards ``/api/*`` to
  the harness API on loopback (``HARNESS_API_ORIGIN``, default http://127.0.0.1:8000), streaming
  responses included (run events).
- Every request needs a login (HTTP Basic auth, so the browser asks once): ``SITE_USERS``
  ("name:password,name2:password2") or ``SITE_PASSWORD`` (any user name). Passwords need at least 12
  characters. After 10 failed logins a client waits 10 minutes.
- The API and the models stay on loopback; only this port goes behind the tunnel. Unlike the public
  gateway (``packages/public_gateway``), nothing is restricted once logged in: every page and path.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

ROOT = Path(__file__).resolve().parents[1]
MIN_PASSWORD = 12
MAX_FAILURES = 10
LOCKOUT_SECONDS = 600
# Hop-by-hop headers (RFC 9110 §7.6.1) and the ones the proxy sets itself.
_DROP_REQUEST = {"host", "authorization", "content-length", "connection", "keep-alive", "te", "trailer",
                 "transfer-encoding", "upgrade", "proxy-authorization", "cookie", "accept-encoding"}
_DROP_RESPONSE = {"content-length", "content-encoding", "connection", "keep-alive", "transfer-encoding",
                  "trailer", "upgrade", "server", "date"}


def users_from_env() -> dict[str, str]:
    """name -> password from SITE_USERS, else "*" -> SITE_PASSWORD; refuses weak or missing passwords."""
    users: dict[str, str] = {}
    for item in (os.getenv("SITE_USERS") or "").split(","):
        if item.strip():
            name, _, password = item.strip().partition(":")
            if not name or not password:
                raise RuntimeError(f"SITE_USERS entry without name:password: {item.strip()[:20]!r}")
            users[name] = password
    if not users and os.getenv("SITE_PASSWORD"):
        users["*"] = os.environ["SITE_PASSWORD"]
    if not users:
        raise RuntimeError("set SITE_PASSWORD (or SITE_USERS=name:password,...) before hosting the site")
    weak = [name for name, password in users.items() if len(password) < MIN_PASSWORD]
    if weak:
        raise RuntimeError(f"passwords need at least {MIN_PASSWORD} characters (too short for: {', '.join(weak)})")
    return users


def create_app(users: dict[str, str] | None = None, origin: str | None = None, dist: Path | None = None,
               transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    users = users if users is not None else users_from_env()
    origin = (origin or os.getenv("HARNESS_API_ORIGIN") or "http://127.0.0.1:8000").rstrip("/")
    dist = dist or Path(os.getenv("SITE_WEB_DIST") or ROOT / "web" / "dist")
    if not (dist / "index.html").exists():
        raise RuntimeError(f"no built UI at {dist}: run `corepack pnpm --dir web build` first")
    client = httpx.AsyncClient(base_url=origin, transport=transport,
                               timeout=httpx.Timeout(None, connect=10.0))
    failures: dict[str, list[float]] = defaultdict(list)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="Research Harness (site)", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)

    def client_key(request: Request) -> str:
        # Behind a Cloudflare tunnel every request comes from loopback; the visitor is in this header.
        return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")

    def authorised(header: str) -> bool:
        scheme, _, value = header.partition(" ")
        if scheme.casefold() != "basic" or not value:
            return False
        try:
            name, _, password = base64.b64decode(value.strip(), validate=True).decode("utf-8").partition(":")
        except (binascii.Error, UnicodeDecodeError):
            return False
        expected = users.get(name, users.get("*"))
        return expected is not None and hmac.compare_digest(password.encode(), expected.encode())

    @app.middleware("http")
    async def login(request: Request, call_next):
        key = client_key(request)
        now = time.monotonic()
        failures[key] = [moment for moment in failures[key] if now - moment < LOCKOUT_SECONDS]
        if len(failures[key]) >= MAX_FAILURES:
            return JSONResponse({"detail": "too many failed logins; try again later"}, status_code=429)
        if not authorised(request.headers.get("authorization", "")):
            if request.headers.get("authorization"):
                failures[key].append(now)
            return Response("Login required", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Research Harness", charset="UTF-8"'})
        failures.pop(key, None)
        response = await call_next(request)
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def forward(path: str, request: Request) -> Response:
        headers = {name: value for name, value in request.headers.items() if name.casefold() not in _DROP_REQUEST}
        upstream_request = client.build_request(request.method, httpx.URL(path="/" + path,
                                                                         query=request.url.query.encode()),
                                                headers=headers, content=await request.body())
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            return JSONResponse({"detail": f"harness API unreachable ({type(exc).__name__})"}, status_code=502)
        kept = {name: value for name, value in upstream.headers.items() if name.casefold() not in _DROP_RESPONSE}
        return StreamingResponse(upstream.aiter_bytes(), status_code=upstream.status_code, headers=kept,
                                 background=BackgroundTask(upstream.aclose))

    root = dist.resolve()

    @app.get("/{path:path}")
    async def ui(path: str) -> Response:
        target = (root / path).resolve()
        if path and target.is_file() and target.is_relative_to(root):
            return FileResponse(target)
        return FileResponse(root / "index.html")          # the app handles its own pages

    return app
