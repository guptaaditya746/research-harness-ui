"""Pre-flight check for a live deployment: run before a demo or evaluation.

    uv run python scripts/preflight.py            # everything
    uv run python scripts/preflight.py --quick    # skip model round-trips

Checks configuration, the corpus, every configured model alias (structured
JSON and, for the agent model, native tool calling), the MCP corpus server,
the running API, and storage. Prints PASS / WARN / FAIL per check and exits
non-zero when any FAIL remains.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, str, str]] = []


def report(level: str, name: str, detail: str = "") -> None:
    RESULTS.append((level, name, detail))
    colour = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}[level]
    print(f"{colour}{level}\033[0m  {name}" + (f" — {detail}" if detail else ""), flush=True)


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def check_config() -> None:
    if not (ROOT / ".env").exists():
        report("FAIL", ".env present", "copy .env.example to .env and fill it in")
    else:
        report("PASS", ".env present")
    for name in ("LITELLM_BASE_URL", "HARNESS_MODEL_SMALL", "HARNESS_MODEL_AGENT", "MCP_CORPUS_URL"):
        report("PASS" if env(name) else "FAIL", f"{name} set", env(name) or "missing")
    for name, minimum in (("BUDGET_BASELINE_FAST_SECONDS", 30), ("BUDGET_BASELINE_SLOW_SECONDS", 60)):
        value = float(env(name, "90") or 90)
        if value < minimum:
            report("WARN", f"{name}={value:g}", f"per-call timeout below {minimum}s; Qwen structured calls often exceed it")
    if env("HARNESS_INCLUDE_FIXTURES", "true").casefold() not in {"0", "false", "no"}:
        report("WARN", "HARNESS_INCLUDE_FIXTURES", "recorded demo runs will be listed next to live runs; set false")
    judge, agent = env("HARNESS_MODEL_JUDGE"), env("HARNESS_MODEL_AGENT")
    if judge and judge == agent:
        report("WARN", "HARNESS_MODEL_JUDGE", "judge equals the agent model")


def check_corpus() -> None:
    root = Path(env("CORPUS_ROOT", str(ROOT / "corpus_data")))
    if not root.is_absolute():
        root = ROOT / root
    missing = [name for name in ("papers.jsonl", "passages.jsonl", "facts.jsonl", "references.jsonl")
               if not (root / name).exists()]
    if missing:
        report("FAIL", f"corpus at {root}", f"missing {', '.join(missing)}; run make ingest")
        return
    counts = {name: sum(1 for line in (root / name).open() if line.strip())
              for name in ("papers.jsonl", "passages.jsonl", "facts.jsonl", "references.jsonl")}
    report("PASS", f"corpus at {root}", ", ".join(f"{k.split('.')[0]}={v}" for k, v in counts.items()))
    if not (root / "index" / "bm25" / "params.index.json").exists():
        report("WARN", "BM25 index", "missing; ranking falls back to lexical order")
    else:
        report("PASS", "BM25 index")


def check_models(quick: bool) -> None:
    base = env("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1").rstrip("/")
    key = env("VLLM_API_KEY", "local")
    headers = {"Authorization": f"Bearer {key}"}
    try:
        listed = httpx.get(f"{base}/models", headers=headers, timeout=10)
        listed.raise_for_status()
        available = {item["id"] for item in listed.json().get("data", [])}
        report("PASS", f"LiteLLM reachable at {base}", f"{len(available)} aliases")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", f"LiteLLM reachable at {base}", f"{type(exc).__name__}: {exc}")
        return
    aliases = {role: env(var) for role, var in (
        ("extraction/planner", "HARNESS_MODEL_SMALL"), ("agent (concepts + MCP)", "HARNESS_MODEL_AGENT"),
        ("heavy", "HARNESS_MODEL_HIGH"), ("judge", "HARNESS_MODEL_JUDGE"),
    ) if env(var)}
    for role, alias in aliases.items():
        if available and alias not in available:
            report("FAIL", f"model alias {alias} ({role})", "not served by LiteLLM")
    if quick:
        return
    from harness_extraction import model_options, strip_reasoning

    for role, alias in aliases.items():
        started = time.monotonic()
        options = model_options(alias)
        try:
            response = httpx.post(f"{base}/chat/completions", headers=headers, timeout=120, json={
                "model": alias, "temperature": 0, "response_format": {"type": "json_object"},
                "max_tokens": int(options.get("max_tokens") or env("HARNESS_MAX_OUTPUT_TOKENS", "4096")),
                "messages": [{"role": "system", "content": 'Return strict JSON {"ok": true}.'},
                             {"role": "user", "content": "ping"}],
                **(options.get("extra_body") or {}),
            })
            response.raise_for_status()
            payload = response.json()
            message = payload["choices"][0]["message"]
            content = str(message.get("content") or "")
            answer = strip_reasoning(content)
            json.loads(answer[answer.find("{"): answer.rfind("}") + 1])
            took = time.monotonic() - started
            tokens = (payload.get("usage") or {}).get("completion_tokens")
            thinking = bool(message.get("reasoning_content")) or "<think>" in content
            detail = f"{took:.1f}s" + (f", {tokens} output tokens for a one-word answer" if tokens else "")
            if thinking or (tokens and tokens > 200):
                structured = role.startswith("extraction")
                report("WARN", f"model {alias} ({role}) is reasoning before answering", detail + (
                    "; every run makes ~30 structured calls with this model, so runs will be slow "
                    "and may time out. Use an instruct model for HARNESS_MODEL_SMALL, or disable "
                    "thinking via HARNESS_MODEL_OPTIONS" if structured else
                    "; consider disabling thinking via HARNESS_MODEL_OPTIONS for structured calls"))
            else:
                report("WARN" if took > 20 else "PASS", f"model {alias} ({role}) JSON round-trip", detail)
        except Exception as exc:  # noqa: BLE001
            report("FAIL", f"model {alias} ({role}) JSON round-trip", f"{type(exc).__name__}: {exc}"[:200])
    agent = env("HARNESS_MODEL_AGENT")
    if agent:
        from harness_agent.context import DEFAULT_WINDOW, discover_window
        from harness_extraction import OpenAICompatibleModel

        window = discover_window(OpenAICompatibleModel(base_url=base, model=agent, api_key=key))
        if env("HARNESS_AGENT_CONTEXT_TOKENS") or window != DEFAULT_WINDOW:
            report("PASS" if window >= 16_384 else "WARN", f"agent context window ({agent})",
                   f"{window} tokens" + ("" if window >= 16_384 else "; the agent will see fewer tools and claims"))
        else:
            report("WARN", f"agent context window ({agent})",
                   f"not reported by LiteLLM; assuming {DEFAULT_WINDOW}. Set HARNESS_AGENT_CONTEXT_TOKENS "
                   "to the served max_model_len (it is also learned from the first length error)")
        try:
            response = httpx.post(f"{base}/chat/completions", headers=headers, timeout=120, json={
                "model": agent, "temperature": 0,
                "messages": [{"role": "user", "content": "Search the corpus for 'world model'."}],
                "tools": [{"type": "function", "function": {
                    "name": "search_evidence", "description": "Search corpus passages.",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                                   "required": ["query"]}}}],
                "tool_choice": "auto",
            })
            response.raise_for_status()
            calls = response.json()["choices"][0]["message"].get("tool_calls") or []
            report("PASS" if calls else "WARN", f"model {agent} native tool calling",
                   "tool call returned" if calls else "answered without a tool call; check the vLLM tool parser")
        except Exception as exc:  # noqa: BLE001
            report("FAIL", f"model {agent} native tool calling", f"{type(exc).__name__}: {exc}"[:200])


def check_mcp() -> None:
    url = env("MCP_CORPUS_URL", "http://127.0.0.1:8765/mcp")
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async def probe() -> list[str]:
            async with streamablehttp_client(url, timeout=10) as (read, write, _), \
                    ClientSession(read, write) as session:
                await session.initialize()
                return [tool.name for tool in (await session.list_tools()).tools]

        names = asyncio.run(asyncio.wait_for(probe(), 20))
        report("PASS" if names else "FAIL", f"MCP corpus server at {url}", f"{len(names)} tools")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", f"MCP corpus server at {url}",
               f"{type(exc).__name__}; start it with make dev (or python -m harness_mcp_corpus.server)")


def check_api() -> None:
    base = env("PREFLIGHT_API_URL", "http://127.0.0.1:8000")
    try:
        for path in ("/runs", "/settings", "/corpus/papers", "/benchmark", "/datasets/candidates"):
            response = httpx.get(base + path, timeout=15)
            response.raise_for_status()
        papers = len(httpx.get(base + "/corpus/papers", timeout=15).json())
        report("PASS", f"API at {base}", f"all read endpoints OK, {papers} papers")
    except httpx.ConnectError:
        report("WARN", f"API at {base}", "not running yet (start with make dev)")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", f"API at {base}", f"{type(exc).__name__}: {exc}"[:200])


def check_storage() -> None:
    artifacts = Path(env("ARTIFACT_ROOT", str(ROOT / "artifacts")))
    artifacts = artifacts if artifacts.is_absolute() else ROOT / artifacts
    try:
        artifacts.mkdir(parents=True, exist_ok=True)
        probe = artifacts / ".preflight"
        probe.write_text("ok")
        probe.unlink()
        report("PASS", f"artifact root writable ({artifacts})")
    except OSError as exc:
        report("FAIL", f"artifact root writable ({artifacts})", str(exc))
    gold = Path(env("DATASETS_ROOT", str(ROOT / "datasets")))
    gold = (gold if gold.is_absolute() else ROOT / gold) / "gold" / "candidates.jsonl"
    rows = [json.loads(line) for line in gold.read_text().splitlines() if line.strip()] if gold.exists() else []
    approved = sum(1 for row in rows if row.get("status") == "approved")
    report("PASS" if approved else "WARN", "gold candidates",
           f"{len(rows)} total, {approved} approved" + ("" if approved else "; the Benchmark page will be empty"))


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip model round-trips")
    args = parser.parse_args()
    check_config()
    check_corpus()
    check_models(args.quick)
    check_mcp()
    check_api()
    check_storage()
    failures = sum(1 for level, _, _ in RESULTS if level == "FAIL")
    warnings = sum(1 for level, _, _ in RESULTS if level == "WARN")
    print(f"\n{failures} failed, {warnings} warnings, {len(RESULTS) - failures - warnings} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
