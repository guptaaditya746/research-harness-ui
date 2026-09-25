"""Run the M1 bare-agent exit questions with corpus tools only."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp.exceptions import ToolError

from harness_mcp_corpus.server import create_server

SYSTEM = (
    "You answer questions using only the corpus tools provided. "
    "Call tools to check evidence before answering. "
    "For citation questions, call citations with direction=in. If total is zero, report that no "
    "indexed paper cites the target and do not cite unrelated outgoing facts. Otherwise call facts "
    "on each citing paper and cite the exact cites fact_id. A reference title names the cited target, "
    "not the citing paper. For citation answers, give paper IDs and fact IDs only; omit titles. "
    "For introduction questions, search passages, read the relevant passage, and cite its passage_id. "
    "Use page limits from 1 to 50 and nonnegative offsets. "
    "If the indexed corpus does not establish an answer, say so."
)


def serialise_result(value: Any) -> Any:
    if isinstance(value, tuple):
        return [serialise_result(item) for item in value]
    if isinstance(value, list):
        return [serialise_result(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


async def ask(
    root: Path, question: str, base_url: str, model: str, api_key: str, max_steps: int = 12
) -> dict:
    server = create_server(root)
    tools = await server.list_tools()
    definitions = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools
    ]
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    calls: list[dict] = []
    async with httpx.AsyncClient(
        timeout=120, headers={"Authorization": f"Bearer {api_key}"}
    ) as client:
        for _ in range(max_steps):
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "tools": definitions,
                    "tool_choice": "auto",
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            requested = message.get("tool_calls") or []
            if not requested:
                return {
                    "question": question,
                    "answer": message.get("content") or "",
                    "tool_calls": calls,
                    "model": model,
                }
            messages.append(
                {"role": "assistant", "content": message.get("content"), "tool_calls": requested}
            )
            for call in requested:
                name = call["function"]["name"]
                if name not in {tool.name for tool in tools}:
                    raise ValueError(f"model requested an unavailable corpus tool: {name}")
                arguments = json.loads(call["function"]["arguments"])
                record: dict = {"name": name, "arguments": arguments}
                calls.append(record)
                try:
                    result = await server.call_tool(name, arguments)
                    serialised = serialise_result(result)
                    record["result"] = serialised
                    content = json.dumps(serialised)
                except ToolError as exc:
                    record["error"] = str(exc)
                    content = json.dumps({"error": str(exc)})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
    raise RuntimeError(f"model did not finish after {max_steps} steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--root", type=Path, default=Path(os.getenv("CORPUS_ROOT", "corpus_data")))
    parser.add_argument(
        "--base-url",
        default=os.getenv("HARNESS_MODEL_AGENT_URL"),
        required="HARNESS_MODEL_AGENT_URL" not in os.environ,
    )
    parser.add_argument("--model", default=os.getenv("HARNESS_MODEL_AGENT", "lite-model"))
    args = parser.parse_args()
    result = asyncio.run(
        ask(
            args.root,
            args.question,
            args.base_url,
            args.model,
            os.getenv("VLLM_API_KEY", "sk-local-vllm"),
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
