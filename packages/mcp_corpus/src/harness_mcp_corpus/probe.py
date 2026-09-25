"""Run the M1 bare-agent exit questions with corpus tools only."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from harness_config import settings
from harness_prompts import render
from mcp.server.fastmcp.exceptions import ToolError

from harness_mcp_corpus.server import create_server


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
        {"role": "system", "content": render("mcp_corpus/probe").system},
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
    parser.add_argument("--root", type=Path, default=settings().corpus_root or Path("corpus_data"))
    parser.add_argument(
        "--base-url",
        default=settings().model_agent_url,
        required=settings().model_agent_url is None,
    )
    parser.add_argument("--model", default=settings().model_agent)
    args = parser.parse_args()
    result = asyncio.run(
        ask(
            args.root,
            args.question,
            args.base_url,
            args.model,
            settings().api_key,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
