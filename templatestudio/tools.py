"""Tool-calling templates: rendering in, parsing out (Chapter 5)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol


def get_temperature(location: str, unit: str = "celsius") -> str:
    """Get the current temperature at a location.

    Args:
        location: The location to look up, e.g. "Paris, France".
        unit: The temperature unit, either "celsius" or "fahrenheit".
    """
    ...


def tool_schema(fn: object) -> dict:
    """Build a JSON-schema tool description from a Python function.

    Wraps Transformers' get_json_schema (imported lazily). The function must have
    type hints and a Google-style docstring; that text is the schema the model sees.
    """
    from transformers.utils import get_json_schema  # lazy: only needed for schemas

    return get_json_schema(fn)


def tool_call_message(
    name: str, arguments: dict, call_id: str | None = None
) -> dict:
    """An assistant message that calls a tool. tool_calls is always a list.
    call_id pairs the call with its result (Chapter 5, the agent loop)."""
    function = {"name": name, "arguments": arguments}
    call: dict = {"type": "function", "function": function}
    if call_id is not None:
        call["id"] = call_id
    return {"role": "assistant", "tool_calls": [call]}


def tool_result_message(
    name: str, content: str, tool_call_id: str | None = None
) -> dict:
    """A tool-role message carrying a result back. content is always a
    string."""
    message = {"role": "tool", "name": name, "content": content}
    if tool_call_id is not None:
        message["tool_call_id"] = tool_call_id
    return message


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict


class ToolCallParser(Protocol):
    """One parser per model output format. Pick the one matching your model."""

    def parse(self, text: str) -> list[ParsedToolCall]: ...


class JSONBlockParser:
    """For models that emit calls as a fenced ```json block."""

    _BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

    def parse(self, text: str) -> list[ParsedToolCall]:
        calls: list[ParsedToolCall] = []
        for match in self._BLOCK.finditer(text):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue  # a code block that is not a tool call; skip it
            calls.append(
                ParsedToolCall(
                    name=payload["name"],
                    arguments=self._normalize_arguments(payload.get("arguments", {})),
                )
            )
        return calls

    @staticmethod
    def _normalize_arguments(arguments: object) -> dict:
        """Coerce arguments to a dict. The message model says arguments is a
        dict, but many models emit it as a JSON string on the wire; normalize
        here so callers never have to guess (Chapter 5). Raise loudly on
        anything that is not a JSON object."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            decoded = json.loads(arguments)
            if not isinstance(decoded, dict):
                raise ValueError(
                    "tool-call arguments did not decode to an object: "
                    f"{arguments!r}"
                )
            return decoded
        raise ValueError(
            "tool-call arguments must be a dict or JSON string, got "
            f"{type(arguments).__name__}"
        )


def parallel_tool_call_message(*calls: dict) -> dict:
    """One assistant turn carrying several calls, built from single-call
    messages. Not every model was trained to emit this (Llama 3 was not)."""
    return {
        "role": "assistant",
        "tool_calls": [c for message in calls for c in message["tool_calls"]],
    }


def validate_tool_pairing(conversation: list[dict]) -> list[str]:
    """Every tool result must answer a call made earlier and not yet
    answered. An agent loop that truncates history, or retries a call, breaks
    this silently; most templates render the orphan without complaint."""
    problems: list[str] = []
    pending: list[str | None] = []  # call ids awaiting results, in order
    for i, message in enumerate(conversation):
        role = message.get("role")
        if role == "assistant":
            pending = [c.get("id") for c in message.get("tool_calls") or []]
        elif role == "tool":
            result_id = message.get("tool_call_id")
            if not pending:
                problems.append(f"message {i}: tool result answers no call")
            elif result_id is not None and result_id not in pending:
                problems.append(
                    f"message {i}: tool_call_id {result_id!r} matches no call"
                )
            else:
                answered = result_id if result_id is not None else pending[0]
                pending.remove(answered)
        elif role == "user" and pending:
            problems.append(f"message {i}: {len(pending)} call(s) unanswered")
            pending = []
    return problems


def mcp_result_to_tool_message(
    name: str, result: dict, tool_call_id: str | None = None
) -> dict:
    """Turn an MCP CallToolResult into a tool message whose content is the
    string these templates expect. Text blocks are kept in order;
    structuredContent is serialized when no text block carries it; other
    blocks (image, audio, resources) become a marker the model can see,
    rather than vanishing."""
    parts: list[str] = []
    for block in result.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        else:
            mime = block.get("mimeType") or block.get("uri") or "unknown"
            parts.append(f"[{kind} omitted: {mime}]")
    structured = result.get("structuredContent")
    if structured is not None:
        serialized = json.dumps(structured, ensure_ascii=False)
        if serialized not in parts:
            parts.append(serialized)
    if result.get("isError"):
        parts.insert(0, "[tool error]")
    return tool_result_message(name, "\n".join(parts), tool_call_id)
