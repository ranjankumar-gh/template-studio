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


def tool_call_message(name: str, arguments: dict) -> dict:
    """An assistant message that calls a tool. tool_calls is always a list."""
    return {
        "role": "assistant",
        "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def tool_result_message(name: str, content: str) -> dict:
    """A tool-role message carrying a result back. content is always a string."""
    return {"role": "tool", "name": name, "content": content}


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
        """Coerce arguments to a dict. The message model says arguments is a dict, but
        many models emit it as a JSON string on the wire; normalize here so callers never
        have to guess (Chapter 5). Raise loudly on anything that is not a JSON object."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            decoded = json.loads(arguments)
            if not isinstance(decoded, dict):
                raise ValueError(f"tool-call arguments did not decode to an object: {arguments!r}")
            return decoded
        raise ValueError(f"tool-call arguments must be a dict or JSON string, got {type(arguments).__name__}")
