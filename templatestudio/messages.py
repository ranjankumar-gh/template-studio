"""The structured message model (Chapter 2): the left side of the render contract."""

from __future__ import annotations

from typing import Literal, TypedDict, Union

Role = Literal["system", "user", "assistant", "tool"]
VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})

# A content part is one item in a multimodal or tool message (Chapter 7).
ContentPart = dict[str, object]
Content = Union[str, list[ContentPart]]


class Message(TypedDict):
    role: Role
    content: Content


Conversation = list[Message]


def _validate_tool_calls(tool_calls: object) -> list[str]:
    """Check the shape of an assistant message's tool_calls (Chapter 5)."""
    if not isinstance(tool_calls, list):
        return ["tool_calls must be a list"]
    problems: list[str] = []
    for j, call in enumerate(tool_calls):
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            problems.append(f"tool_calls[{j}]: missing function object")
        elif "name" not in function or "arguments" not in function:
            problems.append(
                f"tool_calls[{j}]: function needs name and arguments"
            )
    return problems


def validate_conversation(conversation: Conversation) -> list[str]:
    """Return a list of input-contract violations. Empty list means well-formed."""
    if not conversation:
        return ["conversation is empty"]

    problems: list[str] = []
    for i, message in enumerate(conversation):
        role = message.get("role")
        if role not in VALID_ROLES:
            problems.append(f"message {i}: unknown role {role!r}")

        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            problems.extend(
                f"message {i}: {p}" for p in _validate_tool_calls(tool_calls)
            )

        content = message.get("content")
        if content is None:
            # An assistant turn that carries tool_calls legitimately has no
            # content - Chapter 5's tool_call_message builds exactly that
            # shape - so the content rule does not apply to it. An *empty*
            # tool_calls list carries nothing, so it does not earn the
            # exemption either.
            if role != "assistant" or not tool_calls:
                problems.append(f"message {i}: content is None")
        elif not isinstance(content, (str, list)):
            kind = type(content).__name__
            problems.append(f"message {i}: content must be str or list, got {kind}")
        elif isinstance(content, str) and not content.strip():
            problems.append(f"message {i}: content is empty or whitespace")
        elif isinstance(content, list):
            # Multimodal content: validate the typed parts too (Chapter 7). Local import
            # avoids a circular dependency - multimodal imports from this module.
            from templatestudio.multimodal import validate_content_parts

            problems.extend(
                f"message {i}: {problem}" for problem in validate_content_parts(content)
            )

    for i, message in enumerate(conversation):
        if message.get("role") == "system" and i != 0:
            problems.append(f"message {i}: system message must be first")

    return problems


class InvalidConversation(ValueError):
    """Raised when a conversation violates the input contract."""


def assert_valid(conversation: Conversation) -> None:
    """Raise InvalidConversation listing every violation, or return cleanly."""
    problems = validate_conversation(conversation)
    if problems:
        raise InvalidConversation("; ".join(problems))
