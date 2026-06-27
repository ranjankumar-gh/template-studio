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


def validate_conversation(conversation: Conversation) -> list[str]:
    """Return a list of input-contract violations. Empty list means well-formed."""
    if not conversation:
        return ["conversation is empty"]

    problems: list[str] = []
    for i, message in enumerate(conversation):
        role = message.get("role")
        if role not in VALID_ROLES:
            problems.append(f"message {i}: unknown role {role!r}")

        content = message.get("content")
        if content is None:
            problems.append(f"message {i}: content is None")
        elif not isinstance(content, (str, list)):
            kind = type(content).__name__
            problems.append(f"message {i}: content must be str or list, got {kind}")
        elif isinstance(content, str) and not content.strip():
            problems.append(f"message {i}: content is empty or whitespace")

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
