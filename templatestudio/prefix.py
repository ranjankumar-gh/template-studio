"""The prefix contract (Chapter 10): every prompt a conversation has already
sent must be a token prefix of every later one, or the engine cannot reuse its
cached work for it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from templatestudio.messages import Conversation

# conversation -> the token ids the engine would see for that request,
# generation prompt included (a TokenRenderer's .ids from serving.py).
RenderIds = Callable[[Conversation], list[int]]


@dataclass(frozen=True)
class PrefixBreak:
    """A later request that does not start with an earlier one."""

    earlier: int   # messages in the earlier request
    later: int     # messages in the later request
    position: int  # first token index where the later request differs


def request_points(conversation: Conversation) -> list[int]:
    """The lengths at which a client sends a request: after every message the
    model has to answer - a user turn or a tool result. A system message on
    its own is never a request."""
    return [
        i + 1
        for i, message in enumerate(conversation)
        if message["role"] in ("user", "tool")
    ]


def find_prefix_breaks(
    render_ids: RenderIds, conversation: Conversation
) -> list[PrefixBreak]:
    """Render the prompt at each request point and report every consecutive
    pair where the earlier prompt is not a token prefix of the later one."""
    points = request_points(conversation)
    prompts = [render_ids(conversation[:n]) for n in points]
    breaks: list[PrefixBreak] = []
    requests = list(zip(points, prompts))
    for (n_a, a), (n_b, b) in zip(requests, requests[1:]):
        if b[: len(a)] == a:
            continue
        position = next(
            (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
            min(len(a), len(b)),
        )
        breaks.append(PrefixBreak(earlier=n_a, later=n_b, position=position))
    return breaks


def assert_prefix_stable(
    render_ids: RenderIds, conversation: Conversation
) -> None:
    """Raise if any request in the conversation fails to extend the previous
    one token for token."""
    breaks = find_prefix_breaks(render_ids, conversation)
    if breaks:
        first = breaks[0]
        raise AssertionError(
            f"{len(breaks)} prefix break(s); first: the request with "
            f"{first.later} messages diverges from the one with "
            f"{first.earlier} at token {first.position}"
        )
