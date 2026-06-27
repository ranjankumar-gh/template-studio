"""The golden-token test (Chapter 10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from templatestudio.messages import Conversation
from templatestudio.parity import _first_divergence
from templatestudio.render import render

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class GoldenCase:
    """A fixed conversation and the exact wire format it must render to."""

    name: str
    conversation: Conversation
    expected: str
    add_generation_prompt: bool = True
    kwargs: dict = field(default_factory=dict)


@dataclass
class GoldenFailure:
    name: str
    first_divergence: int        # char index where actual diverged from expected
    expected: str
    actual: str


def render_case(tokenizer: "PreTrainedTokenizerBase", case: GoldenCase) -> str:
    return render(
        tokenizer,
        case.conversation,
        add_generation_prompt=case.add_generation_prompt,
        **case.kwargs,
    )


def record_goldens(
    tokenizer: "PreTrainedTokenizerBase", cases: list[GoldenCase]
) -> dict[str, str]:
    """Render each case and return its exact output. Run once on a verified template."""
    return {case.name: render_case(tokenizer, case) for case in cases}


def check_goldens(
    tokenizer: "PreTrainedTokenizerBase", cases: list[GoldenCase]
) -> list[GoldenFailure]:
    """Assert each case still renders to its recorded golden. Empty list = all pass."""
    failures: list[GoldenFailure] = []
    for case in cases:
        actual = render_case(tokenizer, case)
        if actual != case.expected:
            divergence = _first_divergence(case.expected, actual) or 0
            failures.append(
                GoldenFailure(
                    name=case.name,
                    first_divergence=divergence,
                    expected=case.expected,
                    actual=actual,
                )
            )
    return failures
