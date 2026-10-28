"""The golden-token test (Chapter 11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from templatestudio.messages import Conversation
from templatestudio.parity import first_divergence
from templatestudio.render import render
from templatestudio.serving import TokenRenderer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class GoldenCase:
    """A fixed conversation, the exact wire format it must render to, and -
    when recorded from a serving engine - the exact token ids that engine
    sends the model."""

    name: str
    conversation: Conversation
    expected: str
    add_generation_prompt: bool = True
    kwargs: dict = field(default_factory=dict)
    expected_ids: tuple[int, ...] | None = None
    engine: str = ""


@dataclass
class GoldenFailure:
    name: str
    first_divergence: int  # index where actual diverged from expected
    expected: str
    actual: str
    unit: str = "char"     # "token" when the engine's ids diverged


def render_case(
    tokenizer: "PreTrainedTokenizerBase", case: GoldenCase
) -> str:
    return render(
        tokenizer,
        case.conversation,
        add_generation_prompt=case.add_generation_prompt,
        **case.kwargs,
    )


def record_goldens(
    tokenizer: "PreTrainedTokenizerBase", cases: list[GoldenCase]
) -> dict[str, str]:
    """Render each case and return its exact output. Run once on a
    verified template; store the result as the golden the test asserts
    against forever after."""
    return {case.name: render_case(tokenizer, case) for case in cases}


def record_golden_ids(
    engine: TokenRenderer, cases: list[GoldenCase]
) -> dict[str, tuple[int, ...]]:
    """Ask the serving engine for each case's token ids. These are what the
    model receives; store them beside the string goldens. An engine renders
    with the generation prompt, as a served request does."""
    return {
        case.name: tuple(engine.ids(case.conversation, **case.kwargs))
        for case in cases
    }


def _token_failure(
    engine: TokenRenderer, case: GoldenCase
) -> GoldenFailure | None:
    assert case.expected_ids is not None
    actual = tuple(engine.ids(case.conversation, **case.kwargs))
    if actual == case.expected_ids:
        return None
    expected = case.expected_ids
    at = next(
        (i for i, (a, b) in enumerate(zip(expected, actual)) if a != b),
        min(len(expected), len(actual)),
    )
    return GoldenFailure(
        name=case.name,
        first_divergence=at,
        expected=str(list(expected[at : at + 8])),
        actual=str(list(actual[at : at + 8])),
        unit="token",
    )


def check_goldens(
    tokenizer: "PreTrainedTokenizerBase",
    cases: list[GoldenCase],
    *,
    engine: TokenRenderer | None = None,
) -> list[GoldenFailure]:
    """Assert each case still produces its goldens. With an engine, a case
    that carries expected_ids is checked token for token against what the
    engine sends; every case is checked against its string. Empty = pass."""
    failures: list[GoldenFailure] = []
    for case in cases:
        if engine is not None and case.expected_ids is not None:
            token_failure = _token_failure(engine, case)
            if token_failure is not None:
                failures.append(token_failure)
        actual = render_case(tokenizer, case)
        if actual != case.expected:
            divergence = first_divergence(case.expected, actual) or 0
            failures.append(
                GoldenFailure(
                    name=case.name,
                    first_divergence=divergence,
                    expected=case.expected,
                    actual=actual,
                )
            )
    return failures
