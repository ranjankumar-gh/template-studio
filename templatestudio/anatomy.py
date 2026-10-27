"""Reading a template's behavior by differential probing (Chapter 4)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from templatestudio.messages import Conversation
from templatestudio.render import render

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# A probe uses distinctive markers so we can find them unambiguously in the output.
_PROBE: Conversation = [
    {"role": "system", "content": "SYS_MARKER"},
    {"role": "user", "content": "USER_MARKER"},
    {"role": "assistant", "content": "ASST_MARKER"},
]


@dataclass
class TemplateReport:
    has_system_role: bool      # does a system message survive into the wire format?
    generation_prompt: str     # exactly what add_generation_prompt appends
    declares_tools: bool       # does the template reference a tools variable?
    sample_render: str         # the probe render, for eyeballing the skeleton


def _suffix_diff(shorter: str, longer: str) -> str:
    """Return the tail of `longer` that `shorter` does not share from the front."""
    i = 0
    while i < len(shorter) and i < len(longer) and shorter[i] == longer[i]:
        i += 1
    return longer[i:]


def describe_template(tokenizer: "PreTrainedTokenizerBase") -> TemplateReport:
    """Report a template's behavior by differential probing - no source reading."""
    without_gen = render(tokenizer, _PROBE, add_generation_prompt=False)
    with_gen = render(tokenizer, _PROBE, add_generation_prompt=True)

    template_source = getattr(tokenizer, "chat_template", None) or ""
    return TemplateReport(
        has_system_role="SYS_MARKER" in without_gen,
        generation_prompt=_suffix_diff(without_gen, with_gen),
        declares_tools="tools" in template_source,
        sample_render=with_gen,
    )


# Turn boundaries for the families in this chapter. Two sentinels with a boundary
# between them are in different turns; with no boundary between them they are in
# the same turn.
_TURN_BOUNDARY = re.compile(
    r"<\|im_start\|>|<\|im_end\|>"
    r"|<\|start_header_id\|>|<\|end_header_id\|>|<\|eot_id\|>"
    r"|<start_of_turn>|<end_of_turn>"
)


def system_handling(
    tokenizer: "PreTrainedTokenizerBase",
    boundary: re.Pattern[str] = _TURN_BOUNDARY,
) -> str:
    """Report how a template treats a system message: 'own_turn', 'folded', or
    'dropped'. Folding preserves the text, so has_system_role cannot tell the
    first two apart - and folded is where a policy quietly loses force.
    Pass `boundary` for a family whose markers are not in the default set."""
    rendered = render(tokenizer, _PROBE, add_generation_prompt=False)
    sys_at = rendered.find("SYS_MARKER")
    user_at = rendered.find("USER_MARKER")
    if sys_at == -1:
        return "dropped"
    if user_at == -1:
        return "own_turn"
    lo, hi = sorted((sys_at, user_at))
    if boundary.search(rendered[lo:hi]):
        return "own_turn"
    return "folded"
