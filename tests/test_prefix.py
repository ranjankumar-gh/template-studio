from __future__ import annotations

import pytest

from templatestudio.jinja_env import render_template_string
from templatestudio.prefix import (
    assert_prefix_stable,
    find_prefix_breaks,
    request_points,
)

CHATML = (
    "{% for m in messages %}"
    "{{- '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)

# A template that stamps the turn count into the system header: the prefix
# moves on every request, exactly like a date stamp moves at midnight.
STAMPED = "{{- 'turns: ' + (messages|length)|string + '\n' }}" + CHATML

CONVO = [
    {"role": "user", "content": "Where is my order?"},
    {"role": "assistant", "content": "Let me check."},
    {"role": "user", "content": "It was #4471."},
    {"role": "assistant", "content": "Found it."},
    {"role": "user", "content": "Thanks."},
]


def ids_for(template: str):
    def render_ids(conversation):
        text = render_template_string(template, conversation)
        return [ord(c) for c in text]

    return render_ids


def test_request_points_skip_assistant_turns():
    assert request_points(CONVO) == [1, 3, 5]


def test_a_leading_system_message_is_not_a_request():
    convo = [{"role": "system", "content": "be terse"}] + CONVO
    assert request_points(convo) == [2, 4, 6]
    assert find_prefix_breaks(ids_for(CHATML), convo) == []


def test_plain_chatml_is_prefix_stable():
    assert find_prefix_breaks(ids_for(CHATML), CONVO) == []
    assert_prefix_stable(ids_for(CHATML), CONVO)


def test_a_turn_counter_breaks_the_prefix_at_its_first_token():
    breaks = find_prefix_breaks(ids_for(STAMPED), CONVO)
    assert [(b.earlier, b.later) for b in breaks] == [(1, 3), (3, 5)]
    assert breaks[0].position == len("turns: ")
    with pytest.raises(AssertionError, match="2 prefix break"):
        assert_prefix_stable(ids_for(STAMPED), CONVO)
