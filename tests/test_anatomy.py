"""Chapter 4's differential probes, exercised against the three wire formats the
chapter prints. The templates below reproduce those renders offline."""

from __future__ import annotations

import re

from conftest import CHATML, FakeTokenizer

from templatestudio.anatomy import system_handling

# Llama 3: an explicit BOS preamble, header tags, and system in its own turn.
LLAMA3 = (
    "{{- '<|begin_of_text|>' }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
    "{{- message['content'] + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{%- endif %}"
)

# Gemma 3: no system turn at all - the system text is folded into the front of
# the first user turn, and the assistant's role is called 'model'.
GEMMA3 = (
    "{{- '<bos>' }}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{%- set prefix = messages[0]['content'] + '\n\n' %}"
    "{%- set loop_messages = messages[1:] %}"
    "{%- else %}"
    "{%- set prefix = '' %}"
    "{%- set loop_messages = messages %}"
    "{%- endif %}"
    "{%- for message in loop_messages %}"
    "{%- set role = 'model' if message['role'] == 'assistant' "
    "else message['role'] %}"
    "{{- '<start_of_turn>' + role + '\n' }}"
    "{{- (prefix if loop.first else '') + message['content'] }}"
    "{{- '<end_of_turn>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<start_of_turn>model\n' }}{%- endif %}"
)

# A template that discards the system message instead of folding it.
DROPS_SYSTEM = (
    "{%- for message in messages if message['role'] != 'system' %}"
    "{{- '<|im_start|>' + message['role'] + '\n' }}"
    "{{- message['content'] + '<|im_end|>\n' }}"
    "{%- endfor %}"
)


def test_qwen3_gives_system_its_own_turn():
    assert system_handling(FakeTokenizer(CHATML)) == "own_turn"


def test_llama3_gives_system_its_own_turn():
    assert system_handling(FakeTokenizer(LLAMA3)) == "own_turn"


def test_gemma3_folds_system_into_the_first_user_turn():
    """The finding: folding preserves the text, so has_system_role is True on
    exactly the template that costs you the policy."""
    assert system_handling(FakeTokenizer(GEMMA3)) == "folded"


def test_template_that_omits_system_reports_dropped():
    assert system_handling(FakeTokenizer(DROPS_SYSTEM)) == "dropped"


def test_gemma3_render_matches_the_wire_format_printed_in_chapter_4():
    """Pin the fixture itself: SYS_MARKER and USER_MARKER share one user turn,
    with no turn boundary between them."""
    tok = FakeTokenizer(GEMMA3)
    rendered = tok.apply_chat_template(
        [
            {"role": "system", "content": "SYS_MARKER"},
            {"role": "user", "content": "USER_MARKER"},
            {"role": "assistant", "content": "ASST_MARKER"},
        ],
        add_generation_prompt=True,
    )
    assert rendered == (
        "<bos><start_of_turn>user\n"
        "SYS_MARKER\n\nUSER_MARKER<end_of_turn>\n"
        "<start_of_turn>model\nASST_MARKER<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def test_boundary_parameter_overrides_the_default_marker_set():
    """The default marker set is the honest weak point, which is why boundary is
    a parameter: a family with its own markers must be probed with its own set."""
    exotic = (
        "{%- for message in messages %}"
        "{{- '[[TURN:' + message['role'] + ']]\n' + message['content'] + '\n' }}"
        "{%- endfor %}"
    )
    tok = FakeTokenizer(exotic)
    assert system_handling(tok) == "folded"
    assert system_handling(tok, boundary=re.compile(r"\[\[TURN:")) == "own_turn"
