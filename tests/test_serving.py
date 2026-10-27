from __future__ import annotations

from templatestudio.jinja_env import render_template_string
from templatestudio.prefix import assert_prefix_stable
from templatestudio.serving import (
    LlamaCppTokens,
    VLLMTokens,
    compare_token_renders,
)

BOS = 1
TEMPLATE = (
    "{{- '<s>' }}"
    "{% for m in messages %}"
    "{{- '<|' + m['role'] + '|>' + m['content'] + '<|end|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|assistant|>' }}{%- endif %}"
)
SPECIALS = {"<s>": BOS, "<|user|>": 2, "<|assistant|>": 3, "<|end|>": 4}


def encode(text: str, add_special: bool, gguf_adds_bos: bool) -> list[int]:
    ids, i = [], 0
    while i < len(text):
        hit = next((s for s in SPECIALS if text.startswith(s, i)), None)
        if hit:
            ids.append(SPECIALS[hit])
            i += len(hit)
        else:
            ids.append(ord(text[i]))
            i += 1
    return ([BOS] if add_special and gguf_adds_bos else []) + ids


class Reference:
    name = "transformers"

    def ids(self, conversation, **kwargs):
        text = render_template_string(TEMPLATE, conversation, **kwargs)
        return encode(text, add_special=False, gguf_adds_bos=False)


def fake_llama_cpp(gguf_adds_bos: bool):
    calls: list[tuple[str, dict]] = []

    def transport(url: str, body: dict) -> dict:
        calls.append((url, body))
        if url.endswith("/apply-template"):
            kwargs = body.get("chat_template_kwargs", {})
            return {"prompt": render_template_string(
                TEMPLATE, body["messages"], **kwargs)}
        return {"tokens": encode(body["content"], body["add_special"], gguf_adds_bos)}

    return transport, calls


def fake_vllm(url: str, body: dict) -> dict:
    assert url.endswith("/tokenize") and body["model"] == "m"
    text = render_template_string(
        TEMPLATE, body["messages"],
        add_generation_prompt=body["add_generation_prompt"],
        **body.get("chat_template_kwargs", {}),
    )
    return {"tokens": encode(text, add_special=False, gguf_adds_bos=True)}


CONVO = [{"role": "user", "content": "hi"}]


def test_llama_cpp_mirrors_the_chat_path_tokenization():
    transport, calls = fake_llama_cpp(gguf_adds_bos=False)
    LlamaCppTokens("http://x/", transport).ids(CONVO)
    assert [u for u, _ in calls] == ["http://x/apply-template", "http://x/tokenize"]
    assert calls[1][1]["add_special"] is True and calls[1][1]["parse_special"] is True


def test_engines_that_agree_report_nothing():
    transport, _ = fake_llama_cpp(gguf_adds_bos=False)
    engines = [Reference(), LlamaCppTokens("http://x", transport),
               VLLMTokens("http://y", "m", fake_vllm)]
    assert compare_token_renders(engines, CONVO) == []


def test_a_gguf_that_adds_bos_doubles_it_at_position_zero():
    transport, _ = fake_llama_cpp(gguf_adds_bos=True)
    (div,) = compare_token_renders(
        [Reference(), LlamaCppTokens("http://x", transport)], CONVO)
    assert div.engine == "llama.cpp" and div.position == 1
    assert div.engine_len == div.reference_len + 1


def test_template_kwargs_reach_the_engine():
    transport, calls = fake_llama_cpp(gguf_adds_bos=False)
    LlamaCppTokens("http://x", transport).ids(CONVO, enable_thinking=False)
    assert calls[0][1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_prefix_check_runs_on_an_engine_adapter():
    transport, _ = fake_llama_cpp(gguf_adds_bos=False)
    engine = LlamaCppTokens("http://x", transport)
    convo = CONVO + [{"role": "assistant", "content": "hello"},
                     {"role": "user", "content": "bye"}]
    assert_prefix_stable(engine.ids, convo)
