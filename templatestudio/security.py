"""Templates as an attack surface: provenance, scanning, injection (Chapter 12)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def template_fingerprint(template_source: str) -> str:
    """A stable fingerprint of a template's source. Record the
    fingerprint of the template you reviewed; a changed fingerprint means
    re-review before trusting."""
    return hashlib.sha256(template_source.encode("utf-8")).hexdigest()


def assert_pinned(template_source: str, expected_fingerprint: str) -> None:
    """Fail if a template's source no longer matches the
    reviewed-and-pinned version."""
    actual = template_fingerprint(template_source)
    if actual != expected_fingerprint:
        raise ValueError(
            f"template fingerprint changed: "
            f"expected {expected_fingerprint[:12]}..., "
            f"got {actual[:12]}... - re-review before trusting"
        )


def template_sources(tokenizer: Any) -> dict[str, str]:
    """Every template this tokenizer can render with, by name. A dict of
    named templates is pinned entry by entry: Transformers picks tool_use
    when tools are passed, so pinning only the default lets a backdoored
    tool_use ship. No template at all fails loudly, never silently."""
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("no chat template found: nothing to pin or to render")
    if isinstance(template, Mapping):
        return {str(name): str(source) for name, source in template.items()}
    return {"default": str(template)}


def assert_all_pinned(
    sources: Mapping[str, str], pins: Mapping[str, str]
) -> None:
    """assert_pinned for every named template, and fail on any template
    that has no pin - an unreviewed template is the one to worry about."""
    for name, source in sources.items():
        if name not in pins:
            raise ValueError(f"template {name!r} has no pin - review it first")
        assert_pinned(source, pins[name])


import re

from jinja2 import nodes
from jinja2.exceptions import TemplateSyntaxError

from templatestudio.jinja_env import build_template_env

_URL = re.compile(r"https?://")
_LONG_LITERAL = 40


def _literal(node: nodes.Node, ctx: nodes.EvalContext) -> str | None:
    """The string an expression always evaluates to, folding 'ht' ~ 'tps'
    the way the compiler would; None if it depends on the conversation."""
    if isinstance(node, nodes.TemplateData):
        return node.data
    try:
        value = node.as_const(ctx)  # type: ignore[attr-defined]
    except (nodes.Impossible, AttributeError):
        return None
    return value if isinstance(value, str) else None


def _reads_content(node: nodes.Node, tainted: set[str]) -> bool:
    """Does this expression read message content, directly or through a
    variable that was set from it ({% set c = messages[-1]['content'] %})?"""
    for n in (node, *node.find_all(nodes.Node)):  # find_all skips node itself
        if isinstance(n, nodes.Name) and n.name in tainted:
            return True
        if isinstance(n, nodes.Getattr) and n.attr == "content":
            return True
        if isinstance(n, nodes.Getitem) and isinstance(n.arg, nodes.Const):
            if n.arg.value == "content":
                return True
    return False


def scan_template(template_source: str) -> list[str]:
    """Flag backdoor-shaped constructs in a template's parsed source. A hit
    means review, not proof of malice; a clean scan is not proof of safety.
    It reads the syntax tree, so whitespace control, ~ concatenation,
    aliasing with set, and literal text outside tags do not hide a shape."""
    env = build_template_env()
    try:
        tree = env.parse(template_source)
    except TemplateSyntaxError as error:
        return [f"template does not parse: {error.message}"]
    ctx = nodes.EvalContext(env)
    findings: list[str] = []

    def flag(description: str) -> None:
        if description not in findings:
            findings.append(description)

    for node in tree.find_all((nodes.TemplateData, nodes.Const)):
        text = _literal(node, ctx)
        if text and _URL.search(text):
            flag("hardcoded URL in template source")
    for output in tree.find_all(nodes.Output):
        for child in output.nodes:
            text = _literal(child, ctx)
            if text and _URL.search(text):
                flag("hardcoded URL in template source")
            if text and len(text.strip()) >= _LONG_LITERAL:
                flag("long literal string injected into output")

    tainted: set[str] = set()
    for assign in tree.find_all(nodes.Assign):
        if isinstance(assign.target, nodes.Name) and _reads_content(
            assign.node, tainted
        ):
            tainted.add(assign.target.name)
    tests = [n.test for n in tree.find_all((nodes.If, nodes.CondExpr))]
    for test in tests:
        if test is not None and _reads_content(test, tainted) and any(
            isinstance(c.value, str) and c.value != "content"
            for c in test.find_all(nodes.Const)
        ):
            flag("branch on message content - possible trigger condition")
    return findings


# The fallback when you scan before you know which model will render: turn and
# control markers across the families from Chapter 4, gpt-oss Harmony's, and
# Qwen3's tool and reasoning tags. With the deployed tokenizer, use
# control_markers instead - it cannot miss a token the model actually has.
_CONTROL_MARKERS = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>"
    r"|<\|begin_of_text\|>|<\|start_header_id\|>|<\|end_header_id\|>"
    r"|<\|eot_id\|>|<\|eom_id\|>|<\|python_tag\|>"
    r"|<bos>|<eos>|<start_of_turn>|<end_of_turn>"
    r"|<\|start\|>|<\|channel\|>|<\|message\|>|<\|call\|>|<\|end\|>"
    r"|<\|return\|>|<\|constrain\|>"
    r"|</?tool_call>|</?tool_response>|</?think>"
)


def control_markers(tokenizer: Any) -> list[str]:
    """Every added token the tokenizer parses as a single id, special or not,
    longest first. all_special_tokens is not enough: Qwen3 adds <tool_call>,
    <tool_response>, and <think> with special=False, and they still tokenize
    atomically - so user text containing them forges real structure."""
    tokens = {t.content for t in tokenizer.added_tokens_decoder.values()}
    return sorted(tokens, key=len, reverse=True)


def find_control_tokens(content: str, tokenizer: Any = None) -> list[str]:
    """Find control/turn markers embedded in message content - a
    structure-forging injection attempt. Pass the deployed tokenizer to
    check against every token it treats as one id."""
    if tokenizer is None:
        return _CONTROL_MARKERS.findall(content)
    pattern = "|".join(re.escape(m) for m in control_markers(tokenizer))
    return re.findall(pattern, content) if pattern else []


def request_strings(
    conversation: list[dict], tools: list[dict] | None = None
) -> list[tuple[str, str]]:
    """Every string a request lets an attacker put into the render, labeled
    by where it came from: content and text parts, names, reasoning, tool-call
    names and arguments, and the tool schemas' descriptions."""
    found: list[tuple[str, str]] = []
    for i, message in enumerate(conversation):
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"text": content}]
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                found.append((f"message {i} content", part["text"]))
        for key in ("name", "reasoning_content", "reasoning"):
            if isinstance(message.get(key), str):
                found.append((f"message {i} {key}", message[key]))
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            args = function.get("arguments")
            text = args if isinstance(args, str) else json.dumps(args)
            found.append((f"message {i} tool call", f"{function.get('name')} {text}"))
    for j, tool in enumerate(tools or []):
        found.append((f"tool {j} schema", json.dumps(tool, ensure_ascii=False)))
    return found


def find_forged_tokens(
    conversation: list[dict], tools: list[dict] | None, tokenizer: Any
) -> list[str]:
    """find_control_tokens over every request-reachable string. Empty means
    nothing in the request can become structure when it is tokenized."""
    return [
        f"{where}: {marker}"
        for where, text in request_strings(conversation, tools)
        for marker in find_control_tokens(text, tokenizer)
    ]


def neutralize(text: str, tokenizer: Any) -> str:
    """Break every control marker in text with a zero-width space after its
    first character, so it tokenizes as ordinary text, not as the marker.
    Escaping the string does not help: the injection happens when the whole
    render is tokenized with special tokens parsed."""
    for marker in control_markers(tokenizer):
        text = text.replace(marker, marker[0] + "​" + marker[1:])
    return text
