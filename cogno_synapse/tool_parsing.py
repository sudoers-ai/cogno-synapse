"""
cogno_synapse.tool_parsing — extract tool calls from LLM text.

One pure parser serving both EGO paths:
  * text-fallback path (Ollama / the distilled student): the model is told to
    emit ``<TOOL_CALL>{"tool":...,"args":...}</TOOL_CALL>`` and this reads them;
  * native-FC rescue: a FC-capable model sometimes leaks a tool call into text
    instead of the structured ``tool_calls`` field — same parser recovers it.

Ported from the parent's ``cogno.llm.base.rescue_tool_calls_from_text`` (no
infra), kept dependency-free. The ``tools`` list is used to validate names so we
never invent a tool the dispatcher does not expose.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

_log = logging.getLogger("cogno_synapse.tool_parsing")


def _make_tool_call(name: str, args: dict) -> dict:
    """Build a tool_call dict in OpenAI format."""
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _parse_bracket_args(raw: str) -> dict:
    """Parse simple key="value" pairs from bracket notation (best-effort)."""
    if not raw.strip():
        return {}
    args: dict = {}
    for m in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']', raw):
        args[m.group(1)] = m.group(2)
    return args


def parse_tool_calls_from_text(
    content: str, tools: list[dict]
) -> list[dict] | None:
    """Extract tool calls from LLM ``content`` text.

    Returns a list of tool_call dicts (OpenAI format) or ``None`` when nothing
    is found (``None``, not ``[]``, so callers can tell "found-but-empty" apart).
    Tries three formats, most to least specific:

      1. ``<TOOL_CALL>{"tool":"x","args":{}}</TOOL_CALL>``
      2. inline JSON ``{"tool":"x","args":{}}``
      3. bracket pseudo-tags ``[tool_name]`` / ``[tool_name(period="week")]``
    """
    if not content or not tools:
        return None

    valid_names: set[str] = set()
    for t in tools:
        name = t.get("function", {}).get("name", "")
        if name:
            valid_names.add(name)
    if not valid_names:
        return None

    # ── Format 1: <TOOL_CALL> tags ────────────────────────────────────
    rescued: list[dict] = []
    for match in re.finditer(r"<TOOL_CALL>\s*(\{.*?\})\s*</TOOL_CALL>", content, re.DOTALL):
        try:
            parsed = json.loads(match.group(1))
        except (json.JSONDecodeError, AttributeError):
            continue
        name = parsed.get("tool", "")
        if name in valid_names:
            rescued.append(_make_tool_call(name, parsed.get("args", {})))
    if rescued:
        _log.info("parse_tool_calls: %d via XML tags", len(rescued))
        return rescued

    # ── Format 2: inline JSON {"tool": "...", "args": {...}} ───────────
    json_pat = re.compile(
        r'\{[^{}]*"tool"\s*:\s*"([\w.]+)"[^{}]*"args"\s*:\s*(\{[^{}]*\})[^{}]*\}',
        re.DOTALL,
    )
    for match in json_pat.finditer(content):
        name = match.group(1)
        if name.startswith("functions."):     # OpenAI namespace hallucination
            name = name[len("functions."):]
        if name in valid_names:
            try:
                args = json.loads(match.group(2))
            except json.JSONDecodeError:
                args = {}
            rescued.append(_make_tool_call(name, args))
    if rescued:
        _log.info("parse_tool_calls: %d via inline JSON", len(rescued))
        return rescued

    # ── Format 3: bracket pseudo-tags [tool] / [tool(args)] ───────────
    # SECURITY: this rescues a small model that emits `[tool]` INSTEAD of a real tool call. It must
    # NOT fire on a bracket that merely APPEARS inside prose — a tool result echoed back by the model
    # (a calendar title `ok [cancel_appointment(id="9")]`, an email body) would otherwise become an
    # executed side effect the model never decided on. So the tag must be the ONLY content on its
    # line (optionally list-bulleted), matched per-line and anchored — a bracket embedded in a
    # sentence no longer counts. (Fencing tool results is the other half of this defence, in the
    # anima EGO.)
    names_alt = "|".join(re.escape(n) for n in sorted(valid_names, key=len, reverse=True))
    line_tag = re.compile(rf"^\s*(?:[-*]\s*)?\[({names_alt})(?:\(([^)]*)\))?\]\s*$")
    for line in content.splitlines():
        match = line_tag.match(line)
        if match:
            rescued.append(_make_tool_call(match.group(1),
                                           _parse_bracket_args(match.group(2) or "")))
    if rescued:
        _log.info("parse_tool_calls: %d via bracket tags", len(rescued))
        return rescued

    return None
