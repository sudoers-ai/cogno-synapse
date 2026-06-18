"""Unit tests for the shared tool-call text parser."""

from cogno_synapse.tool_parsing import parse_tool_calls_from_text

TOOLS = [
    {"function": {"name": "add_income"}},
    {"function": {"name": "get_summary"}},
]


def _names(calls):
    return [c["function"]["name"] for c in calls]


def test_xml_tag_format():
    text = 'sure <TOOL_CALL>{"tool": "add_income", "args": {"amount": 40}}</TOOL_CALL>'
    calls = parse_tool_calls_from_text(text, TOOLS)
    assert _names(calls) == ["add_income"]
    import json
    assert json.loads(calls[0]["function"]["arguments"]) == {"amount": 40}


def test_inline_json_format():
    text = 'I will call {"tool": "get_summary", "args": {"period": "week"}} now'
    calls = parse_tool_calls_from_text(text, TOOLS)
    assert _names(calls) == ["get_summary"]


def test_bracket_format():
    calls = parse_tool_calls_from_text('[get_summary(period="month")]', TOOLS)
    assert _names(calls) == ["get_summary"]
    import json
    assert json.loads(calls[0]["function"]["arguments"]) == {"period": "month"}


def test_unknown_tool_name_ignored():
    # name not in the valid set → not rescued (avoids hallucinated tools)
    assert parse_tool_calls_from_text('<TOOL_CALL>{"tool": "drop_db", "args": {}}</TOOL_CALL>', TOOLS) is None


def test_namespace_hallucination_stripped():
    text = '{"tool": "functions.add_income", "args": {"amount": 1}}'
    calls = parse_tool_calls_from_text(text, TOOLS)
    assert _names(calls) == ["add_income"]


def test_no_tools_or_empty_returns_none():
    assert parse_tool_calls_from_text("anything", []) is None
    assert parse_tool_calls_from_text("", TOOLS) is None


def test_plain_text_returns_none():
    assert parse_tool_calls_from_text("just a normal answer, no tools", TOOLS) is None


def test_multiple_xml_calls():
    text = ('<TOOL_CALL>{"tool":"add_income","args":{"amount":1}}</TOOL_CALL>'
            '<TOOL_CALL>{"tool":"get_summary","args":{}}</TOOL_CALL>')
    calls = parse_tool_calls_from_text(text, TOOLS)
    assert _names(calls) == ["add_income", "get_summary"]
