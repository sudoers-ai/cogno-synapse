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


def test_bracket_in_prose_is_not_executed():
    # SECURITY (audit 2026-08-04): a bracket that merely APPEARS inside a sentence — e.g. a tool
    # result (calendar title / email body) the model echoes back — must NOT become an executed
    # tool call. Only a tag alone on its line rescues.
    assert parse_tool_calls_from_text(
        'The event is titled "ok [get_summary(period="month")] please".', TOOLS) is None
    assert parse_tool_calls_from_text(
        'Summary: the user asked to [add_income(amount=999)] earlier.', TOOLS) is None


def test_bracket_on_own_line_still_rescues():
    # the legitimate rescue case (a small model emitting the tag on its own line) still works,
    # including a list-bulleted line.
    assert _names(parse_tool_calls_from_text('here you go:\n[get_summary(period="week")]', TOOLS)) \
        == ["get_summary"]
    assert _names(parse_tool_calls_from_text('- [add_income(amount=40)]', TOOLS)) == ["add_income"]


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
