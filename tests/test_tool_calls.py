"""Tool requests reach the caller as checked calls, in the checkpoint's native syntax."""

import json
from pathlib import Path

import pytest

from mlx_reason_ct._payloads import JsonValue
from mlx_reason_ct.response import save_response
from mlx_reason_ct.tool_calls import tool_call_choices

TOOLS: list[dict[str, JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": "detect_nodules",
            "description": "Detect lung nodules.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "segment_organ",
            "description": "Segment an organ.",
            "parameters": {
                "type": "object",
                "properties": {
                    "organ": {"type": "string", "enum": ["liver", "left lung"]},
                    "margin_mm": {"type": "number"},
                    "label": {"type": "string"},
                },
                "required": ["organ"],
            },
        },
    },
]
LIVER = "<tool_call>\n<function=segment_organ>\n<parameter=organ>\nliver\n</parameter>\n"


def response(tmp_path: Path, text: str) -> dict[str, object]:
    save_response(tmp_path, text, {"terminated_by_eos": True}, tools=TOOLS)
    record: dict[str, object] = json.loads((tmp_path / "model_response.json").read_text())
    return record


ANSWERS: list[tuple[str, list[dict[str, object]]]] = [
    ("No focal liver lesion.", []),
    (
        "<tool_call>\n<function=detect_nodules>\n</function>\n</tool_call>",
        [{"name": "detect_nodules", "arguments": {}}],
    ),
    (
        # Prose may precede a call; a number is read as JSON and a string kept raw,
        # including its inner line break.
        "Measuring.\n\n" + LIVER + "<parameter=margin_mm>\n2.5\n</parameter>\n"
        "<parameter=label>\nsegment IV\nand VIII\n</parameter>\n</function>\n</tool_call>",
        [
            {
                "name": "segment_organ",
                "arguments": {
                    "organ": "liver",
                    "margin_mm": 2.5,
                    "label": "segment IV\nand VIII",
                },
            }
        ],
    ),
    (
        "<tool_call>\n<function=detect_nodules>\n</function>\n</tool_call>\n"
        + LIVER
        + "</function>\n</tool_call>",
        [
            {"name": "detect_nodules", "arguments": {}},
            {"name": "segment_organ", "arguments": {"organ": "liver"}},
        ],
    ),
]


@pytest.mark.parametrize(
    ("text", "calls"),
    ANSWERS,
    ids=["prose", "no-arguments", "typed-arguments", "two-calls"],
)
def test_response_lists_the_requested_calls(
    tmp_path: Path, text: str, calls: list[dict[str, object]]
) -> None:
    record = response(tmp_path, text)
    assert record["tool_calls"] == calls
    assert "tool_call_error" not in record


@pytest.mark.parametrize(
    "text",
    [
        LIVER + "</function>",
        LIVER + "</function>\n</tool_call>\nThe liver is enlarged.",
        "<tool_call>\n<function=delete_scan>\n</function>\n</tool_call>",
        "<tool_call>\n<function=segment_organ>\n</function>\n</tool_call>",
        "<tool_call>\n<function=segment_organ>\n<parameter=organ>\nspleen\n</parameter>\n"
        "</function>\n</tool_call>",
        LIVER + "<parameter=margin_mm>\nwide\n</parameter>\n</function>\n</tool_call>",
        LIVER + "<parameter=path>\n/etc/passwd\n</parameter>\n</function>\n</tool_call>",
        LIVER + "<parameter=organ>\nleft lung\n</parameter>\n</function>\n</tool_call>",
    ],
    ids=[
        "unclosed",
        "text-after-call",
        "undeclared-tool",
        "missing-required",
        "outside-enum",
        "wrong-type",
        "undeclared-parameter",
        "repeated-parameter",
    ],
)
def test_unusable_tool_markup_is_reported_instead_of_returned_as_a_call(
    tmp_path: Path, text: str
) -> None:
    record = response(tmp_path, text)
    assert record["tool_calls"] == []
    assert isinstance(record["tool_call_error"], str)
    # The answer text stays available for the caller's own feedback to the model.
    assert record["report"] == text


def test_required_tool_choice_offers_each_enumerated_call_in_native_syntax() -> None:
    # The bytes the pinned chat template renders for a call; decoding is constrained
    # to them and tool_history replays them.
    assert tool_call_choices(TOOLS) == [
        "<tool_call>\n<function=detect_nodules>\n</function>\n</tool_call>",
        "<tool_call>\n<function=segment_organ>\n<parameter=organ>\nliver\n</parameter>\n"
        "</function>\n</tool_call>",
        "<tool_call>\n<function=segment_organ>\n<parameter=organ>\nleft lung\n</parameter>\n"
        "</function>\n</tool_call>",
    ]
