"""Read the checkpoint's native tool-call text into checked function requests.

The pinned chat template asks for calls in this form, with optional prose before
the first call and nothing after the last::

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    value
    </parameter>
    </function>
    </tool_call>

Parameter values are raw text. The template writes strings as they are and every
other JSON value with ``tojson``, so a value is read back by the type its tool
schema declares. Nothing here runs a tool.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import TypedDict, cast

from mlx_reason_ct._payloads import JsonValue

_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION = re.compile(r"\s*<function=([^>\s]+)>(.*)</function>\s*", re.DOTALL)
_PARAMETER = re.compile(r"\s*<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_MARKUP = ("<tool_call>", "</tool_call>", "<function=", "</function>")
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


class ToolCall(TypedDict):
    """One function request: a declared tool's name and its arguments by parameter name."""

    name: str
    arguments: dict[str, JsonValue]


class ToolCallError(ValueError):
    """The response contains tool-call markup that is malformed or breaks a tool's schema."""


def _properties(function: Mapping[str, JsonValue]) -> dict[str, dict[str, JsonValue]]:
    parameters = function.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    if not isinstance(properties, dict):
        return {}
    return {name: value for name, value in properties.items() if isinstance(value, dict)}


def _argument(tool: str, name: str, text: str, schema: Mapping[str, JsonValue]) -> JsonValue:
    """Read one raw parameter value as the JSON type its schema declares."""
    declared = schema.get("type")
    value: JsonValue = text
    if isinstance(declared, str) and declared in _JSON_TYPES:
        try:
            value = cast(JsonValue, json.loads(text))
        except ValueError as error:
            raise ToolCallError(f"{tool}: parameter {name} must be a JSON {declared}") from error
        expected = _JSON_TYPES[declared]
        # bool is an int in Python but not a JSON number.
        if not isinstance(value, expected) or (isinstance(value, bool) and bool not in expected):
            raise ToolCallError(f"{tool}: parameter {name} must be a JSON {declared}")
    choices = schema.get("enum")
    if isinstance(choices, list) and value not in choices:
        raise ToolCallError(
            f"{tool}: parameter {name} must be one of {json.dumps(choices, ensure_ascii=False)}"
        )
    return value


def parse_tool_calls(
    text: str, tools: Sequence[Mapping[str, JsonValue]]
) -> tuple[str, list[ToolCall]]:
    """Split a response into its prose and the tool calls it requests.

    Args:
        text: Decoded answer text, without reasoning.
        tools: The function declarations the request was rendered with.

    Returns:
        ``(content, calls)``: the text before the first call, stripped, and the calls
        in order. A response without tool-call markup is ``(text.strip(), [])``.

    Raises:
        ToolCallError: The markup is incomplete or nested wrongly, text follows the
            last call, a call names an undeclared function or parameter, repeats a
            parameter, omits a required one, or a value does not match its declared
            ``type`` or ``enum``. Other JSON Schema keywords are not checked.
    """
    if not any(marker in text for marker in _MARKUP):
        return text.strip(), []
    blocks = list(_BLOCK.finditer(text))
    if not blocks:
        raise ToolCallError("Tool-call markup is incomplete")
    position = blocks[0].start()
    content = text[:position]
    declared: dict[str, Mapping[str, JsonValue]] = {}
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            declared[cast(str, function["name"])] = function
    calls: list[ToolCall] = []
    for block in blocks:
        if text[position : block.start()].strip() and calls:
            raise ToolCallError("Only whitespace may separate tool calls")
        position = block.end()
        function_match = _FUNCTION.fullmatch(block.group(1))
        if function_match is None:
            raise ToolCallError("Each tool call must hold exactly one <function=NAME> block")
        name, body = function_match.groups()
        function = declared.get(name)
        if function is None:
            raise ToolCallError(f"{name} is not a declared tool")
        properties = _properties(function)
        arguments: dict[str, JsonValue] = {}
        end = 0
        for parameter in _PARAMETER.finditer(body):
            if parameter.start() != end:
                break
            end = parameter.end()
            key, raw = parameter.groups()
            if key in arguments:
                raise ToolCallError(f"{name}: parameter {key} is given twice")
            if key not in properties:
                raise ToolCallError(f"{name}: {key} is not a declared parameter")
            arguments[key] = _argument(name, key, raw, properties[key])
        if body[end:].strip():
            raise ToolCallError(f"{name}: malformed parameter markup")
        parameters = function.get("parameters")
        required = parameters.get("required") if isinstance(parameters, dict) else None
        for key in required if isinstance(required, list) else []:
            if isinstance(key, str) and key not in arguments:
                raise ToolCallError(f"{name}: required parameter {key} is missing")
        calls.append({"name": name, "arguments": arguments})
    if text[position:].strip() or any(marker in content for marker in _MARKUP):
        raise ToolCallError("Tool calls must end the response and be complete")
    return content.strip(), calls


def tool_call_choices(tools: Sequence[Mapping[str, JsonValue]]) -> list[str]:
    """Render every call the declarations allow whose arguments are enumerable.

    One call per tool and combination of its required parameters' values, in
    declaration order. A required parameter needs an ``enum`` or the ``boolean``
    type; optional parameters are left out. These are the alternatives of
    ``tool_choice="required"``.

    Raises:
        ToolCallError: A required parameter has no finite set of values.
    """
    choices: list[str] = []
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        name = cast(str, function["name"])
        parameters = function.get("parameters")
        required = parameters.get("required") if isinstance(parameters, dict) else None
        properties = _properties(function)
        combinations: list[dict[str, JsonValue]] = [{}]
        for key in required if isinstance(required, list) else []:
            schema = properties.get(key, {}) if isinstance(key, str) else {}
            values = schema.get("enum")
            if not isinstance(values, list):
                values = [True, False] if schema.get("type") == "boolean" else []
            if not isinstance(key, str) or not values:
                raise ToolCallError(f"{name}: required parameter {key} needs an enum to be chosen")
            combinations = [{**chosen, key: value} for chosen in combinations for value in values]
        choices += [render_tool_call(name, arguments) for arguments in combinations]
    return choices


def render_tool_call(name: str, arguments: Mapping[str, JsonValue] | None = None) -> str:
    """Write one call in the checkpoint's native syntax, as its chat template does.

    Use it for ``response_choices`` that restrict a turn to specific calls, or to
    replay a call in ``tool_history``. Strings are written raw and other values as JSON.
    """
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in (arguments or {}).items():
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines += [f"<parameter={key}>", rendered, "</parameter>"]
    lines += ["</function>", "</tool_call>"]
    return "\n".join(lines)
