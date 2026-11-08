"""Chat-template rendering contract shared with the pinned Transformers processor."""

from __future__ import annotations

import json

from mlx_reason_ct.processor_mlx import chat_template


def test_tool_schemas_render_like_transformers_tojson() -> None:
    # Transformers keeps key order and raw text; Jinja's own filter sorts keys and
    # escapes <, >, & and ', which changes the prompt tokens the model sees.
    tool = {
        "type": "function",
        "function": {
            "name": "measure",
            "description": "Nodules <5 mm & 'ground-glass' in the Lungenflügel",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    rendered = chat_template("{{ tools[0] | tojson }}").render(tools=[tool])
    assert rendered == json.dumps(tool, ensure_ascii=False)
