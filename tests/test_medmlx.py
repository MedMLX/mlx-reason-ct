"""run checks its arguments before any weight download."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn, cast

import medmlx_core.weights as core_weights
import pytest
from medmlx_core.outputs import validate_outputs

from mlx_reason_ct import api, medmlx
from mlx_reason_ct.errors import InvalidInputError

#: A tool whose required argument has no enum, so its calls cannot be enumerated.
OPEN_TOOL = {
    "type": "function",
    "function": {
        "name": "measure",
        "description": "Measure a finding.",
        "parameters": {
            "type": "object",
            "properties": {"finding": {"type": "string"}},
            "required": ["finding"],
        },
    },
}


@pytest.mark.parametrize(
    "options",
    [
        {"image_path": "ct.png"},
        {"precision": "float16"},
        {"precision": "float32"},
        {"max_new_tokens": 0},
        {"enable_thinking": "false"},
        {"output_dir": "existing"},
        {"anatomy_region": []},
        {"response_choices": []},
        {"response_choices": {"yes": 0}},
        {"tools": {}},
        {"tools": [{"type": "invalid"}]},
        {"tool_history": [{"role": "assistant", "content": "call"}]},
        {"tool_choice": "required"},
        {"tools": [OPEN_TOOL], "tool_choice": "required"},
        {"tools": [OPEN_TOOL], "tool_choice": "auto"},
    ],
    ids=[
        "suffix",
        "choice",
        "removed-float32",
        "range",
        "boolean",
        "existing-output",
        "region-type",
        "empty-choices",
        "choices-mapping",
        "tools-mapping",
        "tool-schema",
        "tool-history",
        "tool-choice-without-tools",
        "tool-choice-open-parameter",
        "tool-choice-value",
    ],
)
def test_run_rejects_invalid_inputs_before_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: dict[str, object]
) -> None:
    """Refuse each category of bad argument without starting the 17 GB download."""
    image = tmp_path / "ct.nii.gz"
    image.write_bytes(b"unread")
    (tmp_path / "ct.png").write_bytes(b"unread")
    (tmp_path / "existing").mkdir()
    (tmp_path / "existing/report.txt").write_text("keep")

    def download(model_id: str, **_options: object) -> NoReturn:
        raise AssertionError(f"downloaded {model_id} before checking arguments")

    monkeypatch.setattr(core_weights, "weights_dir", download)
    arguments: dict[str, object] = {"image_path": image, "output_dir": tmp_path / "out"}
    for key, value in options.items():
        if key in {"image_path", "output_dir"} and isinstance(value, str):
            value = tmp_path / value
        arguments[key] = value
    # Invalid argument types are the point of this test, so call without static checks.
    run = cast(Callable[..., object], medmlx.run)
    with pytest.raises(InvalidInputError):
        run(**arguments)


def test_run_returns_answer_fields_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The result carries the written answer and describes all three files as reports."""
    image = tmp_path / "ct.nii.gz"
    image.write_bytes(b"unread")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    answer: dict[str, object] = {
        "schema_version": 1,
        "report": "No acute finding.",
        "thinking": "Lungs are clear.",
        "tool_calls": [{"name": "detect_nodules", "arguments": {}}],
        "requires_human_review": True,
        "generation": {"terminated_by_eos": True},
    }

    def generate(source: Path, output_dir: Path, **_: object) -> dict[str, object]:
        output_dir.mkdir()
        (output_dir / "model_response.json").write_text(json.dumps(answer), encoding="utf-8")
        return {"status": "succeeded"}

    monkeypatch.setattr(api, "write_report", generate)
    result = medmlx.run(
        image_path=image, weights_path=bundle, output_dir=tmp_path / "out"
    )

    paths, artifacts = validate_outputs(result)
    assert set(paths) == {"report", "response", "metadata"}
    assert {entry["kind"] for entry in artifacts.values()} == {"report"}
    assert {entry["coordinate_space"] for entry in artifacts.values()} == {"not_applicable"}
    assert result["details"] == {
        "status": "succeeded",
        "report": "No acute finding.",
        "thinking": "Lungs are clear.",
        "tool_calls": [{"name": "detect_nodules", "arguments": {}}],
    }
