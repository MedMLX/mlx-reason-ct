"""Admit only processor geometry supported by the native model."""

import json
from pathlib import Path

import pytest

from mlx_reason_ct.errors import InvalidInputError
from mlx_reason_ct.processor_mlx import Processor


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_grid_size", [0, 24, 24]),
        ("final_grid_size", [24.0, 24, 24]),
        ("spatial_size", [192, 192, 1000000]),
        ("pixdim", [0, 2, 2]),
        ("pixdim", [float("nan"), 2, 2]),
        ("pixdim", [10**400, 2, 2]),
    ],
)
def test_processor_rejects_unsupported_geometry(tmp_path: Path, field: str, value: object) -> None:
    # A real minimal tokenizer keeps this check at model-folder admission.
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({
            "model": {
                "type": "WordLevel", "vocab": {"<|image_pad|>": 0}, "unk_token": "<|image_pad|>"
            }
        })
    )
    (tmp_path / "chat_template.jinja").write_text("<|image_pad|>")
    config: dict[str, object] = {
        "final_grid_size": [24, 24, 24],
        "spatial_size": [192, 192, 192],
        "pixdim": [2.0, 2.0, 2.0],
    }
    processor_dir = tmp_path / "image_processor_3d"
    processor_dir.mkdir()
    config_path = processor_dir / "preprocessor_config.json"
    config_path.write_text(json.dumps(config))
    processor = Processor(tmp_path)
    assert processor.grid == (24, 24, 24)
    config[field] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(InvalidInputError):
        Processor(tmp_path)
