"""Runner arguments are checked before any weight download."""

from pathlib import Path
from typing import NoReturn

import pytest

from mlx_reason_ct import medmlx
from mlx_reason_ct.errors import InvalidInputError


@pytest.mark.parametrize(
    "options",
    [
        {"image_path": "ct.png"},
        {"precision": "float16"},
        {"max_new_tokens": 0},
        {"enable_thinking": "false"},
        {"output_dir": "existing"},
    ],
    ids=["suffix", "choice", "range", "boolean", "existing-output"],
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

    def download(model_id: str) -> NoReturn:
        raise AssertionError(f"downloaded {model_id} before checking arguments")

    monkeypatch.setattr(medmlx, "weights_dir", download)
    arguments: dict[str, object] = {"image_path": image, "output_dir": tmp_path / "out"}
    for key, value in options.items():
        if key in {"image_path", "output_dir"} and isinstance(value, str):
            value = tmp_path / value
        arguments[key] = value
    with pytest.raises(InvalidInputError):
        medmlx.run(**arguments)  # pyright: ignore[reportArgumentType]
