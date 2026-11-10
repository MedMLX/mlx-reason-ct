"""Report output paths must not redirect patient-derived text."""

from pathlib import Path
from stat import S_IMODE

import pytest

from mlx_reason_ct.response import save_response


@pytest.mark.parametrize("name", ["report.txt", "model_response.json"])
def test_overwrite_replaces_output_symlink_without_writing_target(
    tmp_path: Path, name: str
) -> None:
    """An existing output symlink cannot redirect a report into another file."""
    destination = tmp_path / "destination.txt"
    destination.write_text("preserve")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / name).symlink_to(destination)

    save_response(output_dir, "finding", {"terminated_by_eos": True}, overwrite=True)

    assert destination.read_text() == "preserve"
    assert not (output_dir / name).is_symlink()
    assert S_IMODE((output_dir / name).stat().st_mode) == 0o600
