"""Report output paths must not redirect patient-derived text."""

from pathlib import Path
from stat import S_IMODE

import pytest

from mlx_reason_ct import response as response_module
from mlx_reason_ct.errors import InvalidInputError, ModelExecutionError
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


@pytest.mark.parametrize("prior", ["none", "files", "symlink"])
def test_save_response_restores_prior_pair_after_second_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior: str
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    report = output_dir / "report.txt"
    response = output_dir / "model_response.json"
    if prior != "none":
        report.write_text("old report", encoding="utf-8")
        response.write_text("old response", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    if prior == "symlink":
        outside.write_text("outside", encoding="utf-8")
        report.unlink()
        report.symlink_to(outside)

    write = response_module.write_private_text

    def fail_response(path: Path, content: str, *, overwrite: bool) -> None:
        if path == response:
            raise OSError("injected response write failure")
        write(path, content, overwrite=overwrite)

    monkeypatch.setattr(response_module, "write_private_text", fail_response)
    with pytest.raises(OSError, match="injected response write failure"):
        save_response(output_dir, "new report", {"terminated_by_eos": True}, overwrite=True)

    if prior == "symlink":
        assert report.is_symlink() and report.resolve() == outside
        assert outside.read_text(encoding="utf-8") == "outside"
    elif prior == "files":
        assert report.read_text(encoding="utf-8") == "old report"
    else:
        assert not report.exists()
    if prior == "none":
        assert not response.exists()
    else:
        assert response.read_text(encoding="utf-8") == "old response"


def test_incomplete_generation_keeps_both_response_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(ModelExecutionError, match="incomplete response"):
        save_response(output_dir, "partial finding", {"terminated_by_eos": False})

    assert (output_dir / "report.txt").read_text(encoding="utf-8") == "partial finding\n"
    assert (output_dir / "model_response.json").is_file()


def test_no_overwrite_conflict_on_response_keeps_report_absent(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    response = output_dir / "model_response.json"
    response.write_text("existing", encoding="utf-8")

    with pytest.raises(InvalidInputError, match="pass overwrite=true"):
        save_response(output_dir, "finding", {"terminated_by_eos": True})

    assert not (output_dir / "report.txt").exists()
    assert response.read_text(encoding="utf-8") == "existing"
