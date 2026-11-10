"""Durable report text and completion contract for native MLX generation."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from mlx_reason_ct.errors import InvalidInputError, ModelExecutionError

THINK_END = "</think>"


@dataclass(frozen=True, slots=True)
class ReportFiles:
    """The report and provenance files owned by one generation request."""

    directory: Path

    def outputs(self) -> dict[str, str]:
        """Return the ``report``, ``response`` and ``run`` file paths as POSIX strings."""
        return {
            "report": (self.directory / "report.txt").as_posix(),
            "response": (self.directory / "model_response.json").as_posix(),
            "run": (self.directory / "run.json").as_posix(),
        }

    def require_writable(self, *, overwrite: bool) -> None:
        """Raise ``InvalidInputError`` if the directory or an output path is unusable.

        An existing output file (or symlink) is an error unless ``overwrite`` is true.
        """
        if self.directory.exists() and not self.directory.is_dir():
            raise InvalidInputError(f"output_dir is not a directory: {self.directory}")
        for value in self.outputs().values():
            path = Path(value)
            if path.is_dir():
                raise InvalidInputError(f"Output is a directory: {path}; choose another output_dir")
            if (path.exists() or path.is_symlink()) and not overwrite:
                raise InvalidInputError(f"Output exists: {path}; pass overwrite=true to replace it")


def write_private_text(path: Path, content: str, *, overwrite: bool) -> None:
    """Publish one private output without following a target symlink.

    The temporary file is created in the destination directory with mode 0600.
    Linking gives no-overwrite writes an atomic existence check; replacement
    swaps the directory entry without opening the prior target.
    """
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(content)
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise InvalidInputError(
                    f"Output exists: {path}; pass overwrite=true to replace it"
                ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def restore_outputs_on_failure(paths: Sequence[Path]) -> Generator[list[Path]]:
    """Restore previously published paths if a later file write fails."""
    recovery = Path(tempfile.mkdtemp(prefix=".mlx-reason-ct-backup-", dir=paths[0].parent))
    cleanup = True
    try:
        backups: dict[Path, Path] = {}
        for index, path in enumerate(paths):
            if path.exists() or path.is_symlink():
                backup = recovery / str(index)
                os.link(path, backup, follow_symlinks=False)
                backups[path] = backup
        published: list[Path] = []
        try:
            yield published
        except BaseException:
            try:
                for path in reversed(published):
                    backup = backups.get(path)
                    if backup is None:
                        path.unlink(missing_ok=True)
                    else:
                        os.replace(backup, path)
            except OSError as error:
                cleanup = False
                raise OSError(
                    f"Cannot restore report files; backups retained at {recovery}"
                ) from error
            raise
    finally:
        if cleanup:
            shutil.rmtree(recovery)


def split_thinking(response: str) -> tuple[str | None, str]:
    """Separate template thinking-mode reasoning from the final answer.

    With thinking enabled the template opens ``<think>`` in the prompt, so the
    decoded output is ``reasoning</think>answer``. Without the closing tag the
    whole response is the answer.
    """
    if THINK_END not in response:
        return None, response
    reasoning, answer = response.split(THINK_END, 1)
    return reasoning.strip(), answer.strip()


def save_response(
    output_dir: Path,
    response: str,
    metrics: Mapping[str, object],
    *,
    overwrite: bool = False,
    literal: bool = False,
) -> dict[str, str]:
    """Write ``report.txt`` and ``model_response.json`` from decoded model text.

    ``response`` is split into optional thinking text and the answer, and a
    trailing empty ``<answer></answer>`` element is removed from the answer.
    With ``literal=True``, preserve the decoded constrained choice without that
    markup cleanup.
    ``metrics`` is stored under ``generation`` and must contain
    ``terminated_by_eos``. The JSON record always sets ``requires_human_review``.

    Returns:
        The ``report`` and ``response`` paths.

    Raises:
        ModelExecutionError: the answer is empty or generation did not end at an
            EOS token. Both files are written before this is raised.
    """
    outputs = ReportFiles(output_dir).outputs()
    report_path = Path(outputs["report"])
    response_path = Path(outputs["response"])
    thinking, report = (None, response) if literal else split_thinking(response)
    if not literal:
        report = re.sub(r"\s*<answer>\s*</answer>\s*$", "", report)
    record: dict[str, object] = {"schema_version": 1, "report": report}
    if thinking is not None:
        record["thinking"] = thinking
    record.update(requires_human_review=True, generation=metrics)
    response_content = json.dumps(record, indent=2) + "\n"
    complete = bool(report.strip()) and bool(metrics["terminated_by_eos"])
    with restore_outputs_on_failure((report_path, response_path)) as published:
        write_private_text(report_path, report + "\n", overwrite=overwrite)
        published.append(report_path)
        write_private_text(response_path, response_content, overwrite=overwrite)
        published.append(response_path)
    if not complete:
        raise ModelExecutionError(
            "NV-Reason-CT returned an empty or incomplete response. "
            "Partial output was retained; increase max_new_tokens for a truncated response."
        )
    return {"report": str(report_path), "response": str(response_path)}
