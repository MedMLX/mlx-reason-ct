"""Durable report text and completion contract for native MLX generation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
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


def save_response(output_dir: Path, response: str, metrics: Mapping[str, object]) -> dict[str, str]:
    """Write ``report.txt`` and ``model_response.json`` from decoded model text.

    ``response`` is split into optional thinking text and the answer, and a
    trailing empty ``<answer></answer>`` element is removed from the answer.
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
    thinking, report = split_thinking(response)
    report = re.sub(r"\s*<answer>\s*</answer>\s*$", "", report)
    report_path.write_text(report + "\n", encoding="utf-8")
    record: dict[str, object] = {"schema_version": 1, "report": report}
    if thinking is not None:
        record["thinking"] = thinking
    record.update(requires_human_review=True, generation=metrics)
    response_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if not report.strip() or not metrics["terminated_by_eos"]:
        raise ModelExecutionError(
            "NV-Reason-CT returned an empty or incomplete response. "
            "Partial output was retained; increase max_new_tokens for a truncated response."
        )
    return {"report": str(report_path), "response": str(response_path)}
