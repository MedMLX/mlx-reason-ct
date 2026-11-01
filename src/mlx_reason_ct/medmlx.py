"""MedMLX declaration and offline CT report generation with explicit converted assets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import cast

from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)
from medmlx_core.weights import weights_dir, weights_revision

from mlx_reason_ct._payloads import JsonValue
from mlx_reason_ct.response import ReportFiles

MODEL: dict[str, str | float] = {
    "id": "mlx-reason-ct",
    "label": "NV-Reason-CT",
    "task": "generation",
    "run": "mlx_reason_ct.medmlx:run",
    "summary": "CT reporting and question answering with NV-Reason-CT on Metal; FP32 by default.",
    "peak_memory_gb": 22.7,
}


def run(
    *,
    image_path: str | PathLike[str],
    weights_path: str | PathLike[str] | None = None,
    output_dir: str | PathLike[str],
    overwrite: bool = False,
    prompt: str | None = None,
    anatomy_region: str | None = "chest",
    enable_thinking: bool = False,
    max_new_tokens: int = 512,
    precision: str = "float32",
    history: Sequence[tuple[str, str]] | None = None,
    tools: Sequence[Mapping[str, JsonValue]] | None = None,
    tool_history: Sequence[Mapping[str, str]] | None = None,
    response_choices: Sequence[str] | None = None,
) -> dict[str, object]:
    """Generate a report from one scalar 3D HU NIfTI CT on Apple Silicon Metal.

    Omit ``weights_path`` to download and reuse ``MedMLX/MLX-Reason-CT``.
    An explicit ``weights_path`` selects a local bundle folder or offline conversion. It contains
    ``config.json``, ``generation_config.json``, ``tokenizer.json``,
    ``chat_template.jinja``,
    ``image_processor_3d/preprocessor_config.json`` and ``mlx/`` with
    35 FP32 ``.safetensors`` shards. Conversion receipts are not required. Writes
    ``report.txt``, ``model_response.json`` and ``run.json`` in ``output_dir``.
    A custom ``prompt`` asks a CT question; otherwise a structured report for
    ``anatomy_region`` (chest or abdomen) is requested. Pass ``anatomy_region=None``
    for the pinned processor's own centered crop, used upstream when the
    chest/abdomen heuristic is unreliable for a given scan. Generation is greedy.
    ``precision`` selects float32 (default), bfloat16 (BF16 weights with the
    pinned source's cast and cache boundaries; not yet qualified against the
    original BF16 model), or bfloat16_fp32 (BF16 weights with FP32 accumulation
    and decoding).
    ``history`` continues a conversation about the same CT: already-answered
    ``(question, answer)`` turns, oldest first; ``prompt`` is the new, still-
    unanswered turn. Empty or truncated responses raise ModelExecutionError
    after retaining partial output and failure metadata. All reports require
    human review.
    ``tools`` passes function schemas to the checkpoint's native chat template.
    ``tool_history`` is alternating assistant/tool text turns, ending with a
    tool response, after the opening ``prompt``; it is exclusive with ``history``.
    Tool requests are returned as text and never executed by this provider.
    ``response_choices`` constrains decoding to one explicit string with
    LLGuidance; install the ``tools`` extra and leave thinking disabled.

    Args:
        image_path: Existing scalar 3D CT ``.nii`` or ``.nii.gz`` file in Hounsfield units.
        overwrite: Allow the report files in ``output_dir`` to be replaced.
        enable_thinking: Select the processor's thinking-enabled prompt template.
        max_new_tokens: Greedy generation limit, including EOS if produced; 1 to 8192.
        weights_path: Optional folder with tokenizer/configuration files and native
            ``mlx/`` shards; omit to download and reuse the published FP32 bundle.
        output_dir: Directory for the report, response and run record.
        prompt: New question or instruction; None requests a structured regional report.
        anatomy_region: ``chest`` (default), ``abdomen``, or None for the source
            processor's centered crop without the regional heuristic.
        precision: ``float32`` (default), ``bfloat16`` with source BF16 boundaries,
            or ``bfloat16_fp32`` with BF16 weights and FP32 accumulation/caches.
        history: Already-answered user/assistant turns for this CT, oldest first.
        tools: Function schemas rendered by the checkpoint's chat template; tool
            requests are returned as text and are not executed by this provider.
        tool_history: Assistant/tool text turns after the opening prompt; must end
            with a tool response and cannot be combined with ``history``.
        response_choices: Explicit output strings enforced with LLGuidance. Requires
            the ``tools`` extra and thinking disabled.

    Returns:
        ``{"outputs": {"report", "response", "run"}, "details": ...}``: absolute
        POSIX file paths and the ``run.json`` record.

    Raises:
        InvalidInputError: ``image_path`` is not an existing ``.nii``/``.nii.gz``
            file, ``overwrite`` is not a bool, outputs exist without ``overwrite``,
            or the options or CT are invalid.
        AssetNotReadyError: ``weights_path`` is not a directory or the bundle is
            missing or mismatched.
        MissingDependencyError: an import fails or Metal is unavailable.
        ModelExecutionError: generation failed, or the response was empty or truncated.
    """
    image = Path(image_path).expanduser().resolve()
    if not image.name.endswith((".nii", ".nii.gz")):
        raise InvalidInputError(f"image_path must be a NIfTI CT file (.nii or .nii.gz): {image}")
    if not image.is_file():
        raise InvalidInputError(f"image_path does not exist: {image}")
    if not isinstance(cast(object, overwrite), bool):
        raise InvalidInputError("overwrite must be a boolean")
    from mlx_reason_ct.api import validate_generation_options

    # Check the options before the automatic download of the 17 GB bundle.
    validate_generation_options(
        prompt, anatomy_region, enable_thinking, max_new_tokens, overwrite, precision, history
    )
    if response_choices is not None and enable_thinking:
        raise InvalidInputError("response_choices requires enable_thinking=False")
    files = ReportFiles(Path(output_dir).expanduser().resolve())
    files.require_writable(overwrite=overwrite)
    weights = (
        weights_dir("mlx-reason-ct") if weights_path is None else Path(weights_path).expanduser()
    )
    if not weights.is_dir():
        raise AssetNotReadyError(
            f"NV-Reason-CT converted weights directory not found: {weights}",
            hint="Pass the model folder containing tokenizer/configuration files "
            "and mlx/ FP32 shards.",
        )
    try:
        from mlx_reason_ct.api import generate_report

        details = generate_report(
            image,
            files.directory,
            model_dir=weights,
            prompt=prompt,
            anatomy_region=anatomy_region,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            precision=precision,
            overwrite=overwrite,
            history=history,
            tools=tools,
            tool_history=tool_history,
            response_choices=response_choices,
        )
    except (InvalidInputError, AssetNotReadyError, MissingDependencyError, ModelExecutionError):
        raise
    except ImportError as error:
        raise MissingDependencyError(
            f"NV-Reason-CT dependency is unavailable: {error}; install mlx-reason-ct.",
        ) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise ModelExecutionError(
            f"NV-Reason-CT generation failed: {error}; check output permissions and Metal memory.",
        ) from error
    details.setdefault("weights_revision", weights_revision(weights))
    return {"outputs": files.outputs(), "details": details}
