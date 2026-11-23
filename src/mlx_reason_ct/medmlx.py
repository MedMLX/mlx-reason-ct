"""NV-Reason-CT's MedMLX operation, its per-CT session and the declaration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)
from medmlx_core.files import PathLike
from medmlx_core.weights import model_folder

from mlx_reason_ct._payloads import JsonValue
from mlx_reason_ct.response import ReportFiles

if TYPE_CHECKING:
    from mlx_reason_ct.api import CTSession

MODEL: dict[str, str | float] = {
    "id": "mlx-reason-ct",
    "label": "NV-Reason-CT",
    "task": "report-generation",
    "run": "mlx_reason_ct.medmlx:run",
    "summary": "CT reporting and question answering with NV-Reason-CT on Metal; BF16 weights.",
    # Measured MLX peaks: 12.19 GB for fast-bf16 chest and abdominal reports,
    # 12.46 GB for source-bf16. The model resamples every scan to one grid, so
    # the peak does not grow with the input.
    "peak_memory_gb": 12.5,
}

type Region = Literal["chest", "abdomen"] | None
type Precision = Literal["fast-bf16", "source-bf16"]
type PrefillReuse = Literal["exact", "fast"]


def run(
    *,
    image_path: PathLike,
    output_dir: PathLike,
    weights_path: PathLike | None = None,
    prompt: str | None = None,
    anatomy_region: Region = "chest",
    enable_thinking: bool = False,
    max_new_tokens: int = 1024,
    precision: Precision = "fast-bf16",
    history: Sequence[tuple[str, str]] | None = None,
    tools: Sequence[Mapping[str, JsonValue]] | None = None,
    tool_history: Sequence[Mapping[str, str]] | None = None,
    response_choices: Sequence[str] | None = None,
    tool_choice: Literal["required"] | None = None,
    overwrite: bool = False,
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
    ``precision`` selects fast-bf16 (default; native BF16 matmuls with FP32
    accumulation for the CT and prompt, FP32 arithmetic and caches elsewhere) or
    source-bf16 (the pinned source's BF16 cast and cache boundaries, reproduced
    step by step; slower).
    ``history`` continues a conversation about the same CT: already-answered
    ``(question, answer)`` turns, oldest first; ``prompt`` is the new, still-
    unanswered turn. Empty or truncated responses raise ModelExecutionError
    after retaining partial output and failure metadata. All reports require
    human review.
    ``tools`` passes function schemas to the checkpoint's native chat template.
    ``tool_history`` is alternating assistant/tool text turns, ending with a
    tool response, after the opening ``prompt``; it is exclusive with ``history``.
    Tool requests are never executed by this provider. The checkpoint does not
    request a tool unprompted, so ask for one with ``tool_choice="required"``,
    which constrains the answer to one call of a declared tool, or list the
    permitted calls yourself as ``response_choices``. With ``tools``, the result's
    ``details["tool_calls"]`` holds the requested calls as names and arguments.
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
        precision: ``fast-bf16`` (default) with native BF16 matmuls and FP32
            elsewhere, or ``source-bf16`` with source BF16 boundaries.
        history: Already-answered user/assistant turns for this CT, oldest first.
        tools: Function schemas rendered by the checkpoint's chat template; tool
            requests are returned as text and are not executed by this provider.
        tool_history: Assistant/tool text turns after the opening prompt; must end
            with a tool response and cannot be combined with ``history``.
        response_choices: Explicit output strings enforced with LLGuidance. Requires
            the ``tools`` extra and thinking disabled.
            ``details["generation"]["choice_probabilities"]`` gives the model's
            probability of each when its first decision settles the choice, as it
            does for ``Yes``/``No``; ``choice_probability`` is that of the selected
            token path and is always present.
        tool_choice: ``"required"`` constrains the answer to one call of a declared
            tool. Each required parameter needs an ``enum`` (or the boolean type) and
            optional parameters are left out; for other values, such as IDs a tool
            returned, list the calls as ``response_choices`` instead. Same
            requirements as ``response_choices``, which it cannot be combined with.

    Returns:
        ``{"outputs": {"report", "response", "metadata"}, "artifacts": ..., "details": ...}``:
        absolute POSIX paths of ``report.txt``, ``model_response.json`` and ``run.json``,
        a ``report`` artifact for each (``not_applicable`` coordinate space), and the
        ``run.json`` record with the answer fields of ``model_response.json`` added:
        ``details["report"]``, the answer without reasoning or a trailing empty
        ``<answer></answer>``; ``details["thinking"]`` with thinking enabled;
        ``details["labels"]``, the finding labels a report or analysis ends with as
        ``<answer> label, label </answer>``, as a list; and, with ``tools``,
        ``details["tool_calls"]``, a list of ``{"name", "arguments"}`` checked
        against the declarations and empty for a prose answer.
        ``details["tool_call_error"]`` is set, and the list empty, when the answer
        holds tool-call markup that is malformed or breaks a declaration.

    Raises:
        InvalidInputError: ``image_path`` is not an existing ``.nii``/``.nii.gz``
            file, ``overwrite`` is not a bool, outputs exist without ``overwrite``,
            or the options or CT are invalid.
        AssetNotReadyError: ``weights_path`` is not a folder or the bundle is
            missing or mismatched.
        MissingDependencyError: an import fails or Metal is unavailable.
        ModelExecutionError: generation failed, or the response was empty or truncated.
    """
    from mlx_reason_ct.api import ct_file, validate_generation_options

    image = ct_file(image_path)
    # Check the options before the automatic download of the 17 GB bundle.
    validate_generation_options(
        prompt, anatomy_region, enable_thinking, max_new_tokens, overwrite, precision, history,
        tools=tools, tool_history=tool_history, response_choices=response_choices,
        tool_choice=tool_choice,
    )
    files = ReportFiles(Path(output_dir).expanduser().resolve())
    files.require_writable(overwrite=overwrite)
    weights = model_folder("mlx-reason-ct", weights_path)
    try:
        from mlx_reason_ct.api import report_result, write_report

        record = write_report(
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
            tool_choice=tool_choice,
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
    return report_result(files.directory, record)


def load_model(
    *,
    image_path: PathLike,
    weights_path: PathLike | None = None,
    anatomy_region: Region = "chest",
    precision: Precision = "fast-bf16",
    cache_chunk_size: int = 256,
    prefill_reuse: PrefillReuse = "exact",
) -> CTSession:
    """Load NV-Reason-CT for several reports or questions about one CT.

    The CT is encoded once and the visual weights are then released, which keeps
    the peak at the one-shot level, so a session serves the one CT it was loaded
    for. ``image_path``, ``weights_path``, ``anatomy_region`` and ``precision`` mean
    what they mean in :func:`run`; the returned session's
    ``run`` takes the remaining options. ``cache_chunk_size`` grows the
    KV cache in multiples of 256 tokens (0 concatenates).

    After its first call, a ``fast-bf16`` session decodes only the end of each
    prompt and reuses the state of the tokens before it. ``prefill_reuse="exact"``
    (default) decodes the last 2,561 to 4,608 CT tokens again with the question and
    reproduces the logits of a whole prefill; ``"fast"`` decodes a few dozen tokens
    with the question, and its logits differ at rounding level: suited to
    constrained choices and tool calls, while the wording of a long free-text
    answer can diverge. The ``source-bf16`` profile always prefills the whole prompt.
    Close the session, or use it as a context manager, to release the model.
    """
    from mlx_reason_ct.api import CTSession

    return CTSession(
        image_path=image_path, weights_path=weights_path, anatomy_region=anatomy_region,
        precision=precision, cache_chunk_size=cache_chunk_size, prefill_reuse=prefill_reuse,
    )
