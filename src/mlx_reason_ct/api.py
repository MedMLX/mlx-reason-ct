"""Offline generation with pinned arithmetic profiles and durable completion status."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, cast

from medmlx_core.files import PathLike, input_file
from medmlx_core.outputs import artifact
from medmlx_core.weights import model_folder, weights_revision

from mlx_reason_ct._payloads import JsonValue
from mlx_reason_ct.constrained import validate_response_choices

if TYPE_CHECKING:
    from mlx_reason_ct._native_types import Array, GenerationModel, Mlx
    from mlx_reason_ct.constrained import ChoiceConstraint
    from mlx_reason_ct.ct_crop import CTCrop
    from mlx_reason_ct.mlx_model import NativeModel

from mlx_reason_ct.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError
from mlx_reason_ct.generation_state import PromptPrefix, cache_capacity
from mlx_reason_ct.mlx_weights import ENGINE, REVISION, require_runtime_assets
from mlx_reason_ct.processor_mlx import Processor, VolumePrompt, validate_tool_context
from mlx_reason_ct.response import (
    ReportFiles,
    restore_outputs_on_failure,
    save_response,
    write_private_text,
)
from mlx_reason_ct.runtime import host_info, import_mlx

#: Report requests without a prompt: the upstream model card's wording by region.
DEFAULT_PROMPTS = {
    "chest": "Write a structured chest CT report.",
    "abdomen": "Write a structured abdominal CT report.",
    None: "Write a structured CT report.",
}

PRECISION_PROFILES: dict[str, str] = {
    "bfloat16_fast": "bfloat16_matmul",
    "bfloat16": "source_bfloat16",
}


def _load_assets(model_dir: Path) -> Processor:
    try:
        require_runtime_assets(model_dir)
        return Processor(model_dir)
    except (InvalidInputError, OSError, ValueError, KeyError, TypeError) as error:
        raise AssetNotReadyError(
            f"NV-Reason-CT model folder is missing or unreadable at {model_dir}: {error}",
            reason=f"invalid model folder {model_dir}",
            hint="Retry the model download, or pass a folder containing the tokenizer, "
            "processor configuration and mlx/ FP32 shards.",
        ) from error


def _load_model(model_dir: Path, *, precision: str) -> NativeModel:
    from mlx_reason_ct.mlx_model import NativeModel

    try:
        return NativeModel(model_dir, precision=precision)
    except (InvalidInputError, OSError, ValueError, KeyError, RuntimeError) as error:
        raise AssetNotReadyError(
            f"Cannot load NV-Reason-CT FP32 tensors at {model_dir}: {error}",
            hint="Verify the converted bundle with mlx-reason-ct verify --model-dir PATH.",
        ) from error


def validate_generation_options(
    prompt: str | None,
    anatomy_region: str | None,
    enable_thinking: bool,
    max_new_tokens: int,
    overwrite: bool,
    precision: str,
    history: Sequence[tuple[str, str]] | None,
    *,
    tools: Sequence[Mapping[str, JsonValue]] | None = None,
    tool_history: Sequence[Mapping[str, str]] | None = None,
    response_choices: Sequence[str] | None = None,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Check the generation options and return the prompt and history to render.

    Raises:
        InvalidInputError: An option is out of range, of the wrong type or unknown.
    """
    if not isinstance(cast(object, precision), str) or precision not in PRECISION_PROFILES:
        raise InvalidInputError(
            "Unsupported NV-Reason-CT MLX precision profile; choose bfloat16_fast or bfloat16."
        )
    if anatomy_region is not None and (
        not isinstance(cast(object, anatomy_region), str)
        or anatomy_region not in {"chest", "abdomen"}
    ):
        raise InvalidInputError("anatomy_region must be chest, abdomen, or None")
    if prompt is None:
        prompt = DEFAULT_PROMPTS[anatomy_region]
    # Annotations do not bind callers, so validate the runtime values.
    token_limit = cast(object, max_new_tokens)
    if (
        isinstance(token_limit, bool)
        or not isinstance(token_limit, int)
        or not (1 <= token_limit <= 8192)
    ):
        raise InvalidInputError("max_new_tokens must be an integer from 1 to 8192")
    prompt_text = cast(object, prompt)
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise InvalidInputError("prompt must be nonempty text")
    if not isinstance(cast(object, enable_thinking), bool) or not isinstance(
        cast(object, overwrite), bool
    ):
        raise InvalidInputError("enable_thinking and overwrite must be booleans")
    resolved_history = _validate_history(history)
    declarations, _ = validate_tool_context(tools, tool_history)
    if declarations and resolved_history:
        raise InvalidInputError("tools and question/answer history are mutually exclusive")
    if response_choices is not None:
        if enable_thinking:
            raise InvalidInputError("response_choices requires enable_thinking=False")
        validate_response_choices(response_choices)
    return prompt, resolved_history


def _validate_history(
    history: Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    """Validate prior ``(question, answer)`` turns for a follow-up request.

    Upstream's own chat template supports replaying completed turns ahead of
    a new question (the model card documents "follow-up questions and
    clarification requests"); each entry here is one already-answered turn,
    oldest first, with the CT volume attached only once, on the first turn.
    """
    if history is None:
        return ()
    turns = cast(object, history)
    if isinstance(turns, str | bytes) or not isinstance(turns, Sequence):
        raise InvalidInputError("history must be a sequence of (question, answer) pairs")
    resolved: list[tuple[str, str]] = []
    for turn in cast(Sequence[object], turns):
        if isinstance(turn, str | bytes) or not isinstance(turn, Sequence):
            raise InvalidInputError("Each history turn must be a (question, answer) pair")
        values = cast(Sequence[object], turn)
        if len(values) != 2:
            raise InvalidInputError("Each history turn must be a (question, answer) pair")
        question, answer = values
        if (
            not isinstance(question, str)
            or not isinstance(answer, str)
            or not question.strip()
            or not answer.strip()
        ):
            raise InvalidInputError("Each history turn must hold nonempty question/answer text")
        resolved.append((question, answer))
    return tuple(resolved)


def generate(
    model: GenerationModel,
    inputs: VolumePrompt,
    embeddings: Array,
    *,
    max_new_tokens: int,
    prefix: PromptPrefix | None = None,
    cache_chunk_size: int = 0,
    constraint: ChoiceConstraint | None = None,
) -> tuple[list[int], dict[str, object]]:
    """Greedy generation with the pinned EOS set and request-owned hybrid cache.

    Args:
        model: loaded network.
        inputs: tokenized prompt; its image placeholders start at ``image_start``.
        embeddings: projected CT embeddings, shape ``(1, N, hidden)``, replacing
            the ``N`` placeholder token embeddings.
        max_new_tokens: generation limit, including the EOS token if produced.
        prefix: Optional complete prefill snapshot owned by one fixed model/CT
            session. Only exact token/position prefixes are reused; a mismatch
            rebuilds prefill. Updated before generation, so callers must discard
            it if generation or response persistence fails.
        cache_chunk_size: Zero retains concatenating KV storage; a positive
            multiple of 256 allocates capacity in chunks, without token eviction.
        constraint: Optional request-owned LLGuidance choice constraint. Raw
            model logits are unchanged; only greedy token selection is restricted.

    Returns:
        ``(token_ids, metrics)``. Generation stops at the first EOS id or at
        ``max_new_tokens``; ``metrics`` records whether it ended at EOS or was
        truncated, token counts, and decoder-cache sizes.

    Raises:
        ModelExecutionError: a logit tensor is not finite.
    """
    mx = import_mlx()
    tokens = mx.array(inputs.input_ids)
    start = inputs.image_start
    length = embeddings.shape[1]
    hidden = mx.concatenate(
        [model.embed(tokens[:, :start]), embeddings, model.embed(tokens[:, start + length :])],
        axis=1,
    )
    cache_capacity(0, cache_chunk_size)
    reused = 0 if prefix is None else prefix.matching_tokens(inputs)
    if reused and prefix is not None and prefix.cache is not None:
        cache = prefix.cache.clone()
        if reused < hidden.shape[1]:
            hidden, cache = model.decode(
                hidden[:, reused:], mx.array(inputs.position_ids[:, :, reused:]), cache
            )
            logit = model.logits(hidden[:, -1:])
        else:
            assert prefix.logits is not None
            logit = prefix.logits
    else:
        initial_cache = None
        if cache_chunk_size:
            from mlx_reason_ct.mlx_model import DecoderCache

            initial_cache = DecoderCache(kv_chunk_size=cache_chunk_size)
        hidden, cache = model.decode(hidden, mx.array(inputs.position_ids), initial_cache)
        logit = model.logits(hidden[:, -1:])
    # Only the last position is projected to logits; the prompt-length hidden states
    # are dropped afterwards.
    mx.eval(logit, *cache.arrays())
    if prefix is not None:
        prefix.inputs, prefix.cache, prefix.logits = inputs, cache.clone(), mx.array(logit)
    del hidden
    initial_bytes = cache.nbytes
    growth: list[dict[str, int]] = [{"decoded_tokens": 0, "cache_bytes": initial_bytes}]
    output: list[int] = []

    def launch(values: Array) -> tuple[Array, Array]:
        """Queue the finiteness check and greedy choice without waiting for them."""
        finite, choice = mx.all(mx.isfinite(values)), mx.argmax(values[0, -1])
        mx.async_eval(finite, choice, *cache.arrays())
        return finite, choice

    # Unconstrained greedy decoding keeps the chosen token on the GPU and queues
    # step n + 1 before reading token n, so graph construction overlaps the
    # previous step. A step queued after the final EOS is discarded; it never
    # changes emitted tokens. Constrained sampling needs host logits per step.
    pending = launch(logit) if constraint is None else None
    fed_bytes = initial_bytes
    for index in range(max_new_tokens):
        fed_bytes = cache.nbytes
        if pending is not None:
            finite, choice = pending
            if index + 1 < max_new_tokens:
                # Image tokens consume fewer MRoPE positions than sequence slots, so
                # new text positions are the cache length shifted by rope_delta.
                position = mx.full((3, 1, 1), cache.offset + inputs.rope_delta, dtype=mx.int64)
                hidden, cache = model.decode(
                    model.embed(mx.reshape(choice, (1, 1))), position, cache
                )
                pending = launch(model.logits(hidden))
            if not bool(finite.item()):
                raise ModelExecutionError("NV-Reason-CT produced nonfinite logits")
            token = int(cast(int, choice.item()))
        else:
            assert constraint is not None
            if not bool(mx.all(mx.isfinite(logit)).item()):
                raise ModelExecutionError("NV-Reason-CT produced nonfinite logits")
            token = constraint.sample(logit[0, -1], model.eos_ids)
        output.append(token)
        if token in model.eos_ids or index + 1 == max_new_tokens:
            break
        if pending is None:
            position = mx.full((3, 1, 1), cache.offset + inputs.rope_delta, dtype=mx.int64)
            hidden, cache = model.decode(model.embed(mx.array([[token]])), position, cache)
            logit = model.logits(hidden)
            mx.eval(logit, *cache.arrays())
        if len(output) in {1, 8, 32, 128, 256, 512, 1024, 2048, 4096}:
            growth.append({"decoded_tokens": len(output), "cache_bytes": cache.nbytes})
    mx.synchronize()
    # The last sampled token is never fed back, so the cache holds one fewer token.
    growth.append({"decoded_tokens": len(output) - 1, "cache_bytes": fed_bytes})
    complete = bool(output) and output[-1] in model.eos_ids
    return output, {
        "generated_tokens": len(output),
        "constrained_choice_count": 0 if constraint is None else len(constraint.choices),
        "reused_prefix_tokens": reused,
        "cache_chunk_size": cache_chunk_size,
        "terminated_by_eos": complete,
        "truncated_by_max_new_tokens": len(output) >= max_new_tokens and not complete,
        "prompt_tokens": int(tokens.shape[1]),
        "initial_cache_bytes": initial_bytes,
        "final_cache_bytes": cache.nbytes,
        "cache_growth": growth,
    }


def write_report(
    source: str | Path,
    output_dir: str | Path,
    *,
    model_dir: str | Path,
    prompt: str | None = None,
    anatomy_region: str | None = "chest",
    enable_thinking: bool = False,
    max_new_tokens: int = 512,
    precision: str = "bfloat16_fast",
    overwrite: bool = False,
    history: Sequence[tuple[str, str]] | None = None,
    tools: Sequence[Mapping[str, JsonValue]] | None = None,
    tool_history: Sequence[Mapping[str, str]] | None = None,
    response_choices: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run one HU NIfTI CT locally on Metal; retain partial output on truncation.

    Without a prompt, a structured report for ``anatomy_region`` is requested.
    ``anatomy_region`` is ``"chest"``, ``"abdomen"``, or ``None`` for the
    pinned processor's own centered crop, used upstream when its chest/abdomen
    heuristic is unreliable for a given geometry. ``history`` continues a
    conversation about the same CT: a sequence of already-answered
    ``(question, answer)`` turns, oldest first, with ``prompt``/the default
    report request as the new, still-unanswered turn. Existing report files
    require ``overwrite=True`` to replace.
    ``tools`` supplies function schemas to the pinned chat template;
    ``tool_history`` replays complete assistant/tool exchanges after ``prompt``.
    These are exclusive with Q/A ``history``. Returned tool-call text is inert;
    the caller owns validation, execution and result feedback.
    ``response_choices`` restricts greedy decoding to one supplied string using
    LLGuidance (the ``tools`` extra). It requires thinking disabled; the model
    still chooses between permitted alternatives using its own logits.

    ``source`` is a HU NIfTI CT and ``model_dir`` the converted bundle root.
    ``precision`` is a public profile name (see ``PRECISION_PROFILES``) and
    ``max_new_tokens`` must be an integer in 1-8192. Generation is greedy.

    Returns:
        The ``run.json`` record: status, generation metrics, CT geometry,
        generated token ids and runtime details. The report and response are
        written to ``output_dir`` next to it. The record always sets
        ``requires_human_review`` and ``clinical_validation: false``.

    Raises:
        InvalidInputError: invalid options, unreadable CT, or existing outputs
            without ``overwrite``.
        AssetNotReadyError: the converted bundle is missing or mismatched.
        MissingDependencyError: Metal or MLX is unavailable.
        ModelExecutionError: non-finite activations, or an empty or truncated
            response (partial files and a ``status: failed`` run record are kept).
    """
    prompt, resolved_history = validate_generation_options(
        prompt,
        anatomy_region,
        enable_thinking,
        max_new_tokens,
        overwrite,
        precision,
        history,
        tools=tools,
        tool_history=tool_history,
        response_choices=response_choices,
    )
    model_dir = Path(model_dir).expanduser()
    output_dir = Path(output_dir).expanduser().resolve()
    files = ReportFiles(output_dir)
    files.require_writable(overwrite=overwrite)
    processor = _load_assets(model_dir)
    checkpoint_revision = weights_revision(model_dir)
    constraint = None
    if response_choices is not None:
        from mlx_reason_ct.constrained import ChoiceConstraint

        constraint = ChoiceConstraint(model_dir / "tokenizer.json", response_choices)
    pixels, crop = processor.image(Path(source), anatomy_region)
    inputs = processor.prompt(
        prompt,
        enable_thinking,
        resolved_history,
        tools=tools,
        tool_history=tool_history,
    )
    mx = import_mlx()
    started = perf_counter()
    mx.reset_peak_memory()
    native_profile = PRECISION_PROFILES[precision]
    model = _load_model(model_dir, precision=native_profile)
    features, embeddings = model.vision(mx.array(pixels))
    if not bool(mx.all(mx.isfinite(embeddings)).item()):
        raise ModelExecutionError("NV-Reason-CT produced nonfinite image embeddings")
    # Only the projected embeddings are needed for decoding.
    del features
    model.release_vision()
    tokens, metrics = generate(
        model,
        inputs,
        embeddings,
        max_new_tokens=max_new_tokens,
        # Preallocated 256-token KV chunks: concatenating at a ~14k-token CT
        # context allocates new K/V for every decoded token.
        cache_chunk_size=256,
        constraint=constraint,
    )
    return _finish_report(
        source,
        output_dir,
        files,
        processor,
        model,
        crop,
        tokens,
        metrics,
        mx=mx,
        started=started,
        precision=precision,
        anatomy_region=anatomy_region,
        prior_turns=len(resolved_history) + len(tool_history or ()) // 2,
        overwrite=overwrite,
        checkpoint_revision=checkpoint_revision,
        literal_response=constraint is not None,
    )


def _finish_report(
    source: str | Path,
    output_dir: Path,
    files: ReportFiles,
    processor: Processor,
    model: NativeModel,
    crop: CTCrop,
    tokens: list[int],
    metrics: dict[str, object],
    *,
    mx: Mlx,
    started: float,
    precision: str,
    anatomy_region: str | None,
    prior_turns: int,
    overwrite: bool,
    checkpoint_revision: str | None,
    literal_response: bool = False,
) -> dict[str, object]:
    """Persist the same completion and geometry contract for one-shot and session calls."""
    native_profile = PRECISION_PROFILES[precision]
    metrics.update(
        {
            "elapsed_s": round(perf_counter() - started, 4),
            "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
            "precision": precision,
            "learned_weight_dtype": str(model.dtype),
            "decoder_arithmetic": {"bfloat16": "bfloat16", "bfloat16_fast": "bfloat16_matmul"}[
                precision
            ],
            "native_arithmetic_profile": native_profile,
            "synchronization": "prefill eval per layer; decode async_eval per token",
            "anatomy_region": anatomy_region,
            "prior_turns": prior_turns,
        }
    )
    files.require_writable(overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": 1,
        "schema": "nv_reason_ct_mlx.run.v1",
        "status": "succeeded",
        "source": str(Path(source)),
        "weights_revision": checkpoint_revision,
        "model_revision": checkpoint_revision,
        "reference_revision": REVISION,
        "bundle_engine": ENGINE,
        "generation": metrics,
        "ct_geometry": crop.geometry(),
        "generated_token_ids": tokens,
        "runtime": host_info(),
        "requires_human_review": True,
        "clinical_validation": False,
    }
    outputs = files.outputs()
    report_path = Path(outputs["report"])
    response_path = Path(outputs["response"])
    run_path = Path(outputs["metadata"])
    incomplete: ModelExecutionError | None = None
    with restore_outputs_on_failure((report_path, response_path, run_path)) as published:
        try:
            save_response(
                output_dir,
                processor.decode(tokens),
                metrics,
                overwrite=overwrite,
                literal=literal_response,
            )
        except ModelExecutionError as error:
            # An incomplete generation intentionally retains all three failed-run files.
            result.update(status="failed", error=str(error))
            incomplete = error
        published.extend((report_path, response_path))
        write_private_text(run_path, json.dumps(result, indent=2) + "\n", overwrite=overwrite)
        published.append(run_path)
    if incomplete is not None:
        raise incomplete
    return result


def ct_file(image_path: PathLike) -> Path:
    """Return the absolute CT path, refusing anything but an existing NIfTI file."""
    image = input_file("image_path", image_path).resolve()
    if not image.name.endswith((".nii", ".nii.gz")):
        raise InvalidInputError(f"image_path must be a NIfTI CT file (.nii or .nii.gz): {image}")
    return image


def report_result(output_dir: Path, record: dict[str, object]) -> dict[str, object]:
    """Describe one written report as a MedMLX result.

    ``details`` is the ``run.json`` record plus the text of ``report.txt`` as
    ``details["report"]``: the answer without reasoning or an empty trailing
    ``<answer></answer>``.
    """
    outputs = ReportFiles(output_dir).outputs()
    details = dict(record)
    details["report"] = Path(outputs["report"]).read_text(encoding="utf-8").removesuffix("\n")
    entry = artifact(kind="report", coordinate_space="not_applicable")
    return {"outputs": outputs, "artifacts": dict.fromkeys(outputs, entry), "details": details}


@dataclass
class _PreparedCT:
    processor: Processor
    model: NativeModel
    crop: CTCrop
    embeddings: Array
    weights_revision: str | None


class CTSession:
    """One CT encoded once, with the model kept loaded, for several reports or questions.

    The model loads and encodes the CT on the first call; its visual weights are
    then released, so a session serves one CT. :meth:`generate_report` takes the
    one-shot ``generate_report`` options that do not change the encoding; pass
    earlier turns explicitly as ``history`` or ``tool_history``. Calls are
    serialized per session. A changed CT file is rejected. Closing releases the
    model and embeddings without changing process-wide MLX caches.

    ``cache_chunk_size`` grows KV storage in multiples of 256 tokens; zero selects
    the original concatenating cache. ``reuse_prefix`` additionally snapshots a
    complete prefill and reuses it only when token IDs, image markers and MRoPE
    positions match exactly. It is opt-in because partitioned prefill must pass
    the selected arithmetic profile's numerical comparison on the target GPU.
    """

    def __init__(
        self,
        *,
        image_path: PathLike,
        weights_path: PathLike | None = None,
        anatomy_region: str | None = "chest",
        precision: str = "bfloat16_fast",
        cache_chunk_size: int = 256,
        reuse_prefix: bool = False,
    ) -> None:
        """Check the CT and options, then resolve (downloading if omitted) the weights."""
        self._source = ct_file(image_path)
        validate_generation_options(None, anatomy_region, False, 512, False, precision, None)
        cache_capacity(0, cache_chunk_size)
        if type(reuse_prefix) is not bool:
            raise InvalidInputError("reuse_prefix must be boolean")
        self._model_dir = model_folder("mlx-reason-ct", weights_path)
        self._anatomy_region = anatomy_region
        self._precision = precision
        self._chunk_size = cache_chunk_size
        self._reuse_prefix = reuse_prefix
        self._prepared: _PreparedCT | None = None
        self._stamp: tuple[int, int, int, int, int] | None = None
        self._prefix = PromptPrefix()
        self._closed = False
        self._lock = Lock()

    def _source_stamp(self) -> tuple[int, int, int, int, int]:
        """Return source identity and filesystem change metadata for cache admission."""
        stat = self._source.stat()
        # Writers can preserve size and restore mtime; ctime still changes on
        # the target filesystem when the source data is rewritten.
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def _prepare(self) -> _PreparedCT:
        stamp = self._source_stamp()
        if self._stamp is not None and stamp != self._stamp:
            raise InvalidInputError("CT file changed; load the model for it again")
        if self._prepared is None:
            processor = _load_assets(self._model_dir)
            pixels, crop = processor.image(self._source, self._anatomy_region)
            mx = import_mlx()
            model = _load_model(self._model_dir, precision=PRECISION_PROFILES[self._precision])
            features, embeddings = model.vision(mx.array(pixels))
            mx.eval(embeddings)
            if not bool(mx.all(mx.isfinite(embeddings)).item()):
                raise ModelExecutionError("NV-Reason-CT produced nonfinite image embeddings")
            del features
            model.release_vision()
            if stamp != self._source_stamp():
                raise InvalidInputError("CT file changed during encoding; load the model again")
            self._prepared = _PreparedCT(
                processor,
                model,
                crop,
                embeddings,
                weights_revision(self._model_dir),
            )
            self._stamp = stamp
        return self._prepared

    def generate_report(
        self,
        *,
        output_dir: PathLike,
        prompt: str | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 512,
        history: Sequence[tuple[str, str]] | None = None,
        tools: Sequence[Mapping[str, JsonValue]] | None = None,
        tool_history: Sequence[Mapping[str, str]] | None = None,
        response_choices: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> dict[str, object]:
        """Write one report or answer for the session's CT; see ``generate_report``.

        Raises:
            RuntimeError: The session is closed.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("NV-Reason-CT session is closed")
            question, resolved_history = validate_generation_options(
                prompt,
                self._anatomy_region,
                enable_thinking,
                max_new_tokens,
                overwrite,
                self._precision,
                history,
                tools=tools,
                tool_history=tool_history,
                response_choices=response_choices,
            )
            directory = Path(output_dir).expanduser().resolve()
            files = ReportFiles(directory)
            files.require_writable(overwrite=overwrite)
            started = perf_counter()
            mx = import_mlx()
            mx.reset_peak_memory()
            prepared = self._prepare()
            constraint = None
            if response_choices is not None:
                from mlx_reason_ct.constrained import ChoiceConstraint

                constraint = ChoiceConstraint(self._model_dir / "tokenizer.json", response_choices)
            inputs = prepared.processor.prompt(
                question, enable_thinking, resolved_history, tools=tools, tool_history=tool_history
            )
            try:
                tokens, metrics = generate(
                    prepared.model,
                    inputs,
                    prepared.embeddings,
                    max_new_tokens=max_new_tokens,
                    prefix=self._prefix if self._reuse_prefix else None,
                    cache_chunk_size=self._chunk_size,
                    constraint=constraint,
                )
                if self._stamp != self._source_stamp():
                    raise InvalidInputError(
                        "CT file changed during generation; load the model for it again"
                    )
                record = _finish_report(
                    self._source,
                    directory,
                    files,
                    prepared.processor,
                    prepared.model,
                    prepared.crop,
                    tokens,
                    metrics,
                    mx=mx,
                    started=started,
                    precision=self._precision,
                    anatomy_region=self._anatomy_region,
                    prior_turns=len(resolved_history) + len(tool_history or ()) // 2,
                    overwrite=overwrite,
                    checkpoint_revision=prepared.weights_revision,
                    literal_response=constraint is not None,
                )
            except BaseException:
                self._prefix = PromptPrefix()
                raise
            return report_result(directory, record)

    def close(self) -> None:
        """Release the model and CT embeddings; later calls raise ``RuntimeError``."""
        with self._lock:
            self._prepared = None
            self._prefix = PromptPrefix()
            self._closed = True

    def __enter__(self) -> CTSession:
        """Return this session for a ``with`` block that closes it on exit."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the session on any exit path, including exceptions."""
        self.close()
