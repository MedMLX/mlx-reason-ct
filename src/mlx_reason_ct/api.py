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
    from mlx_reason_ct.mlx_model import DecoderCache, NativeModel

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
from mlx_reason_ct.tool_calls import ToolCallError, tool_call_choices

#: Report requests without a prompt: the upstream model card's wording by region.
DEFAULT_PROMPTS = {
    "chest": "Write a structured chest CT report.",
    "abdomen": "Write a structured abdominal CT report.",
    None: "Write a structured CT report.",
}

PRECISION_PROFILES: dict[str, str] = {
    "fast-bf16": "bfloat16_matmul",
    "source-bf16": "source_bfloat16",
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
    tool_choice: str | None = None,
) -> tuple[str, tuple[tuple[str, str], ...], tuple[str, ...] | None]:
    """Check the generation options.

    Returns:
        The prompt and history to render, and the literal responses decoding is
        restricted to: ``response_choices``, the calls ``tool_choice="required"``
        allows, or ``None`` for free generation.

    Raises:
        InvalidInputError: An option is out of range, of the wrong type or unknown.
    """
    if not isinstance(cast(object, precision), str) or precision not in PRECISION_PROFILES:
        raise InvalidInputError(
            "Unsupported NV-Reason-CT MLX precision profile; choose fast-bf16 or source-bf16."
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
    if tool_choice is not None:
        if tool_choice != "required":
            raise InvalidInputError('tool_choice must be "required" or None')
        if response_choices is not None or not declarations:
            raise InvalidInputError("tool_choice requires tools and no response_choices")
        try:
            response_choices = tool_call_choices(declarations)
        except ToolCallError as error:
            raise InvalidInputError(
                f"tool_choice cannot enumerate the calls: {error}; pass them as response_choices"
            ) from error
    if response_choices is None:
        return prompt, resolved_history, None
    if enable_thinking:
        raise InvalidInputError("response_choices and tool_choice require enable_thinking=False")
    return prompt, resolved_history, validate_response_choices(response_choices)


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
        prefix: Optional prefill snapshot owned by one fixed model/CT session.
            The prompt is then decoded in two calls split at the snapshot's
            boundary, and the first call's state is kept for later prompts that
            start with the same tokens and positions; a mismatch rebuilds it. See
            ``PromptPrefix`` for when the split repeats the one-call arithmetic,
            which was established for the ``fast-bf16`` profile only.
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
    total = inputs.input_ids.shape[1]

    def embedded(begin: int, end: int) -> Array:
        """Input embeddings of prompt tokens ``begin:end`` with the CT spliced in."""
        parts: list[Array] = []
        if begin < start:
            parts.append(model.embed(tokens[:, begin : min(end, start)]))
        first, last = max(begin, start), min(end, start + length)
        if first < last:
            parts.append(embeddings[:, first - start : last - start])
        if end > start + length:
            parts.append(model.embed(tokens[:, max(begin, start + length) : end]))
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)

    cache_capacity(0, cache_chunk_size)
    initial: DecoderCache | None = None
    if cache_chunk_size:
        from mlx_reason_ct import mlx_model

        initial = mlx_model.DecoderCache(kv_chunk_size=cache_chunk_size)
    # A prompt too short to reach the first boundary has nothing to keep.
    split = 0 if prefix is None else prefix.boundary(inputs, length)
    reused = 0
    if prefix is not None and split:
        if prefix.matches(inputs, split):
            assert prefix.cache is not None
            initial, reused = prefix.cache.clone(), split
        else:
            prefix.clear()
            _, initial = model.decode(
                embedded(0, split), mx.array(inputs.position_ids[:, :, :split]), initial
            )
            mx.eval(*initial.arrays())
            # Stored only once evaluated, so an interrupted prefill leaves no snapshot.
            prefix.store(inputs, split, initial.clone())
    hidden, cache = model.decode(
        embedded(split, total), mx.array(inputs.position_ids[:, :, split:]), initial
    )
    # Only the last position is projected to logits; the prompt-length hidden states
    # are dropped afterwards.
    logit = model.logits(hidden[:, -1:])
    mx.eval(logit, *cache.arrays())
    del hidden
    initial_bytes = cache.nbytes
    growth: list[dict[str, int]] = [{"decoded_tokens": 0, "cache_bytes": initial_bytes}]
    output: list[int] = []

    def feed(token: Array) -> Array:
        """Decode one generated token, shaped ``(1, 1)``; return its hidden state."""
        nonlocal cache
        # Image tokens consume fewer MRoPE positions than sequence slots, so new
        # text positions are the cache length shifted by rope_delta.
        position = mx.full((3, 1, 1), cache.offset + inputs.rope_delta, dtype=mx.int64)
        hidden, cache = model.decode(model.embed(token), position, cache)
        return hidden

    def launch(values: Array) -> tuple[Array, Array]:
        """Queue the finiteness check and greedy choice without waiting for them."""
        finite, choice = mx.all(mx.isfinite(values)), mx.argmax(values[0, -1])
        mx.async_eval(finite, choice, *cache.arrays())
        return finite, choice

    metrics: dict[str, object] = {}
    if constraint is None:
        # Unconstrained greedy decoding keeps the chosen token on the GPU and queues
        # step n + 1 before reading token n, so graph construction overlaps the
        # previous step. A step queued after the final EOS is discarded; it never
        # changes emitted tokens.
        pending = launch(logit)
        fed_bytes = initial_bytes
        for index in range(max_new_tokens):
            fed_bytes = cache.nbytes
            finite, choice = pending
            if index + 1 < max_new_tokens:
                pending = launch(model.logits(feed(mx.reshape(choice, (1, 1)))))
            if not bool(finite.item()):
                raise ModelExecutionError("NV-Reason-CT produced nonfinite logits")
            output.append(int(cast(int, choice.item())))
            if output[-1] in model.eos_ids or index + 1 == max_new_tokens:
                break
            if len(output) in {1, 8, 32, 128, 256, 512, 1024, 2048, 4096}:
                growth.append({"decoded_tokens": len(output), "cache_bytes": cache.nbytes})
        mx.synchronize()
        # The last sampled token is never fed back, so the cache holds one fewer token.
        growth.append({"decoded_tokens": len(output) - 1, "cache_bytes": fed_bytes})
    else:
        # The model is consulted only where the permitted choices diverge. Tokens
        # they share are emitted from the grammar and decoded, one position at a
        # time as in free generation, only when a later decision needs their state;
        # the forced end of a choice is never decoded.
        vocabulary = logit.shape[-1]
        unfed: list[int] = []
        decoded = 0

        def ended() -> bool:
            return len(output) >= max_new_tokens or (bool(output) and output[-1] in model.eos_ids)

        while not ended():
            forced = constraint.forced(vocabulary, model.eos_ids)
            output += forced
            unfed += forced
            if ended():
                break
            state: Array | None = None
            for token in unfed:
                state = feed(mx.array([[token]]))
                mx.async_eval(state, *cache.arrays())
            if state is not None:
                logit = model.logits(state)
            decoded += len(unfed)
            unfed = []
            if not bool(mx.all(mx.isfinite(logit)).item()):
                raise ModelExecutionError("NV-Reason-CT produced nonfinite logits")
            output.append(constraint.sample(logit[0, -1], model.eos_ids))
            unfed.append(output[-1])
        del output[max_new_tokens:]
        growth.append({"decoded_tokens": decoded, "cache_bytes": cache.nbytes})
        metrics["choice_probability"] = constraint.probability
        if constraint.probabilities is not None:
            metrics["choice_probabilities"] = constraint.probabilities
    complete = bool(output) and output[-1] in model.eos_ids
    return output, metrics | {
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
    max_new_tokens: int = 1024,
    precision: str = "fast-bf16",
    overwrite: bool = False,
    history: Sequence[tuple[str, str]] | None = None,
    tools: Sequence[Mapping[str, JsonValue]] | None = None,
    tool_history: Sequence[Mapping[str, str]] | None = None,
    response_choices: Sequence[str] | None = None,
    tool_choice: str | None = None,
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
    ``tool_choice="required"`` restricts it the same way to one call of a declared
    tool, with arguments drawn from the declared enums.

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
    prompt, resolved_history, choices = validate_generation_options(
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
        tool_choice=tool_choice,
    )
    model_dir = Path(model_dir).expanduser()
    output_dir = Path(output_dir).expanduser().resolve()
    files = ReportFiles(output_dir)
    files.require_writable(overwrite=overwrite)
    processor = _load_assets(model_dir)
    checkpoint_revision = weights_revision(model_dir)
    constraint = None
    if choices is not None:
        from mlx_reason_ct.constrained import ChoiceConstraint

        constraint = ChoiceConstraint(model_dir / "tokenizer.json", choices)
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
        tools=tools or (),
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
    tools: Sequence[Mapping[str, JsonValue]] = (),
) -> dict[str, object]:
    """Persist the same completion and geometry contract for one-shot and session calls."""
    native_profile = PRECISION_PROFILES[precision]
    metrics.update(
        {
            "elapsed_s": round(perf_counter() - started, 4),
            "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
            "precision": precision,
            "learned_weight_dtype": str(model.dtype),
            "decoder_arithmetic": {"source-bf16": "bfloat16", "fast-bf16": "bfloat16_matmul"}[
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
                tools=tools,
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

    ``details`` is the ``run.json`` record plus the answer fields of
    ``model_response.json``: ``report`` (the answer without reasoning or an empty
    trailing ``<answer></answer>``), ``thinking`` when thinking was enabled,
    ``labels`` when the answer ends with a finding-label list, and ``tool_calls``
    (with ``tool_call_error`` when the markup was unusable) when tools were declared.
    """
    outputs = ReportFiles(output_dir).outputs()
    response = cast(
        dict[str, object], json.loads(Path(outputs["response"]).read_text(encoding="utf-8"))
    )
    details = dict(record)
    for name in ("report", "thinking", "labels", "tool_calls", "tool_call_error"):
        if name in response:
            details[name] = response[name]
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
    then released, so a session serves one CT. :meth:`run` takes the
    one-shot ``run`` options that do not change the encoding; pass
    earlier turns explicitly as ``history`` or ``tool_history``. Calls are
    serialized per session. A changed CT file is rejected. Closing releases the
    model and embeddings without changing process-wide MLX caches.

    ``cache_chunk_size`` grows KV storage in multiples of 256 tokens; zero selects
    the original concatenating cache.

    With the ``fast-bf16`` profile the session keeps the decoder state of the
    prompt tokens that every question shares (a tool block and most of the CT),
    0.7 to 1.0 GB, so each call after the first decodes only the rest of its prompt.
    ``prefill_reuse="exact"`` keeps the state at a 2,048-token boundary and decodes
    the last 2,561 to 4,608 CT tokens again with the question, which reproduces
    the logits of a whole prefill. ``"fast"`` keeps it at the last 64-token boundary
    and decodes only a few dozen tokens with the question; those few rows take
    other Metal matmul kernels, so logits differ at rounding level. That changed no
    constrained choice in the measured CTs, but the wording of a long free-text
    answer can diverge, including the session's first. See ``PromptPrefix``.
    Changing ``tools`` between calls changes the shared tokens and rebuilds the
    state. The ``source-bf16`` profile always prefills the whole prompt.
    """

    def __init__(
        self,
        *,
        image_path: PathLike,
        weights_path: PathLike | None = None,
        anatomy_region: str | None = "chest",
        precision: str = "fast-bf16",
        cache_chunk_size: int = 256,
        prefill_reuse: str = "exact",
    ) -> None:
        """Check the CT and options, then resolve (downloading if omitted) the weights."""
        self._source = ct_file(image_path)
        validate_generation_options(None, anatomy_region, False, 512, False, precision, None)
        cache_capacity(0, cache_chunk_size)
        if prefill_reuse not in ("exact", "fast"):
            raise InvalidInputError('prefill_reuse must be "exact" or "fast"')
        if prefill_reuse == "fast" and precision != "fast-bf16":
            raise InvalidInputError('prefill_reuse="fast" requires precision="fast-bf16"')
        self._model_dir = model_folder("mlx-reason-ct", weights_path)
        self._anatomy_region = anatomy_region
        self._precision = precision
        self._chunk_size = cache_chunk_size
        self._prepared: _PreparedCT | None = None
        self._stamp: tuple[int, int, int, int, int] | None = None
        # The source profile's emulated kernels partition by the whole prompt's row
        # count, so a split prefill was only established for fast-bf16.
        self._prefix = (
            PromptPrefix(fast=prefill_reuse == "fast") if precision == "fast-bf16" else None
        )
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

    def run(
        self,
        *,
        output_dir: PathLike,
        prompt: str | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 1024,
        history: Sequence[tuple[str, str]] | None = None,
        tools: Sequence[Mapping[str, JsonValue]] | None = None,
        tool_history: Sequence[Mapping[str, str]] | None = None,
        response_choices: Sequence[str] | None = None,
        tool_choice: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, object]:
        """Write one report or answer for the session's CT; see ``run``.

        Raises:
            RuntimeError: The session is closed.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("NV-Reason-CT session is closed")
            question, resolved_history, choices = validate_generation_options(
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
                tool_choice=tool_choice,
            )
            directory = Path(output_dir).expanduser().resolve()
            files = ReportFiles(directory)
            files.require_writable(overwrite=overwrite)
            started = perf_counter()
            mx = import_mlx()
            mx.reset_peak_memory()
            prepared = self._prepare()
            constraint = None
            if choices is not None:
                from mlx_reason_ct.constrained import ChoiceConstraint

                constraint = ChoiceConstraint(self._model_dir / "tokenizer.json", choices)
            inputs = prepared.processor.prompt(
                question, enable_thinking, resolved_history, tools=tools, tool_history=tool_history
            )
            tokens, metrics = generate(
                prepared.model,
                inputs,
                prepared.embeddings,
                max_new_tokens=max_new_tokens,
                prefix=self._prefix,
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
                tools=tools or (),
            )
            return report_result(directory, record)

    def close(self) -> None:
        """Release the model and CT embeddings; later calls raise ``RuntimeError``."""
        with self._lock:
            self._prepared = None
            self._prefix = None
            self._closed = True

    def __enter__(self) -> CTSession:
        """Return this session for a ``with`` block that closes it on exit."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the session on any exit path, including exceptions."""
        self.close()
