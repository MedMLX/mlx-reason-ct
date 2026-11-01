"""Offline generation with pinned arithmetic profiles and durable completion status."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, cast

from medmlx_core.weights import weights_revision

from mlx_reason_ct._payloads import JsonValue

if TYPE_CHECKING:
    from mlx_reason_ct._native_types import Array, GenerationModel, Mlx
    from mlx_reason_ct.constrained import ChoiceConstraint
    from mlx_reason_ct.ct_crop import CTCrop
    from mlx_reason_ct.mlx_model import NativeModel

from mlx_reason_ct.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError
from mlx_reason_ct.generation_state import PromptPrefix, cache_capacity
from mlx_reason_ct.mlx_weights import ENGINE, REVISION, require_runtime_assets
from mlx_reason_ct.processor_mlx import Processor, VolumePrompt
from mlx_reason_ct.response import ReportFiles, save_response
from mlx_reason_ct.runtime import host_info, import_mlx

PRECISION_PROFILES: dict[str, str] = {
    "float32": "float32",
    "bfloat16": "source_bfloat16",
    "bfloat16_fp32": "bfloat16",
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
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Check the generation options and return the prompt and history to render.

    Raises:
        InvalidInputError: An option is out of range, of the wrong type or unknown.
    """
    if not isinstance(cast(object, precision), str) or precision not in PRECISION_PROFILES:
        raise InvalidInputError(
            "Unsupported NV-Reason-CT MLX precision profile; "
            "choose float32, bfloat16 or bfloat16_fp32."
        )
    if anatomy_region is not None and anatomy_region not in {"chest", "abdomen"}:
        raise InvalidInputError("anatomy_region must be chest, abdomen, or None")
    if prompt is None:
        prompt = (
            f"write a structured {anatomy_region} CT report"
            if anatomy_region is not None
            else "write a structured CT report"
        )
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
    for index in range(max_new_tokens):
        if not bool(mx.all(mx.isfinite(logit)).item()):
            raise ModelExecutionError("NV-Reason-CT produced nonfinite logits")
        token = (
            int(cast(int, mx.argmax(logit[0, -1]).item()))
            if constraint is None
            else constraint.sample(logit[0, -1], model.eos_ids)
        )
        output.append(token)
        if token in model.eos_ids or index + 1 == max_new_tokens:
            break
        # Image tokens consume fewer MRoPE positions than sequence slots, so new text
        # positions are the cache length shifted by rope_delta.
        position = mx.full((3, 1, 1), cache.offset + inputs.rope_delta, dtype=mx.int64)
        hidden, cache = model.decode(model.embed(mx.array([[token]])), position, cache)
        logit = model.logits(hidden)
        mx.eval(logit, *cache.arrays())
        if len(output) in {1, 8, 32, 128, 256, 512, 1024, 2048, 4096}:
            growth.append({"decoded_tokens": len(output), "cache_bytes": cache.nbytes})
    mx.synchronize()
    # The last sampled token is never fed back, so the cache holds one fewer token.
    growth.append({"decoded_tokens": len(output) - 1, "cache_bytes": cache.nbytes})
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


def generate_report(
    source: str | Path,
    output_dir: str | Path,
    *,
    model_dir: str | Path,
    prompt: str | None = None,
    anatomy_region: str | None = "chest",
    enable_thinking: bool = False,
    max_new_tokens: int = 512,
    precision: str = "float32",
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
        prompt, anatomy_region, enable_thinking, max_new_tokens, overwrite, precision, history
    )
    model_dir = Path(model_dir).expanduser()
    output_dir = Path(output_dir).expanduser().resolve()
    files = ReportFiles(output_dir)
    files.require_writable(overwrite=overwrite)
    processor = _load_assets(model_dir)
    checkpoint_revision = weights_revision(model_dir)
    constraint = None
    if response_choices is not None:
        if enable_thinking:
            raise InvalidInputError("response_choices requires enable_thinking=False")
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
    tokens, metrics = generate(
        model,
        inputs,
        embeddings,
        max_new_tokens=max_new_tokens,
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
) -> dict[str, object]:
    """Persist the same completion and geometry contract for one-shot and session calls."""
    native_profile = PRECISION_PROFILES[precision]
    metrics.update(
        {
            "elapsed_s": round(perf_counter() - started, 4),
            "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
            "precision": precision,
            "learned_weight_dtype": str(model.dtype),
            "decoder_arithmetic": "bfloat16" if precision == "bfloat16" else "float32",
            "native_arithmetic_profile": native_profile,
            "synchronization": "mx.eval per layer/token and mx.synchronize",
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
    # Completion failure is recorded after retaining report and response files.
    try:
        save_response(output_dir, processor.decode(tokens), metrics)
    except ModelExecutionError as error:
        result.update(status="failed", error=str(error))
        Path(files.outputs()["run"]).write_text(json.dumps(result, indent=2) + "\n")
        raise
    Path(files.outputs()["run"]).write_text(json.dumps(result, indent=2) + "\n")
    return result


@dataclass
class _PreparedCT:
    processor: Processor
    model: NativeModel
    crop: CTCrop
    embeddings: Array
    weights_revision: str | None


class CTSession:
    """Reuse one loaded model and one CT's projected embeddings across questions.

    Loading is lazy on the first report. Completed responses enter history as raw
    decoded text, without report-specific thinking separation or empty-tag cleanup;
    failed/truncated responses do not. Calls are serialized per session. Closing
    releases owned arrays and history without changing process-wide MLX caches.
    A changed CT file is rejected: create a new session to encode a new image.

    ``cache_chunk_size`` grows KV storage in multiples of 256 tokens; zero selects
    the original concatenating cache. ``reuse_prefix`` additionally snapshots a
    complete prefill and reuses it only when token IDs, image markers and MRoPE
    positions match exactly. It is opt-in because partitioned prefill must pass
    the selected arithmetic profile's numerical comparison on the target GPU.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        model_dir: str | Path,
        anatomy_region: str | None = "chest",
        precision: str = "float32",
        cache_chunk_size: int = 256,
        reuse_prefix: bool = False,
    ) -> None:
        """Validate options and remember the request; assets load on the first report."""
        validate_generation_options(None, anatomy_region, False, 512, False, precision, None)
        cache_capacity(0, cache_chunk_size)
        if type(reuse_prefix) is not bool:
            raise InvalidInputError("reuse_prefix must be boolean")
        self._source = Path(source).expanduser().resolve()
        self._model_dir = Path(model_dir).expanduser()
        self._anatomy_region = anatomy_region
        self._precision = precision
        self._chunk_size = cache_chunk_size
        self._reuse_prefix = reuse_prefix
        self._prepared: _PreparedCT | None = None
        self._stamp: tuple[int, int, int, int] | None = None
        self._history: list[tuple[str, str]] = []
        self._prefix = PromptPrefix()
        self._closed = False
        self._lock = Lock()

    @property
    def history(self) -> tuple[tuple[str, str], ...]:
        """Completed ``(question, answer)`` turns in this session, oldest first."""
        with self._lock:
            return tuple(self._history)

    def _prepare(self) -> _PreparedCT:
        if self._closed:
            raise InvalidInputError("CTSession is closed")
        stat = self._source.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if self._stamp is not None and stamp != self._stamp:
            raise InvalidInputError("CT file changed; create a new CTSession")
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
            after = self._source.stat()
            if stamp != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise InvalidInputError("CT file changed during encoding; create a new CTSession")
            self._prepared = _PreparedCT(
                processor,
                model,
                crop,
                embeddings,
                weights_revision(self._model_dir),
            )
            self._stamp = stamp
        return self._prepared

    def report(
        self,
        output_dir: str | Path,
        *,
        prompt: str | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 512,
        overwrite: bool = False,
    ) -> dict[str, object]:
        """Write a report/follow-up with ``generate_report``'s durable completion contract."""
        with self._lock:
            if self._closed:
                raise InvalidInputError("CTSession is closed")
            question, history = validate_generation_options(
                prompt,
                self._anatomy_region,
                enable_thinking,
                max_new_tokens,
                overwrite,
                self._precision,
                self._history,
            )
            directory = Path(output_dir).expanduser().resolve()
            files = ReportFiles(directory)
            files.require_writable(overwrite=overwrite)
            started = perf_counter()
            mx = import_mlx()
            mx.reset_peak_memory()
            prepared = self._prepare()
            inputs = prepared.processor.prompt(question, enable_thinking, history)
            try:
                tokens, metrics = generate(
                    prepared.model,
                    inputs,
                    prepared.embeddings,
                    max_new_tokens=max_new_tokens,
                    prefix=self._prefix if self._reuse_prefix else None,
                    cache_chunk_size=self._chunk_size,
                )
                result = _finish_report(
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
                    prior_turns=len(history),
                    overwrite=overwrite,
                    checkpoint_revision=prepared.weights_revision,
                )
            except BaseException:
                self._prefix = PromptPrefix()
                raise
            self._history.append((question, prepared.processor.decode(tokens)))
            return result

    def reset(self) -> None:
        """Forget conversation/prefix state while retaining the model and CT embeddings."""
        with self._lock:
            if self._closed:
                raise InvalidInputError("CTSession is closed")
            self._history.clear()
            self._prefix = PromptPrefix()

    def close(self) -> None:
        """Release the model, CT embeddings and history; further reports are rejected."""
        with self._lock:
            self._prepared = None
            self._prefix = PromptPrefix()
            self._history.clear()
            self._closed = True

    def __enter__(self) -> CTSession:
        """Return this session, rejecting use after ``close``."""
        if self._closed:
            raise InvalidInputError("CTSession is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the session on any exit path, including exceptions."""
        self.close()
