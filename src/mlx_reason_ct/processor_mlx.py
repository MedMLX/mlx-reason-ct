"""Offline CPU tokenizer/template and CT preprocessing for the native runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
from jinja2 import Template
from jinja2.sandbox import ImmutableSandboxedEnvironment

from mlx_reason_ct._host_types import TokenizerContract, Tokenizers
from mlx_reason_ct._payloads import JsonValue
from mlx_reason_ct.ct_crop import CTCrop, load_anatomy_ct
from mlx_reason_ct.errors import InvalidPromptError


def _tool_context(
    tools: Sequence[Mapping[str, JsonValue]] | None,
    history: Sequence[Mapping[str, str]] | None,
) -> tuple[list[dict[str, JsonValue]], list[dict[str, object]]]:
    """Validate function declarations and complete assistant/tool exchanges."""
    declarations: list[dict[str, JsonValue]] = []
    for item in cast(Sequence[object], tools or ()):
        if not isinstance(item, Mapping):
            raise InvalidPromptError("Each tool must be a function declaration")
        declaration = dict(cast(Mapping[str, JsonValue], item))
        function = declaration.get("function")
        if declaration.get("type") != "function" or not isinstance(function, dict):
            raise InvalidPromptError("Each tool requires type=function and a function object")
        name = function.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise InvalidPromptError("Tool names must be identifiers")
        if not isinstance(function.get("parameters"), dict):
            raise InvalidPromptError("Each tool requires a parameters schema")
        try:
            json.dumps(declaration, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise InvalidPromptError("Tool declarations must be finite JSON objects") from error
        declarations.append(declaration)
    turns: list[dict[str, object]] = []
    for index, item in enumerate(cast(Sequence[object], history or ())):
        if not isinstance(item, Mapping):
            raise InvalidPromptError("Each tool-history turn requires role and content")
        turn = dict(cast(Mapping[str, object], item))
        role = "assistant" if index % 2 == 0 else "tool"
        content = turn.get("content")
        if set(turn) != {"role", "content"} or turn["role"] != role:
            raise InvalidPromptError("tool_history must alternate assistant and tool roles")
        if not isinstance(content, str) or not content.strip():
            raise InvalidPromptError("Tool-history content must be nonempty text")
        turns.append(turn)
    if turns and (not declarations or len(turns) % 2):
        raise InvalidPromptError("tool_history requires tools and a final tool response")
    return declarations, turns


@dataclass(frozen=True)
class VolumePrompt:
    """Tokenized single-CT request; ``T`` is the prompt length after image expansion.

    The CT occupies one contiguous run of ``<|image_pad|>`` placeholders starting
    at ``image_start``, one per visual token (``prod(grid)``, 13,824 for the
    pinned 24 x 24 x 24 grid). Their embeddings are spliced in by the generator.

    Attributes:
        text: Rendered chat text with a single, unexpanded image marker.
        input_ids: int64 token ids, shape ``(1, T)``.
        position_ids: int64 3D MRoPE indices, shape ``(3, 1, T)``, rows ordered
            as the visual grid axes (Z, Y, X of the cropped CT).
        attention_mask: all-ones int64, shape ``(1, T)``.
        mm_token_type_ids: int64, shape ``(1, T)``; 1 at image placeholders.
        rope_delta: ``max(position_ids) + 1 - T``; added to the cache offset to
            get the position of each newly generated token.
        image_start: index of the first image placeholder.
    """

    text: str
    input_ids: np.ndarray
    position_ids: np.ndarray
    attention_mask: np.ndarray
    mm_token_type_ids: np.ndarray
    rope_delta: int
    image_start: int


def volume_positions(
    ids: np.ndarray, image_token_id: int, grid: tuple[int, int, int]
) -> tuple[np.ndarray, int, int]:
    """Pinned unmerged 3D MRoPE indices for the public single-volume request.

    Args:
        ids: 1D token ids with the image marker already expanded.
        image_token_id: id of the image placeholder token.
        grid: visual token grid ``(Z, Y, X)`` of the cropped CT; placeholders are
            laid out in row-major order over it (X fastest).

    Returns:
        ``(positions, rope_delta, start)``: int64 positions of shape
        ``(3, 1, len(ids))``, the ``rope_delta`` (``max(positions) + 1 -
        len(ids)``), and the index of the first placeholder. Text before
        the image uses its token index on all three rows; image tokens use their
        grid index plus ``start``; text after continues from ``start`` plus the
        larger of the height and width extents.

    Raises:
        InvalidPromptError: the placeholders are not one contiguous run of
            exactly ``prod(grid)`` tokens.
    """
    indices = np.flatnonzero(ids == image_token_id)
    count = int(np.prod(grid))
    if len(indices) != count or not np.array_equal(
        indices, np.arange(int(indices[0]), int(indices[0]) + count)
    ):
        raise InvalidPromptError("The prompt must contain exactly one CT image marker")
    start = int(indices[0])
    end = start + count
    before = np.broadcast_to(np.arange(start), (3, start))
    visual = np.indices(grid).reshape(3, -1) + start
    after = np.broadcast_to(np.arange(len(ids) - end) + start + max(grid[1:]), (3, len(ids) - end))
    positions = np.concatenate([before, visual, after], axis=1).astype(np.int64)
    return positions[:, None], int(positions.max() + 1 - len(ids)), start


class Processor:
    """Owns only CPU text and non-neural image preparation; no Transformers import."""

    def __init__(self, model_dir: Path) -> None:
        """Read the tokenizer, image geometry and chat template from ``model_dir``.

        The tokenizer runs without padding or truncation; grid, spacing and spatial
        size come from ``image_processor_3d/preprocessor_config.json``. The sandboxed
        template raises ``InvalidPromptError`` from its ``raise_exception`` global.
        """
        tokenizers = cast(Tokenizers, import_module("tokenizers"))
        self.tokenizer: TokenizerContract = tokenizers.Tokenizer.from_file(
            str(model_dir / "tokenizer.json")
        )
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.image_token = "<|image_pad|>"
        self.image_id: int = int(cast(int, self.tokenizer.token_to_id(self.image_token)))
        config = json.loads((model_dir / "image_processor_3d/preprocessor_config.json").read_text())
        self.grid: tuple[int, int, int] = tuple(config["final_grid_size"])
        self.spacing: tuple[float, float, float] = tuple(config["pixdim"])
        self.shape: tuple[int, int, int] = tuple(config["spatial_size"])
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )

        def fail(message: str) -> None:
            raise InvalidPromptError(message)

        cast(dict[str, object], environment.globals)["raise_exception"] = fail
        self.template: Template = environment.from_string(
            (model_dir / "chat_template.jinja").read_text()
        )

    def prompt(
        self,
        question: str,
        enable_thinking: bool,
        history: Sequence[tuple[str, str]] = (),
        *,
        tools: Sequence[Mapping[str, JsonValue]] | None = None,
        tool_history: Sequence[Mapping[str, str]] | None = None,
    ) -> VolumePrompt:
        """Render one request, optionally continuing prior completed turns.

        The pinned CT volume is attached once, on the first user turn. Earlier
        completed turns in ``history`` are ``(question, answer)`` pairs. Answers
        are passed to the template as supplied; the template controls their
        rendering. ``question`` is the new, still-unanswered turn.
        With ``tools``, ``question`` remains the opening request and
        ``tool_history`` contains alternating assistant/tool text turns. The
        checkpoint's own template renders schemas and tool responses; model
        execution never dispatches a tool. Plain Q/A history is exclusive.
        """
        declarations, tool_turns = _tool_context(tools, tool_history)
        if declarations and history:
            raise InvalidPromptError("tools and question/answer history are mutually exclusive")
        opening_question = history[0][0] if history else question
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": opening_question},
                ],
            }
        ]
        if history:
            first_answer = history[0][1]
            messages.append({"role": "assistant", "content": first_answer})
            for turn_question, turn_answer in history[1:]:
                messages.append({"role": "user", "content": turn_question})
                messages.append({"role": "assistant", "content": turn_answer})
            messages.append({"role": "user", "content": question})
        messages.extend(tool_turns)
        text = self.template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tools=declarations,
        )
        if text.count(self.image_token) != 1:
            raise InvalidPromptError("The prompt must contain exactly one CT image marker")
        expanded = text.replace(self.image_token, self.image_token * int(np.prod(self.grid)))
        ids = np.asarray(
            self.tokenizer.encode(expanded, add_special_tokens=False).ids, dtype=np.int64
        )
        positions, delta, start = volume_positions(ids, self.image_id, self.grid)
        return VolumePrompt(
            text,
            ids[None],
            positions,
            np.ones((1, len(ids)), dtype=np.int64),
            (ids == self.image_id).astype(np.int64)[None],
            delta,
            start,
        )

    def image(self, source: Path, region: str | None) -> tuple[np.ndarray, CTCrop]:
        """Load, resample and crop a NIfTI CT into the vision-encoder input.

        Args:
            source: HU NIfTI volume.
            region: ``"chest"``, ``"abdomen"`` or ``None`` for a plain centered crop.

        Returns:
            ``(pixels, crop)``. ``pixels`` is float32 of shape ``(1, 1, Z, Y, X)``
            (batch, channel, then the LPS-oriented crop axes reversed from the
            ``(X, Y, Z)`` order of ``crop.values``). HU values are clipped to
            [-1000, 1000] and divided by 1000, so the range is [-1, 1].
        """
        crop = load_anatomy_ct(source, region, spacing=self.spacing, roi=self.shape)
        pixels = np.clip(crop.values, -1000.0, 1000.0) / np.float32(1000.0)
        return np.ascontiguousarray(pixels.transpose(2, 1, 0)[None, None]), crop

    def decode(self, tokens: list[int]) -> str:
        """Decode generated token ids to text, dropping special tokens."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
