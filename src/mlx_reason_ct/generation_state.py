"""Host-side capacity and exact prompt-prefix admission for reusable decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mlx_reason_ct.errors import InvalidInputError
from mlx_reason_ct.processor_mlx import VolumePrompt

if TYPE_CHECKING:
    from mlx_reason_ct.mlx_model import DecoderCache


def cache_capacity(required: int, chunk_size: int) -> int:
    """Round token capacity up without changing logical positions or attention extent."""
    if type(chunk_size) is not int or chunk_size < 0 or chunk_size % 256:
        raise InvalidInputError("cache_chunk_size must be zero or a positive multiple of 256")
    return required if not chunk_size else (required + chunk_size - 1) // chunk_size * chunk_size


@dataclass
class PromptPrefix:
    """Evaluated prefill of the prompt tokens that precede a session's questions.

    One fixed-CT session owns it. Every prompt of a session starts with the same
    tokens up to the end of the CT (a tool block, if any, precedes the CT), so the
    decoder state at a boundary inside them serves each later prompt, which then
    decodes only the tokens after the boundary. Recurrent state cannot be cropped
    backwards, so the boundary is fixed when the snapshot is built. Only an exact
    token and three-axis-position match is reused; a different tool block rebuilds
    the snapshot.

    The boundary decides whether the two calls repeat the arithmetic of one call
    over the whole prompt. MLX 0.32.4 picks its Metal matmul kernel from the row
    count (thin kernels up to 64 rows, split-K up to 1,024 rows or while the input
    width is at least the row count, a second thin variant below 2,048), the
    feed-forward runs in 2,048-row chunks, and the recurrence in 64-token chunks
    evaluated 16 at a time. The exact boundary is a multiple of 2,048 that leaves
    more than 2,560 tokens to decode, so every operation meets the kernel and
    chunking of the whole prefill; on the M5 Max the logits are then bit-identical.
    The fast boundary is the last multiple of 64: the few remaining rows take the
    small-row kernels, whose products round differently in rare elements.
    """

    fast: bool = False
    input_ids: np.ndarray | None = None
    position_ids: np.ndarray | None = None
    cache: DecoderCache | None = None

    def boundary(self, inputs: VolumePrompt, image_tokens: int) -> int:
        """Token count to snapshot for ``inputs``, whose CT spans ``image_tokens`` tokens."""
        end = inputs.image_start + image_tokens + 1  # through the CT's closing marker
        return end // 64 * 64 if self.fast else max(end - 2561, 0) // 2048 * 2048

    def matches(self, inputs: VolumePrompt, boundary: int) -> bool:
        """Whether the stored prefill is exactly the first ``boundary`` tokens of ``inputs``."""
        if self.input_ids is None or self.position_ids is None or self.cache is None:
            return False
        return (
            self.input_ids.shape[1] == boundary
            and np.array_equal(inputs.input_ids[:, :boundary], self.input_ids)
            and np.array_equal(inputs.position_ids[:, :, :boundary], self.position_ids)
        )

    def clear(self) -> None:
        """Release the stored prefill, so a replacement is not decoded beside it."""
        self.input_ids = self.position_ids = self.cache = None

    def store(self, inputs: VolumePrompt, boundary: int, cache: DecoderCache) -> None:
        """Keep ``cache``, the evaluated prefill of the first ``boundary`` tokens of ``inputs``."""
        self.input_ids = inputs.input_ids[:, :boundary].copy()
        self.position_ids = inputs.position_ids[:, :, :boundary].copy()
        self.cache = cache
