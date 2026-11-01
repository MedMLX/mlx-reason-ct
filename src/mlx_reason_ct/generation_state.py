"""Host-side capacity and exact prompt-prefix admission for reusable decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mlx_reason_ct.errors import InvalidInputError
from mlx_reason_ct.processor_mlx import VolumePrompt

if TYPE_CHECKING:
    from mlx_reason_ct._native_types import Array
    from mlx_reason_ct.mlx_model import DecoderCache


def cache_capacity(required: int, chunk_size: int) -> int:
    """Round token capacity up without changing logical positions or attention extent."""
    if type(chunk_size) is not int or chunk_size < 0 or chunk_size % 256:
        raise InvalidInputError("cache_chunk_size must be zero or a positive multiple of 256")
    return required if not chunk_size else (required + chunk_size - 1) // chunk_size * chunk_size


@dataclass
class PromptPrefix:
    """A complete evaluated prefill snapshot, owned by one fixed-CT session.

    Only an exact token AND three-axis-position prefix may be reused. A changed
    template or tokenization rebuilds prefill. Recurrent state cannot be cropped
    backwards to an arbitrary common prefix, so partial matches never qualify.
    """

    inputs: VolumePrompt | None = None
    cache: DecoderCache | None = None
    logits: Array | None = None

    def matching_tokens(self, inputs: VolumePrompt) -> int:
        """Length of the reusable prefix of ``inputs``: all stored tokens, or 0 if any differ."""
        previous = self.inputs
        if previous is None or self.cache is None or self.logits is None:
            return 0
        count = previous.input_ids.shape[1]
        if (
            inputs.input_ids.shape[1] < count
            or inputs.image_start != previous.image_start
            or not np.array_equal(inputs.input_ids[:, :count], previous.input_ids)
            or not np.array_equal(inputs.position_ids[:, :, :count], previous.position_ids)
            or not np.array_equal(inputs.mm_token_type_ids[:, :count], previous.mm_token_type_ids)
        ):
            return 0
        return count
