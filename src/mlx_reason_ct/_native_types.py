"""Native MLX interfaces used by the pinned arithmetic graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_reason_ct.mlx_model import DecoderCache

type Array = mx.array
type Dtype = mx.Dtype


class GenerationModel(Protocol):
    @property
    def eos_ids(self) -> tuple[int, ...]: ...
    def embed(self, ids: Array) -> Array: ...
    def decode(
        self,
        embeddings: Array,
        positions: Array,
        cache: DecoderCache | None = None,
    ) -> tuple[Array, DecoderCache]: ...
    def logits(self, hidden: Array) -> Array: ...
