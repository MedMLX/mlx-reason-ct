"""Native MLX interfaces used by the pinned arithmetic graphs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, overload

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mlx_reason_ct.mlx_model import DecoderCache

type Array = mx.array
type Dtype = mx.Dtype
type Scalar = bool | int | float | complex
type ArrayInput = bool | int | float | complex | list[ArrayInput] | tuple[ArrayInput, ...]


class ArrayConstructor(Protocol):
    @overload
    def __call__[T: Scalar](
        self, value: list[T] | tuple[T, ...], /, dtype: Dtype | None = None
    ) -> Array: ...
    @overload
    def __call__[T: Scalar](self, value: list[list[T]], /, dtype: Dtype | None = None) -> Array: ...
    @overload
    def __call__[T: Scalar](
        self, value: list[list[list[T]]], /, dtype: Dtype | None = None
    ) -> Array: ...
    @overload
    def __call__(
        self,
        value: NDArray[np.generic] | Array | ArrayInput,
        /,
        dtype: Dtype | None = None,
    ) -> Array: ...


class MetalKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: Sequence[Array | int | float],
        template: Sequence[tuple[str, Dtype | int | bool]] = (),
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[Dtype],
        init_value: float | None = None,
    ) -> list[Array]: ...


class Fast(Protocol):
    def layer_norm(
        self,
        x: Array,
        weight: Array | None,
        bias: Array | None,
        eps: float,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def scaled_dot_product_attention(
        self,
        q: Array,
        k: Array,
        v: Array,
        *,
        scale: float,
        mask: None | str | Array = None,
        sinks: Array | None = None,
        force_fused: bool = False,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...

    def metal_kernel(
        self,
        *,
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        source: str,
        header: str = "",
        atomic_outputs: bool = False,
    ) -> MetalKernel: ...


class Mlx(Protocol):
    array: ArrayConstructor
    bfloat16: Dtype
    bool_: Dtype
    float32: Dtype
    int32: Dtype
    int64: Dtype
    uint16: Dtype
    uint32: Dtype
    fast: Fast

    def addmm(
        self,
        c: Array,
        a: Array,
        b: Array,
        /,
        alpha: float = 1.0,
        beta: float = 1.0,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def all(
        self,
        a: Array,
        /,
        axis: None | int | Sequence[int] = None,
        keepdims: bool = False,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def argmax(
        self,
        a: Array,
        /,
        axis: None | int = None,
        keepdims: bool = False,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def contiguous(
        self, a: Array, /, allow_col_major: bool = False, *, stream: mx.StreamOrDevice = None
    ) -> Array: ...
    def conv1d(
        self,
        input: Array,
        weight: Array,
        /,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def cos(self, a: Array, /, *, stream: mx.StreamOrDevice = None) -> Array: ...
    def exp(self, a: Array, /, *, stream: mx.StreamOrDevice = None) -> Array: ...
    def eye(
        self,
        n: int,
        m: int | None = None,
        k: int = 0,
        dtype: Dtype | None = mx.float32,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def full(
        self,
        shape: int | Sequence[int],
        vals: Scalar | Array,
        dtype: Dtype | None = None,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def get_peak_memory(self) -> int: ...
    def isfinite(self, a: Array, stream: mx.StreamOrDevice = None) -> Array: ...
    def log1p(self, a: Array, /, *, stream: mx.StreamOrDevice = None) -> Array: ...
    def mean(
        self,
        a: Array,
        /,
        axis: None | int | Sequence[int] = None,
        keepdims: bool = False,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def minimum(
        self, a: Scalar | Array, b: Scalar | Array, /, *, stream: mx.StreamOrDevice = None
    ) -> Array: ...
    def repeat(
        self,
        array: Array,
        repeats: int,
        axis: int | None = None,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def reset_peak_memory(self) -> None: ...
    def reshape(
        self, a: Array, /, shape: Sequence[int], *, stream: mx.StreamOrDevice = None
    ) -> Array: ...
    def rsqrt(self, a: Array, /, *, stream: mx.StreamOrDevice = None) -> Array: ...
    def sin(self, a: Array, /, *, stream: mx.StreamOrDevice = None) -> Array: ...
    def split(
        self,
        a: Array,
        /,
        indices_or_sections: int | Sequence[int],
        axis: int = 0,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> list[Array]: ...
    def stack(
        self, arrays: list[Array], axis: int | None = 0, *, stream: mx.StreamOrDevice = None
    ) -> Array: ...
    def sum(
        self,
        a: Array,
        /,
        axis: None | int | Sequence[int] = None,
        keepdims: bool = False,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def synchronize(
        self, stream: mx.Stream | mx.ThreadLocalStream | mx.Device | mx.DeviceType | None = None
    ) -> None: ...
    def transpose(
        self, a: Array, /, axes: Sequence[int] | None = None, *, stream: mx.StreamOrDevice = None
    ) -> Array: ...
    def tile(
        self, a: Array, reps: int | Sequence[int], /, *, stream: mx.StreamOrDevice = None
    ) -> Array: ...
    def where(
        self,
        condition: Scalar | Array,
        x: Scalar | Array,
        y: Scalar | Array,
        /,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def zeros(
        self,
        shape: int | Sequence[int],
        dtype: Dtype | None = mx.float32,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def ones(
        self,
        shape: int | Sequence[int],
        dtype: Dtype | None = mx.float32,
        *,
        stream: mx.StreamOrDevice = None,
    ) -> Array: ...
    def broadcast_to(
        self, a: Scalar | Array, /, shape: Sequence[int], *, stream: mx.StreamOrDevice = None
    ) -> Array: ...

    def eval(self, *arrays: Array) -> None: ...
    def load(self, file: str) -> Array | dict[str, Array]: ...
    def concatenate(self, arrays: Sequence[Array], axis: int | None = 0) -> Array: ...
    def pad(self, a: Array, pad_width: Sequence[tuple[int, int]]) -> Array: ...
    @overload
    def arange(self, stop: int | float, *, dtype: Dtype | None = None) -> Array: ...
    @overload
    def arange(
        self,
        start: int | float,
        stop: int | float,
        step: int | float = 1,
        *,
        dtype: Dtype | None = None,
    ) -> Array: ...


class NeuralRuntime(Protocol):
    def gelu(self, x: Array) -> Array: ...


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
