"""Metal arithmetic for the pinned original RTX 4090 BF16 source profile.

Learned operations stay on Metal. Compact unary assets contain only losslessly
encoded, model-independent source instruction outputs and fixed rotary constants.
See docs/development.md for source pins and independent controls.
"""

from __future__ import annotations

import json
import struct
import zlib
from functools import cache, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

from numpy.typing import NDArray

from mlx_reason_ct._payloads import ArithmeticManifest, EncodedTable, KernelSpec

if TYPE_CHECKING:
    from mlx_reason_ct._native_types import Array, MetalKernel, Mlx

import numpy as np

from mlx_reason_ct.errors import MissingDependencyError
from mlx_reason_ct.runtime import import_mlx

mx: Mlx = import_mlx()
ASSETS = Path(__file__).with_name("source_arithmetic_assets")
# (in_features, out_features) of single-row decoder projections mapped to the number of
# lanes the source GPU kernel was observed to use; see ``cached_projection``.
OBSERVED_LANES = {
    (2560, 32): 32,
    (2560, 1024): 16,
    (2560, 4096): 16,
    (2560, 8192): 16,
    (2560, 9216): 16,
    (4096, 2560): 16,
    (9216, 2560): 32,
}


@lru_cache(maxsize=1)
def _manifest() -> ArithmeticManifest:
    return cast(ArithmeticManifest, json.loads((ASSETS / "manifest.json").read_bytes()))


@cache
def _decoded(name: str) -> NDArray[np.uint32]:
    row = _manifest()["tables"][name]
    data = (ASSETS / row["file"]).read_bytes()
    if data[:10] != b"RADNNCUDA\x01":
        raise MissingDependencyError("Invalid NV-Reason-CT source arithmetic asset")
    length = struct.unpack("<I", data[10:14])[0]
    metadata = cast(EncodedTable, json.loads(data[14 : 14 + length]))
    delta = zlib.decompress(data[14 + length :])
    if (
        metadata["dtype"] != "<u4"
        or metadata["codec"] != "zlib.modular_uint32_delta"
        or len(delta) != metadata["decoded_bytes"]
    ):
        raise MissingDependencyError("Invalid NV-Reason-CT source arithmetic payload")
    values = np.cumsum(np.frombuffer(delta, dtype="<u4"), dtype=np.uint32)
    values = values.reshape(metadata["shape"])
    values.setflags(write=False)
    return values


@cache
def _table(name: str) -> Array:
    return mx.array(_decoded(name), dtype=mx.uint32)


@lru_cache(maxsize=1)
def _kernel_text() -> dict[str, KernelSpec]:
    return cast(
        dict[str, KernelSpec],
        json.loads((ASSETS / "kernels.json").read_bytes()),
    )


@cache
def kernel(name: str) -> MetalKernel:
    """Create and cache the named Metal kernel from the packaged kernel specification."""
    value = _kernel_text()[name]
    return mx.fast.metal_kernel(
        name="nv_reason_source_" + name,
        input_names=value["inputs"],
        output_names=value["outputs"],
        header=value["header"],
        source=value["source"],
        atomic_outputs=value.get("atomic_outputs", False),
    )


def _elementwise(name: str, values: Array, extra: list[Array] | None = None) -> Array:
    return kernel(name)(
        inputs=[values, *(extra or [])],
        template=[("Count", values.size)],
        grid=(values.size, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[mx.float32],
    )[0]


def rsqrt(values: Array) -> Array:
    """Elementwise FP32 reciprocal square root using the encoded source instruction table."""
    return _elementwise("rsqrt", values, [_table("rsqrt")])


def _exp_tables() -> list[Array]:
    return [_table("exp2_" + name) for name in ("positive", "negative", "negative_direct")]


def exp(values: Array) -> Array:
    """Elementwise FP32 exponential using the encoded source exp2 tables."""
    return _elementwise("exp", values, _exp_tables())


def log1p(values: Array) -> Array:
    """Elementwise FP32 ``log(1 + x)`` Metal kernel reproducing the source's log1p."""
    return _elementwise("log1p", values)


def silu(values: Array) -> Array:
    """SiLU computed in FP32 from the encoded source table, cast back to the input dtype."""
    return _activation(values, False)


def sigmoid(values: Array) -> Array:
    """Sigmoid computed in FP32 from the encoded source table, cast back to the input dtype."""
    return _activation(values, True)


def _activation(values: Array, sigmoid_mode: bool) -> Array:
    table = _table("sigmoid_bf16" if sigmoid_mode else "silu_fp32")
    result = kernel("activation")(
        inputs=[values.astype(mx.float32), table],
        template=[("Count", values.size), ("Sigmoid", sigmoid_mode)],
        grid=(values.size, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[mx.float32],
    )[0]
    return result.astype(values.dtype)


def mean_square(square: Array) -> Array:
    """Mean over the last axis of already-squared FP32 values, in the source's tree order.

    The last-axis width must be 128, 256 or 2560. Returns float32 ``(..., 1)``.
    """
    width = square.shape[-1]
    # FP32 bit pattern of 1 / width, applied once after the tree sum.
    factor = {128: 0x3C000000, 256: 0x3B800000, 2560: 0x39CCCCCD}[width]
    rows = square.size // width
    # Pinned ATen Reduce.cuh vectorizes four inputs per lane, then chooses
    # block width from both the reduction width and the number of output rows.
    # Cached RMS rows therefore use a different tree from long prefill rows.
    vector_lanes = min(1 << ((width // 4).bit_length() - 1), 512)
    initial_lanes = min(vector_lanes, 32)
    row_lanes = min(1 << (rows.bit_length() - 1), 512 // initial_lanes)
    lanes = min(vector_lanes, 512 // row_lanes)
    return kernel("mean")(
        inputs=[square],
        template=[("Rows", rows), ("Width", width), ("FactorBits", factor), ("Lanes", lanes)],
        grid=(rows * lanes, 1, 1),
        threadgroup=(lanes, 1, 1),
        output_shapes=[(*square.shape[:-1], 1)],
        output_dtypes=[mx.float32],
    )[0]


def sum_square(square: Array) -> Array:
    """Sum already-squared FP32 values over a last axis of width 128 into ``(..., 1)``."""
    rows = square.size // 128
    return kernel("square_sum")(
        inputs=[square],
        template=[("Rows", rows)],
        grid=(rows * 16, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*square.shape[:-1], 1)],
        output_dtypes=[mx.float32],
    )[0]


def sum_product(product: Array) -> Array:
    """Sum ``(..., rows, W)`` products over the second-to-last axis into ``(..., W)``."""
    width = product.shape[-1]
    shape = (*product.shape[:-2], width)
    outputs = product.size // width
    return kernel("triangle_sum")(
        inputs=[product],
        template=[("Width", width), ("Outputs", outputs)],
        grid=(outputs, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.float32],
    )[0]


def _operand(values: Array) -> Array:
    # Widening to FP32 is exact; BF16 and FP32 operands are read in place.
    return values if values.dtype in (mx.bfloat16, mx.float32) else values.astype(mx.float32)


def operand_flags(*operands: Array) -> Array:
    """Flag special values in two operands.

    Returns a ``(1,)`` uint32 that is nonzero if either operand holds inf, NaN or a
    subnormal. Exactly two operands are expected. Flagged projections skip the fast
    alignment path.
    """
    flags = [
        kernel("operand_check")(
            inputs=[values],
            template=[("Count", values.size)],
            grid=(values.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(1,)],
            output_dtypes=[mx.uint32],
            init_value=0,
        )[0]
        for values in operands
    ]
    return flags[0] | flags[1]


def projection(
    x: Array, weight: Array, *, bias: Array | None = None, slices: int = 1, sliced: bool = False
) -> Array:
    """Measured K8 HMMA carries and source BF16 epilogue/partition stores.

    Args:
        x: activations ``(B, T, K)``, BF16 or FP32.
        weight: ``(N, K)`` weight, BF16 or FP32.
        bias: optional ``(N,)`` bias, added in the kernel.
        slices: number of equal K partitions; partial results are stored between
            partitions as in the source.
        sliced: require partition widths that are multiples of 64 instead of 8.

    Returns:
        BF16 ``(B, T, N)``.

    Raises:
        ValueError: ``K`` does not split into ``slices`` equal partitions whose width
            is a multiple of 8 (64 when ``sliced``).
    """
    x, weight = _operand(x), _operand(weight)
    length, width = x.shape[-1], weight.shape[0]
    part_width = length // slices
    if part_width * slices != length or part_width % (64 if sliced else 8):
        raise ValueError("Source projection width does not match its K8 partition")
    special = operand_flags(x, weight)
    zero = mx.zeros((1,), dtype=mx.float32)
    parts: list[Array] = []
    for start in range(0, x.shape[1], 512):
        part = x[:, start : start + 512]
        rows = part.size // length
        # More rows per thread reuse each weight block; small grids keep one.
        per_thread = 4 if rows * width >= 262144 else 2 if rows * width >= 65536 else 1
        value = mx.zeros((*part.shape[:-1], width), dtype=mx.float32)
        for first in range(0, length, part_width):
            value = kernel("projection")(
                inputs=[
                    part,
                    weight,
                    value,
                    zero if bias is None else bias.astype(mx.float32),
                    special,
                ],
                template=[
                    ("M", rows),
                    ("N", width),
                    ("K", part_width),
                    ("Ldx", length),
                    ("Ldw", length),
                    ("KOffset", first),
                    ("Rows", per_thread),
                    ("Sliced", sliced),
                    ("HasC", first > 0),
                    ("HasBias", bias is not None),
                ],
                grid=(-(-rows // per_thread) * width, 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(*part.shape[:-1], width)],
                output_dtypes=[mx.float32],
            )[0]
            mx.eval(value)
        parts.append(value.astype(mx.bfloat16))
    return mx.concatenate(parts, axis=1)


def cached_projection(x: Array, weight: Array) -> Array:
    """Single-token projection ``(1, 1, K)`` by ``(N, K)`` weight, returned as BF16 ``(1, 1, N)``.

    ``(K, N)`` must be a key of ``OBSERVED_LANES``.
    """
    width, length = weight.shape
    lanes = OBSERVED_LANES[(length, width)]
    return kernel("cached_projection")(
        inputs=[x, weight],
        template=[("Lanes", lanes), ("K", length), ("N", width)],
        grid=(lanes, ((width + 3) // 4) * 4, 1),
        threadgroup=(lanes, 4, 1),
        output_shapes=[(1, 1, width)],
        output_dtypes=[mx.float32],
    )[0].astype(mx.bfloat16)


def convolution(x: Array, weight: Array) -> Array:
    """C1 padded to C8 inside each source Conv3D tap's HMMA operation.

    ``x`` is the flattened patch matrix ``(1, 13824, 512)`` and ``weight`` the
    ``(864, 512)`` patch projection; returns BF16 ``(1, 13824, 864)`` without bias.
    """
    parts: list[Array] = []
    for start in range(0, x.shape[1], 128):
        part = x[:, start : start + 128]
        rows, width = part.size // 512, weight.shape[0]
        value = kernel("stem")(
            inputs=[part.astype(mx.float32), weight.astype(mx.float32)],
            template=[("M", rows), ("N", width), ("K", 512)],
            grid=(rows * width, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(*part.shape[:-1], width)],
            output_dtypes=[mx.float32],
        )[0].astype(mx.bfloat16)
        mx.eval(value)
        parts.append(value)
    return mx.concatenate(parts, axis=1)


def layernorm(values: Array, weight: Array, bias: Array, eps: float) -> Array:
    """LayerNorm over the last axis with FP32 statistics, returned as BF16.

    ``weight`` and ``bias`` have the last-axis width; ``values`` is any shape ending
    in that width.
    """
    fp = values.astype(mx.float32)
    width, rows = fp.shape[-1], fp.size // fp.shape[-1]
    reciprocal = mx.array(
        np.concatenate(
            [
                np.zeros(1, dtype=np.float32),
                (1 / np.arange(1, width + 1, dtype=np.float64)).astype(np.float32),
            ]
        )
    )
    means, variances = kernel("layernorm_statistics")(
        inputs=[fp, reciprocal],
        template=[("Width", width)],
        grid=(rows * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(*fp.shape[:-1], 1)] * 2,
        output_dtypes=[mx.float32] * 2,
    )
    inverses = rsqrt(variances + eps)
    return kernel("layernorm_affine")(
        inputs=[fp, means, inverses, weight.astype(mx.float32), bias.astype(mx.float32)],
        template=[("Width", width), ("Count", fp.size)],
        grid=(fp.size, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[fp.shape],
        output_dtypes=[mx.float32],
    )[0].astype(mx.bfloat16)


def vision_constants() -> tuple[Array, Array]:
    """Return the source's FP32 vision rotary ``(sin, cos)``, each ``(13824, 72)``.

    Rows follow the 24 x 24 x 24 patch grid in ``(Z, Y, X)`` row-major order.
    """
    values = _decoded("vision_rotary").view(np.float32)
    coords = np.indices((24, 24, 24), dtype=np.int32).reshape(3, -1).T
    sin, cos = [
        mx.array(values[kind][coords].reshape(13824, 36).repeat(2, axis=-1)) for kind in (0, 1)
    ]
    return sin, cos


def gated_delta(
    q: Array, k: Array, v: Array, decay: Array, beta: Array, state: Array
) -> tuple[Array, Array]:
    """One-token gated-delta recurrence for batch size 1.

    Args:
        q: ``(1, 1, Hk, 128)`` normalized and scaled queries.
        k: ``(1, 1, Hk, 128)`` normalized keys.
        v: ``(1, 1, 32, 128)`` values.
        decay: per-head decay factor (the exponential of the log decay), ``(1, 1, 32)``.
        beta: ``(1, 1, 32)`` write strength.
        state: ``(1, 32, V, K)`` recurrent state in (V, K) order.

    Returns:
        ``(output, state)``: float32, shaped like ``v`` and ``state``.
    """
    output, updated = kernel("recurrence")(
        inputs=[q, k, v, decay, beta, state.swapaxes(-1, -2)],
        template=[("Ratio", 32 // q.shape[2])],
        grid=(4096, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[v.shape, state.shape],
        output_dtypes=[mx.float32] * 2,
    )
    return output, updated.swapaxes(-1, -2)


def _split_attention(q: Array, k: Array, v: Array) -> Array:
    # Pinned source 128-SM GQA dispatch; independently checked at every transition.
    splits = (k.shape[2] + 255) // 256
    meta = mx.array([1, k.shape[2], 16, 4, 0, splits, 1], dtype=mx.int32)
    parts, lse, _ = kernel("split_attention")(
        inputs=[
            q,
            k,
            v,
            meta,
            mx.array(0.0625 * 1.4426950408889634, dtype=mx.float32),
            mx.array(0.0625, dtype=mx.float32),
            *_exp_tables(),
            _table("lg2"),
        ],
        template=[("T", mx.bfloat16), ("D", 256), ("Causal", False)],
        grid=(128, 16, splits),
        threadgroup=(128, 1, 1),
        output_shapes=[(splits, 1, 16, 1, 256), (splits, 1, 16, 1), (splits, 1, 16, 1)],
        output_dtypes=[mx.float32] * 3,
    )
    scales, _ = kernel("split_scales")(
        inputs=[lse, *_exp_tables()],
        template=[("Splits", splits)],
        grid=(16, 1, 1),
        threadgroup=(16, 1, 1),
        output_shapes=[(splits, 1, 16, 1), (32,)],
        output_dtypes=[mx.float32] * 2,
    )
    return kernel("split_combine")(
        inputs=[parts, scales],
        template=[("Splits", splits)],
        grid=(4096, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[mx.float32],
    )[0].astype(mx.bfloat16)


def attention(q: Array, k: Array, v: Array, *, causal: bool, offset: int) -> Array:
    """BF16 attention with the source's K8 accumulation order.

    Args:
        q: ``(B, heads, Tq, D)``; ``D`` is 72 (vision) or 256 (decoder).
        k: ``(B, kv_heads, Tk, D)``.
        v: ``(B, kv_heads, Tk, D)``.
        causal: mask keys after each query's position.
        offset: number of keys preceding the first query.

    Returns:
        BF16 array shaped like ``q``. Queries are processed in blocks of 64 (vision)
        or 128 (decoder). A single-token decoder query against 13,824-16,384 cached
        keys uses the source's split-key dispatch.
    """
    if (
        q.shape == (1, 16, 1, 256)
        and k.shape[:2] == (1, 4)
        and 13824 <= k.shape[2] <= 16384
        and offset == k.shape[2] - 1
    ):
        return _split_attention(q, k, v)
    logical = q.shape[-1]
    padded = ((logical + 31) // 32) * 32
    if padded != logical:
        q, k, v = [
            mx.pad(value, [(0, 0), (0, 0), (0, 0), (0, padded - logical)]) for value in (q, k, v)
        ]
    batch, heads, length, width = q.shape
    outputs: list[Array] = []
    for start in range(0, length, 64 if logical == 72 else 128):
        query = q[:, :, start : start + (64 if logical == 72 else 128)]
        count = query.shape[2]
        meta = mx.array([count, k.shape[2], heads, k.shape[1], offset + start], dtype=mx.int32)
        value = kernel("attention")(
            inputs=[
                query,
                k,
                v,
                meta,
                mx.array(logical**-0.5 * 1.4426950408889634, dtype=mx.float32),
                *_exp_tables(),
            ],
            template=[("T", mx.bfloat16), ("D", width), ("Causal", causal)],
            grid=(((count + 31) // 32) * 128, heads, batch),
            threadgroup=(128, 1, 1),
            output_shapes=[query.shape],
            output_dtypes=[mx.bfloat16],
        )[0]
        mx.eval(value)
        outputs.append(value)
    return mx.concatenate(outputs, axis=2)[..., :logical]
