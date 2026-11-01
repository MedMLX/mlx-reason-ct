"""Metal implementation of Transformers 5.10.4's 64-token gated-delta prefill.

The chunk algebra and FP32 accumulation boundaries follow torch_chunk_gated_delta_rule.
Sequential recurrence can produce materially different hidden states over
long CT token sequences.
"""

from __future__ import annotations

import math
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_reason_ct._native_types import Array, MetalKernel, Mlx

from mlx_reason_ct.runtime import import_mlx

mx: Mlx = import_mlx()


@cache
def _triangle_kernel() -> MetalKernel:
    """Build the FP32 64-row solve with MLX 0.32.4's column reduction order."""
    return mx.fast.metal_kernel(
        name="reason_ct_triangle_fp32",
        input_names=["input"],
        output_names=["output"],
        source="""
        uint tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup;
        uint group = simdgroup_index_in_threadgroup;
        uint base = threadgroup_position_in_grid.x * 4096;
        threadgroup float a[4096];
        threadgroup float updated[64];
        for (uint j = tid; j < 4096; j += 256) a[j] = input[base + j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 1; i < 64; ++i) {
            for (uint c = group; c < 64; c += 8) {
                float total = 0.0f;
                if (i == 1) {
                    // MLX treats the one-element reduction as an identity.
                    if (c == 0) total = a[64] * a[0];
                } else if (i < 32) {
                    uint stripes = min(i, 8u);
                    if (lane < stripes && c < i) {
                        for (uint r = lane; r < i; r += stripes) {
                            volatile float product = a[i * 64 + r] * a[r * 64 + c];
                            total = product + total;
                        }
                    }
                    float combined = simd_shuffle(total, 0);
                    for (uint r = 1; r < stripes; ++r) {
                        combined = simd_shuffle(total, r) + combined;
                    }
                    total = combined;
                } else {
                    if (c < i) {
                        for (uint r = lane; r < i; r += 32) {
                            volatile float product = a[i * 64 + r] * a[r * 64 + c];
                            total = product + total;
                        }
                    }
                    total = simd_sum(total);
                }
                if (lane == 0 && c < i) updated[c] = a[i * 64 + c] + total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < i) a[i * 64 + tid] = updated[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint j = tid; j < 4096; j += 256) output[base + j] = a[j];
        """,
    )


def triangle_fp32(attn: Array) -> Array:
    """Resolve strictly lower FP32 ``(..., 64, 64)`` attention in one dispatch.

    The caller supplies the documented dtype and shape; this kernel wrapper
    does not validate them.
    """
    return _triangle_kernel()(
        inputs=[attn],
        grid=(attn.size // 4096 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[attn.shape],
        output_dtypes=[mx.float32],
    )[0]


def source_cumsum(g: Array) -> Array:
    """ATen CUDA's Sklansky order, including its row-count-dependent block size.

    The input is a nonempty array of FP32 rows with last-axis length 64; these
    preconditions are supplied by the chunked recurrence and are not checked.
    Decay uses differences of these sums inside exp; reassociation at long CT
    contexts measurably changes decoder states. No higher/lower precision cast.
    """
    rows = math.prod(g.shape[:-1])
    threads_log = min(9, max(4, (9 + 6 - (rows - 1).bit_length()) // 2))
    block_size = min(64, 2 << threads_log)
    output: list[Array] = []
    previous = mx.zeros(g.shape[:-1], dtype=mx.float32)
    index = mx.arange(block_size)
    for start in range(0, 64, block_size):
        part = g[..., start : start + block_size]
        part = mx.concatenate([part[..., :1] + previous[..., None], part[..., 1:]], axis=-1)
        step = 1
        while step < block_size:
            source = (index // (2 * step)) * 2 * step + step - 1
            part = mx.where(index % (2 * step) >= step, part + part[..., source], part)
            step *= 2
        previous = part[..., -1]
        output.append(part)
    return mx.concatenate(output, axis=-1)


def chunk_delta(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Array,
    *,
    source_bfloat16: bool = False,
) -> tuple[Array, Array]:
    """Run the chunked gated-delta recurrence over ``T`` tokens in 64-token chunks.

    Args:
        q: normalized and scaled queries, ``(B, T, Hk, 128)``.
        k: normalized keys, ``(B, T, Hk, 128)``. Repeated to ``Hv`` heads.
        v: values, ``(B, T, Hv, 128)``.
        g: log decay (non-positive), ``(B, T, Hv)``.
        beta: write strength, ``(B, T, Hv)``.
        state: recurrent state ``(B, Hv, V, K)`` (MLX-LM's V, K order).
        source_bfloat16: use the source-BF16 exponential and reduction kernels.

    Returns:
        ``(output, state)``: float32 ``(B, T, Hv, 128)`` and the updated state in
        the same layout as the input.

    Callers must supply a nonempty sequence, positive head counts with ``Hv`` a
    multiple of ``Hk``, and mutually compatible shapes. This helper does not
    validate those preconditions.
    """
    from mlx_reason_ct import source_arithmetic

    exponential = source_arithmetic.exp if source_bfloat16 else mx.exp
    batch, length, heads, _ = v.shape
    count = (length + 63) // 64
    padding = count * 64 - length
    repeat = heads // q.shape[2]
    q, k = mx.repeat(q, repeat, axis=2), mx.repeat(k, repeat, axis=2)

    def chunks(x: Array) -> Array:
        x = mx.transpose(x, (0, 2, 1, 3)).astype(mx.float32)
        x = mx.pad(x, [(0, 0), (0, 0), (0, padding), (0, 0)])
        return mx.reshape(x, (batch, heads, count, 64, x.shape[-1]))

    # Pad T to a multiple of 64 with zeros; padded rows are sliced off the output.
    q, k, v = chunks(q), chunks(k), chunks(v)
    beta = chunks(beta[..., None])
    g = source_cumsum(chunks(g[..., None])[..., 0])
    lower = mx.arange(64)[:, None] >= mx.arange(64)[None, :]
    decay = mx.where(lower, exponential(mx.minimum(g[..., :, None] - g[..., None, :], 0)), 0)
    kb, vb = k * beta, v * beta
    attn = mx.where(mx.eye(64, dtype=mx.bool_) | ~lower, 0, -(kb @ k.swapaxes(-1, -2)) * decay)
    if source_bfloat16:
        # The source profile has a distinct reduction and gradual-underflow
        # contract; retain its explicit materialization boundaries.
        for i in range(1, 64):
            row = attn[..., i, :i]
            product = row[..., :, None] * attn[..., :i, :i]
            attn[..., i, :i] = row + source_arithmetic.sum_product(product)
            mx.eval(attn)
    else:
        attn = triangle_fp32(attn)
    attn = attn + mx.eye(64)
    values = attn @ vb
    keys = attn @ (kb * exponential(g)[..., None])
    state = state.swapaxes(-1, -2)
    output: list[Array] = []
    mx.eval(values, keys, decay)
    for i in range(count):
        qi, ki = q[:, :, i], k[:, :, i]
        gi = g[:, :, i]
        new = values[:, :, i] - keys[:, :, i] @ state
        attention = (qi @ ki.swapaxes(-1, -2)) * decay[:, :, i]
        part = (qi * exponential(gi)[..., None]) @ state + attention @ new
        state = (
            state * exponential(gi[..., -1, None, None])
            + (ki * exponential(gi[..., -1, None] - gi)[..., None]).swapaxes(-1, -2) @ new
        )
        mx.eval(part, state)
        output.append(part)
    result = mx.transpose(mx.concatenate(output, axis=2)[:, :, :length], (0, 2, 1, 3))
    return result, mx.contiguous(state.swapaxes(-1, -2))
