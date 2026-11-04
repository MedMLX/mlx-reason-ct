"""FP32 matrix-vector products that read BF16-stored weights.

NV-Reason-CT's learned weights are BF16 values; the FP32 bundle widens them
losslessly. Single-token decoding is bound by weight bandwidth, so the decoder
stores its 2-D language-model weights as BF16 and this kernel widens each element
to FP32 in registers. The source is MLX 0.32.4's non-transposed ``GEMVKernel``
(``mlx/backend/metal/kernels/gemv.h``) with only the matrix load type changed, and
``config`` reproduces that release's ``gemv_axbpy`` tile selection. Products and
reductions therefore run in MLX's FP32 order: on Metal the outputs were bit-identical
to ``x @ w.astype(mx.float32).T`` for every decoder and LM-head shape.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_reason_ct._native_types import Array, MetalKernel, Mlx

from mlx_reason_ct.runtime import import_mlx

mx: Mlx = import_mlx()

_SOURCE = """
    constexpr int threadsM = BM * SM;
    constexpr int threadsN = BN * SN;
    constexpr int blockM = threadsM * TM;
    constexpr int blockN = threadsN * TN;
    threadgroup float tgp_memory[BN > 1 ? BN * (blockM + TM) : 1];
    const int in_vec_size = sizes[0];
    const int out_vec_size = sizes[1];
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;

    float result[TM] = {0};
    float inter[TN];
    float v_coeff[TN];
    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);
    const int sgN = BN != 1 ? (simd_gid % BN) : 0;
    const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;
    const int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;
    int out_row = tid.x * blockM + bm;
    if (out_row >= out_vec_size) return;
    // Shift the tail block inwards so every read stays inside the matrix.
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;
    const device bfloat16_t* m = mat + size_t(out_row) * in_vec_size;
    const int n_iter = in_vec_size / blockN;
    const int leftover = in_vec_size - blockN * n_iter;
    for (int i = 0; i < n_iter; ++i) {
      for (int tn = 0; tn < TN; tn++) v_coeff[tn] = vec[bn + tn];
      int offset = 0;
      for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) inter[tn] = static_cast<float>(m[offset + bn + tn]);
        for (int tn = 0; tn < TN; tn++) result[tm] += inter[tn] * v_coeff[tn];
        offset += in_vec_size;
      }
      bn += blockN;
    }
    if (leftover > 0) {
      for (int tn = 0; tn < TN; tn++) v_coeff[tn] = bn + tn < in_vec_size ? vec[bn + tn] : 0.0f;
      for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
          inter[tn] = bn + tn < in_vec_size
              ? static_cast<float>(m[tm * in_vec_size + bn + tn]) : 0.0f;
        }
        for (int tn = 0; tn < TN; tn++) result[tm] += inter[tn] * v_coeff[tn];
      }
    }
    for (int tm = 0; tm < TM; tm++) {
      for (ushort sn = SN / 2; sn >= 1; sn >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], sn);
      }
    }
    if (BN > 1) {
      threadgroup float* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
      if (thrN == 0) {
        for (int tm = 0; tm < TM; tm++) tgp_results[tm] = result[tm];
        threadgroup_barrier(mem_flags::mem_none);
        if (sgN == 0) {
          for (int sgn = 1; sgn < BN; sgn++) {
            for (int tm = 0; tm < TM; tm++) result[tm] += tgp_results[sgn * (blockM + TM) + tm];
          }
        }
      }
    }
    if (simdN == 0 && thrN == 0) {
      for (int tm = 0; tm < TM; tm++) out[out_row + tm] = result[tm];
    }
"""


@cache
def _kernel() -> MetalKernel:
    return mx.fast.metal_kernel(
        name="reason_ct_bf16_weight_gemv",
        input_names=["mat", "vec", "sizes"],
        output_names=["out"],
        source=_SOURCE,
    )


def config(in_size: int, out_size: int) -> tuple[int, int, int, int, int, int]:
    """MLX 0.32.4 ``gemv_axbpy`` tiles ``(bm, bn, sm, sn, tm, tn)`` for a row-major matrix."""
    bm, bn, sm, sn, tm, tn = (8 if out_size >= 4096 else 4), 1, 1, 32, 4, 4
    if in_size <= 64:
        bm, sm, sn = 1, 8, 4
    elif in_size >= 16 * out_size:
        bm, bn = 1, 8
    return bm, bn, sm, sn, (1 if out_size < tm else tm), tn


def gemv(x: Array, weight: Array) -> Array:
    """Return FP32 ``weight @ x`` for FP32 ``x`` of shape ``(K,)`` and BF16 ``weight`` ``(N, K)``.

    Callers supply those dtypes and shapes; they are not validated here.
    """
    out_size, in_size = weight.shape
    bm, bn, sm, sn, tm, tn = config(in_size, out_size)
    groups = (out_size + bm * sm * tm - 1) // (bm * sm * tm)
    return _kernel()(
        inputs=[weight, x, mx.array([in_size, out_size], dtype=mx.int32)],
        template=[("BM", bm), ("BN", bn), ("SM", sm), ("SN", sn), ("TM", tm), ("TN", tn)],
        grid=(groups * 32, bn, bm),
        threadgroup=(32, bn, bm),
        output_shapes=[(out_size,)],
        output_dtypes=[mx.float32],
    )[0]
