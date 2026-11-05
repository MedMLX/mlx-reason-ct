# Copyright © 2025 Apple Inc.
"""Adapted MLX-LM Metal gated-delta recurrence, specialized for NV-Reason-CT decoding.

Only the scalar-gate, unmasked, ``Dk == 128`` FP32-gate/FP32-state case that the
decoder reaches is retained, as a packed kernel (eight value rows per
SIMD-group). On Metal it was bit-identical to MLX-LM's original simd_sum kernel
for the decoder shapes. Other shapes raise ``InvalidInputError``. The kernel
operates on already prepared arrays; it does not perform the model's query/key
normalization or gate calculation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_reason_ct._native_types import Array, MetalKernel

from mlx_reason_ct.errors import InvalidInputError
from mlx_reason_ct.runtime import import_mlx

mx = import_mlx()


def _make_gated_delta_packed_kernel() -> MetalKernel:
    """Make the scalar-gate Dk=128 recurrence kernel.

    MLX-LM's generic kernel assigns one 32-lane SIMD-group to each value row.
    For Dk=128 that leaves every lane with only four state elements and performs
    two full-SIMD reductions per row and token. This kernel instead packs
    eight independent value rows into a SIMD-group: four lanes own each row
    and each lane keeps 32 contiguous state elements in registers.

    The reduction is written out as the ascending butterfly (shuffle_xor
    1,2,4,8,16) rather than relying on how simd_sum lowers. The first three
    levels combine partials that live in a single packed lane (IEEE addition
    is commutative, so the local pairwise tree is bit-identical), and the
    last two levels map onto shuffle_xor(1) and shuffle_xor(2) within the
    four-lane row group. Each 4-element partial keeps the generic kernel's
    per-lane sequential order. On current Apple GPUs simd_sum lowers to the
    same tree, so outputs and state match the generic kernel bitwise.
    """

    source = r"""
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);

          // Partials mirror the generic kernel: each 4-element chain is one
          // original lane's sequential accumulation.
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          // Butterfly levels xor 1,2,4 stay inside this lane (commutative
          // pairwise tree); levels xor 8,16 become the row-group shuffles.
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_step_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


_gated_delta_kernel_packed = _make_gated_delta_packed_kernel()


def gated_delta_kernel(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Array,
) -> list[Array]:
    """Run gated-delta recurrence and return ``[outputs, final_state]``.

    ``q`` and ``k`` are ``(B, T, Hk, 128)``, ``v`` is ``(B, T, Hv, Dv)``, ``g``
    and ``beta`` are ``(B, T, Hv)``, and ``state`` is ``(B, Hv, Dv, 128)``.
    ``g`` and ``state`` must be FP32 and ``Dv`` a multiple of 8. Callers must
    also supply a positive ``Hk`` and ``Hv`` a multiple of ``Hk``; those are
    not checked here.

    Output shape is ``(B, T, Hv, Dv)`` with ``q``'s dtype; final state retains
    the input state's shape and dtype.

    Raises:
        InvalidInputError: the shapes or dtypes are outside the packed kernel.
    """
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    # Each lane owns Dk/4 = 32 state elements and a SIMD-group packs 32/4 value
    # rows, so the kernel needs Dk == 128 and Dv divisible by 8.
    if not (
        g.ndim == 3
        and Dk == 128
        and Dv % 8 == 0
        and g.dtype == mx.float32
        and state.dtype == mx.float32
    ):
        raise InvalidInputError(
            "gated_delta_kernel supports scalar FP32 gates, FP32 state, Dk == 128 "
            "and Dv divisible by 8"
        )
    return _gated_delta_kernel_packed(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv // 8, B * Hv),
        threadgroup=(32, 2, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
