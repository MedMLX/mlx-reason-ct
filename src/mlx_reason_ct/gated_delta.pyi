"""Types for the adapted MLX-LM Metal recurrence kernels."""

import mlx.core as mx

def gated_delta_kernel_xtree(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> list[mx.array]: ...
def gated_delta_kernel_unpacked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> list[mx.array]: ...
def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> list[mx.array]: ...
