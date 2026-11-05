"""Types for the adapted MLX-LM Metal recurrence kernel."""

import mlx.core as mx

def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> list[mx.array]: ...
