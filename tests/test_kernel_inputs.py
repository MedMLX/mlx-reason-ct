"""Malformed model tensors must be rejected before raw Metal pointer reads."""

from typing import NoReturn

import pytest

from mlx_reason_ct import chunk_delta, gated_delta, source_arithmetic
from mlx_reason_ct.errors import InvalidInputError
from mlx_reason_ct.runtime import import_mlx

mx = import_mlx()


def refuse_dispatch(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("Malformed buffers reached Metal dispatch")


@pytest.mark.parametrize("case", ["query", "state", "beta-shape", "head-ratio"])
def test_packed_recurrence_rejects_incompatible_buffers(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gated_delta, "_gated_delta_kernel_packed", refuse_dispatch)
    q = mx.zeros((1, 1, 3 if case == "head-ratio" else 2, 128))
    k = mx.zeros(q.shape)
    v = mx.zeros((1, 1, 4, 8))
    gates = mx.full((1, 1, 4), 1.0)
    beta = gates
    state = mx.zeros((1, 4, 8, 128))
    if case == "query":
        q = q[:, :, :1]
    elif case == "state":
        state = state[..., :64]
    elif case == "beta-shape":
        beta = beta[..., :1]
    with pytest.raises(InvalidInputError):
        gated_delta.gated_delta_kernel(q, k, v, gates, beta, state)


@pytest.mark.parametrize(
    "case", ["projection", "cached-projection", "layernorm", "attention", "recurrence", "triangle"]
)
def test_source_kernels_reject_undersized_buffers(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(source_arithmetic, "kernel", refuse_dispatch)
    monkeypatch.setattr(chunk_delta, "_triangle_kernel", refuse_dispatch)
    with pytest.raises(InvalidInputError):
        if case == "projection":
            source_arithmetic.projection(mx.zeros((1, 2, 16)), mx.zeros((4, 8)))
        elif case == "cached-projection":
            source_arithmetic.cached_projection(mx.zeros((1, 1, 8)), mx.zeros((32, 2560)))
        elif case == "layernorm":
            source_arithmetic.layernorm(
                mx.zeros((1, 2, 864)), mx.full((864,), 1.0), mx.zeros((1,)), 1e-5
            )
        elif case == "attention":
            source_arithmetic.attention(
                mx.zeros((1, 2, 1, 72), dtype=mx.bfloat16),
                mx.zeros((1, 1, 3, 72), dtype=mx.bfloat16),
                mx.zeros((1, 1, 1, 72), dtype=mx.bfloat16),
                causal=False,
                offset=0,
            )
        elif case == "recurrence":
            source_arithmetic.gated_delta(
                mx.zeros((1, 1, 16, 128)),
                mx.zeros((1, 1, 16, 128)),
                mx.zeros((1, 1, 32, 128)),
                mx.zeros((1, 1, 1)),
                mx.zeros((1, 1, 32)),
                mx.zeros((1, 32, 128, 128)),
            )
        else:
            chunk_delta.triangle_fp32(mx.zeros((2, 64, 64), dtype=mx.bfloat16))


def test_packed_recurrence_matches_exact_two_step_solution() -> None:
    """Binary fractions give an independent exact recurrence expectation."""
    q = mx.full((1, 2, 2, 128), 0.125)
    output, state = gated_delta.gated_delta_kernel(
        q,
        q,
        mx.full((1, 2, 4, 8), 1.0),
        mx.full((1, 2, 4), 0.5),
        mx.full((1, 2, 4), 0.25),
        mx.zeros((1, 4, 8, 128)),
    )
    assert bool(mx.all(output[:, 0] == 0.5).item())
    assert bool(mx.all(output[:, 1] == 0.625).item())
    assert bool(mx.all(state == 0.0390625).item())
