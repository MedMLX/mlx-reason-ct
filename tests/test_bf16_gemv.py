"""The BF16-weight decode kernel must stay bit-identical to MLX's FP32 matvec."""

from typing import NoReturn, cast

import numpy as np
import pytest

from mlx_reason_ct._host_types import FloatArray
from mlx_reason_ct.bf16_gemv import gemv
from mlx_reason_ct.errors import InvalidInputError
from mlx_reason_ct.runtime import import_mlx

mx = import_mlx()


@pytest.mark.parametrize("case", ["short-vector", "wrong-vector-dtype", "wrong-weight-dtype"])
def test_invalid_buffers_are_rejected_before_dispatch(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed projections must not reach raw Metal pointer reads."""
    from mlx_reason_ct import bf16_gemv

    def refuse_dispatch() -> NoReturn:
        raise AssertionError("Unsafe input reached the Metal kernel")

    monkeypatch.setattr(bf16_gemv, "_kernel", refuse_dispatch)
    x = mx.zeros((4 if case == "short-vector" else 8,), dtype=mx.float32)
    weight = mx.zeros((4, 8), dtype=mx.bfloat16)
    if case == "wrong-vector-dtype":
        x = x.astype(mx.bfloat16)
    if case == "wrong-weight-dtype":
        weight = weight.astype(mx.float32)
    with pytest.raises(InvalidInputError):
        gemv(x, weight)


@pytest.mark.parametrize(
    "in_size,out_size",
    [(2560, 8192), (2560, 32), (4096, 2560), (9216, 2560), (2560, 1024)],
    ids=["qkv", "gate-scalar", "out-proj", "down-proj", "kv-proj"],
)
def test_matches_float32_matvec_on_widened_weights(in_size: int, out_size: int) -> None:
    """Decoder shapes cover every MLX 0.32.4 tile configuration the model reaches."""
    rng = np.random.default_rng(in_size + out_size)
    x = mx.array(rng.standard_normal((1, 1, in_size), dtype=np.float32))
    weight = mx.array(rng.standard_normal((out_size, in_size), dtype=np.float32) * 0.02).astype(
        mx.bfloat16
    )
    expected = cast(FloatArray, np.array(x @ weight.astype(mx.float32).T)).reshape(out_size)
    actual = cast(FloatArray, np.array(gemv(mx.reshape(x, (in_size,)), weight)))
    np.testing.assert_array_equal(actual, expected)
