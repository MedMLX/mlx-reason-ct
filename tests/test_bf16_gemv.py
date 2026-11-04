"""The BF16-weight decode kernel must stay bit-identical to MLX's FP32 matvec."""

from typing import cast

import numpy as np
import pytest

from mlx_reason_ct._host_types import FloatArray
from mlx_reason_ct.bf16_gemv import gemv
from mlx_reason_ct.runtime import import_mlx

mx = import_mlx()


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
