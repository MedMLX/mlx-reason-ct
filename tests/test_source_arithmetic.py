"""Metal arithmetic against analytic references recorded on macOS arm64."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from mlx_reason_ct._host_types import FloatArray
from mlx_reason_ct._payloads import KernelSpec
from mlx_reason_ct.errors import MissingDependencyError
from mlx_reason_ct.runtime import import_mlx

try:
    import_mlx()
except MissingDependencyError as error:
    pytest.skip(str(error), allow_module_level=True)

if TYPE_CHECKING:
    from mlx_reason_ct._native_types import Array


class DecayReference(TypedDict):
    input: list[float]
    expected: list[float]


class LayerNormReference(TypedDict):
    input: list[list[list[float]]]
    weight: list[float]
    bias: list[float]
    eps: float
    float32: list[list[list[float]]]
    bfloat16: list[list[list[float]]]


class ActivationReference(TypedDict):
    input: list[float]
    sigmoid_bf16: list[float]
    silu_bf16: list[float]
    silu_fp32: list[float]


class MeanSquareReference(TypedDict):
    seed: int
    expected: list[list[list[float]]]


class ProjectionReference(TypedDict):
    input: list[list[list[float]]]
    weight: list[list[float]]
    bias: list[float]
    float32: list[list[list[float]]]
    rounded_operands: list[list[list[float]]]
    bfloat16: list[list[list[float]]]
    error_scale: list[list[list[float]]]
    rounded_error_scale: list[list[list[float]]]


class ArithmeticReference(TypedDict):
    schema_version: int
    recorded_on: dict[str, str]
    decay: DecayReference
    layer_norm: LayerNormReference
    activation: ActivationReference
    mean_square: dict[str, MeanSquareReference]
    projection: ProjectionReference


@pytest.mark.parametrize("matrices", [1, 32, 513])
def test_native_triangle_preserves_mlx_reduction_order(matrices: int) -> None:
    from mlx_reason_ct.chunk_delta import mx, triangle_fp32

    rng = np.random.default_rng(431)
    values = np.tril(rng.normal(size=(matrices, 64, 64)).astype(np.float32) * 0.05, -1)
    values[0, 1, 0] = -0.0
    expected = mx.array(values)
    # The unfused expression is the numerical oracle, including its change of
    # reduction tree at row 32 and separate product rounding before summation.
    for i in range(1, 64):
        row = expected[..., i, :i]
        expected[..., i, :i] = row + mx.sum(row[..., :, None] * expected[..., :i, :i], axis=-2)
        mx.eval(expected)
    np.testing.assert_array_equal(
        cast(FloatArray, np.array(triangle_fp32(mx.array(values)))).view(np.uint32),
        cast(FloatArray, np.array(expected)).view(np.uint32),
    )


@pytest.mark.parametrize("length", [1, 63, 64, 65, 129])
def test_native_chunk_delta_matches_independent_recurrence_and_continuation(length: int) -> None:
    from mlx_reason_ct.chunk_delta import chunk_delta, mx

    rng = np.random.default_rng(431)
    q = rng.normal(size=(1, length, 2, 8)).astype(np.float32) * 0.1
    k = rng.normal(size=q.shape).astype(np.float32) * 0.1
    v = rng.normal(size=(1, length, 4, 6)).astype(np.float32)
    g = -rng.uniform(0.001, 0.2, size=(1, length, 4)).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, size=g.shape).astype(np.float32)
    initial = rng.normal(size=(1, 4, 6, 8)).astype(np.float32) * 0.1
    # Independent scalar time recurrence in double precision, including a
    # non-square V,K state and nonzero prior state to catch axis/cache errors.
    expected_state = initial.astype(np.float64).transpose(0, 1, 3, 2).copy()
    expected: list[NDArray[np.float64]] = []
    for t in range(length):
        qt = np.repeat(q[:, t], 2, axis=1).astype(np.float64)
        kt = np.repeat(k[:, t], 2, axis=1).astype(np.float64)
        expected_state *= np.exp(g[:, t].astype(np.float64))[..., None, None]
        delta = (v[:, t] - np.sum(expected_state * kt[..., None], axis=-2)) * beta[:, t, :, None]
        expected_state += kt[..., None] * delta[..., None, :]
        expected.append(np.sum(expected_state * qt[..., None], axis=-2))
    actual, state = chunk_delta(*map(mx.array, (q, k, v, g, beta, initial)))
    np.testing.assert_allclose(
        cast(FloatArray, np.array(actual)), np.stack(expected, axis=1), atol=2e-6, rtol=2e-5
    )
    np.testing.assert_allclose(
        cast(FloatArray, np.array(state)),
        expected_state.transpose(0, 1, 3, 2),
        atol=2e-6,
        rtol=2e-5,
    )
    if length > 1:
        split = min(62, length - 1)
        first, prior = chunk_delta(
            mx.array(q[:, :split]),
            mx.array(k[:, :split]),
            mx.array(v[:, :split]),
            mx.array(g[:, :split]),
            mx.array(beta[:, :split]),
            mx.array(initial),
        )
        rest, continued = chunk_delta(
            mx.array(q[:, split:]),
            mx.array(k[:, split:]),
            mx.array(v[:, split:]),
            mx.array(g[:, split:]),
            mx.array(beta[:, split:]),
            prior,
        )
        np.testing.assert_allclose(
            cast(FloatArray, np.array(mx.concatenate([first, rest], axis=1))),
            np.stack(expected, axis=1),
            atol=2e-6,
            rtol=2e-5,
        )
        np.testing.assert_allclose(
            cast(FloatArray, np.array(continued)),
            expected_state.transpose(0, 1, 3, 2),
            atol=2e-6,
            rtol=2e-5,
        )


@pytest.fixture(scope="module")
def reference() -> ArithmeticReference:
    path = Path(__file__).parent / "fixtures/nv_reason_ct/arithmetic.json"
    recorded = cast(ArithmeticReference, json.loads(path.read_text()))
    assert recorded["schema_version"] == 1
    assert recorded["recorded_on"]["system"] == "Darwin"
    assert recorded["recorded_on"]["machine"] == "arm64"
    return recorded


def _gamma(operations: int) -> float:
    """Conservative FP32 accumulation bound for the seeded, well-conditioned inputs."""
    unit = float(np.finfo(np.float32).eps) / 2
    return operations * unit / (1 - operations * unit)


@pytest.mark.parametrize("rows", [1, 513])
def test_native_decay_scan_matches_mac_reference(reference: ArithmeticReference, rows: int) -> None:
    from mlx_reason_ct.chunk_delta import mx, source_cumsum

    case = reference["decay"]
    shape = (1, rows, 1, 64)
    values = mx.broadcast_to(mx.array(case["input"], dtype=mx.float32), shape)
    expected = np.broadcast_to(np.array(case["expected"]), shape)
    np.testing.assert_allclose(
        cast(FloatArray, np.array(source_cumsum(values))), expected, rtol=_gamma(64), atol=0
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_native_layer_norm_respects_precision_and_affine_math(
    reference: ArithmeticReference, mixed: bool
) -> None:
    from mlx_reason_ct.mlx_model import NativeModel, mx

    case = reference["layer_norm"]
    model = object.__new__(NativeModel)
    model.accumulate_float32 = mixed
    model.weights = {
        "norm.weight": mx.array(case["weight"], dtype=mx.bfloat16),
        "norm.bias": mx.array(case["bias"], dtype=mx.bfloat16),
    }
    result = model.norm(mx.array(case["input"], dtype=mx.bfloat16), "norm", case["eps"])
    assert result.dtype == (mx.float32 if mixed else mx.bfloat16)
    expected = case["float32" if mixed else "bfloat16"]
    # Mean, variance, normalization and affine operations each contribute error;
    # the BF16 result additionally permits one storage ULP, not foreign bit patterns.
    bound = 4 * _gamma(len(case["weight"]))
    np.testing.assert_allclose(
        cast(FloatArray, np.array(result.astype(mx.float32))),
        expected,
        atol=bound,
        rtol=bound + (0 if mixed else 2**-7),
    )


@pytest.mark.parametrize("operation", ["sigmoid_bf16", "silu_bf16", "silu_fp32"])
def test_native_activation_matches_finite_scalar_formula(
    reference: ArithmeticReference, operation: Literal["sigmoid_bf16", "silu_bf16", "silu_fp32"]
) -> None:
    from mlx_reason_ct import source_arithmetic
    from mlx_reason_ct.mlx_model import NativeModel, mx

    case = reference["activation"]
    values = mx.array(case["input"], dtype=mx.float32 if operation == "silu_fp32" else mx.bfloat16)
    result = (
        NativeModel.sigmoid(values)
        if operation == "sigmoid_bf16"
        else NativeModel.silu(values)
        if operation == "silu_bf16"
        else source_arithmetic.silu(values)
    )
    assert result.dtype == values.dtype
    actual = cast(FloatArray, np.array(result.astype(mx.float32)))
    # One BF16 storage ULP, or eight FP32 epsilons for exp/divide approximations.
    relative = 8 * float(np.finfo(np.float32).eps) if operation == "silu_fp32" else 2**-7
    np.testing.assert_allclose(actual, case[operation], rtol=relative, atol=0)
    assert actual[1] == (0.5 if operation == "sigmoid_bf16" else 0.0)


@pytest.mark.parametrize("width", [128, 256, 2560])
@pytest.mark.parametrize("rows", [1, 2, 17])
def test_native_rms_mean_matches_mac_reference(
    reference: ArithmeticReference, width: int, rows: int
) -> None:
    from mlx_reason_ct import source_arithmetic
    from mlx_reason_ct.mlx_model import mx

    case = reference["mean_square"][f"w{width}-r{rows}"]
    rng = np.random.Generator(np.random.PCG64(case["seed"]))
    raw = rng.normal(0, 0.3, (1, rows, width)).astype(np.float32)
    values = mx.array(raw, dtype=mx.bfloat16).astype(mx.float32)
    actual = cast(FloatArray, np.array(source_arithmetic.mean_square(values * values)))
    np.testing.assert_allclose(actual, case["expected"], rtol=_gamma(width + 1), atol=0)


@pytest.mark.parametrize("tower", ["vision3d", "language_model"])
@pytest.mark.parametrize("precision", ["float32", "bfloat16", "source_bfloat16"])
def test_native_projection_respects_profile_operands_and_output_dtype(
    reference: ArithmeticReference, tower: str, precision: str
) -> None:
    from mlx_reason_ct.mlx_model import NativeModel, mx

    case = reference["projection"]
    model = object.__new__(NativeModel)
    model.precision = precision
    model.accumulate_float32 = precision == "bfloat16"
    model.dtype = mx.float32 if precision == "float32" else mx.bfloat16
    name = (
        "model.vision3d.projection"
        if tower == "vision3d"
        else "model.language_model.layers.0.self_attn.q_proj"
    )
    model.weights = {
        name + ".weight": mx.array(case["weight"], dtype=model.dtype),
        name + ".bias": mx.array(case["bias"], dtype=model.dtype),
    }
    original = precision == "source_bfloat16"
    result = model.linear(
        mx.array(case["input"], dtype=mx.bfloat16 if original else mx.float32), name
    )
    assert result.dtype == (mx.bfloat16 if original else mx.float32)
    rounded = original or (tower == "vision3d" and precision == "bfloat16")
    expected = np.array(
        case["bfloat16" if original else "rounded_operands" if rounded else "float32"]
    )
    scale = np.array(case["rounded_error_scale" if rounded else "error_scale"])
    # The absolute-product sum bounds a 32-term dot plus its bias; source BF16
    # additionally rounds the output once. No hardware dispatch or bitwise oracle.
    bound = _gamma(34) * scale + (np.abs(expected) * 2**-7 if original else 0)
    np.testing.assert_array_less(
        np.abs(cast(FloatArray, np.array(result.astype(mx.float32))) - expected), bound
    )


@pytest.mark.parametrize("rows,sliced", [(1, False), (2, False), (4, False), (4, True)])
@pytest.mark.parametrize("activation", ["bfloat16", "float32"])
def test_native_projection_fast_alignment_matches_reference_blocks(
    rows: int, sliced: bool, activation: str
) -> None:
    from mlx_reason_ct import source_arithmetic

    mx = source_arithmetic.mx
    rng = np.random.default_rng(5)

    def operand(shape: tuple[int, ...]) -> Array:
        # Wide exponents, zeros and coarse values reach both sides of the fast
        # alignment window, truncation carries and exact cancellation.
        values = rng.normal(size=shape) * np.exp2(rng.integers(-64, 64, size=shape))
        values[rng.random(shape) < 0.1] = 0
        coarse = rng.random(shape) < 0.3
        values[coarse] = np.round(values[coarse] * 4) / 4
        return mx.array(values.astype(np.float32))

    m, n, k = 37, 70, 192
    x = operand((m, k)).astype(mx.bfloat16 if activation == "bfloat16" else mx.float32)
    weight = operand((n, k)).astype(mx.bfloat16)
    carry, bias = operand((m, n)), operand((n,))
    outputs: list[Array] = []
    for reference in (0, 1):
        outputs.append(
            source_arithmetic.kernel("projection")(
                inputs=[x, weight, carry, bias, mx.array([reference], dtype=mx.uint32)],
                template=[
                    ("M", m),
                    ("N", n),
                    ("K", k),
                    ("Ldx", k),
                    ("Ldw", k),
                    ("KOffset", 0),
                    ("Rows", rows),
                    ("Sliced", sliced),
                    ("HasC", True),
                    ("HasBias", True),
                ],
                grid=(-(-m // rows) * n, 1, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(m, n)],
                output_dtypes=[mx.float32],
            )[0]
        )
    fast, reference_blocks = (
        cast(FloatArray, np.array(value)).view(np.uint32) for value in outputs
    )
    np.testing.assert_array_equal(fast, reference_blocks)


def test_native_projection_batches_preserve_partitioned_outputs() -> None:
    from mlx_reason_ct import source_arithmetic

    mx = source_arithmetic.mx
    rng = np.random.default_rng(23)
    m, n, k = 513, 530, 192
    x = mx.array(rng.normal(size=(1, m, k)).astype(np.float32))
    weight = mx.array(rng.normal(size=(n, k)).astype(np.float32)).astype(mx.bfloat16)
    bias = mx.array(rng.normal(size=(n,)).astype(np.float32))
    # One unbatched reference launch per source partition checks the wrapper's
    # row slicing, ragged final batch, bias and inter-partition BF16 stores.
    reference = mx.zeros((1, m, n), dtype=mx.float32)
    for offset in range(0, k, 64):
        reference = source_arithmetic.kernel("projection")(
            inputs=[x, weight, reference, bias, mx.array([1], dtype=mx.uint32)],
            template=[
                ("M", m),
                ("N", n),
                ("K", 64),
                ("Ldx", k),
                ("Ldw", k),
                ("KOffset", offset),
                ("Rows", 1),
                ("Sliced", True),
                ("HasC", offset > 0),
                ("HasBias", True),
            ],
            grid=(m * n, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(1, m, n)],
            output_dtypes=[mx.float32],
        )[0]
    actual = source_arithmetic.projection(x, weight, bias=bias, slices=3, sliced=True)
    np.testing.assert_array_equal(
        cast(FloatArray, np.array(actual.astype(mx.float32))).view(np.uint32),
        cast(FloatArray, np.array(reference.astype(mx.bfloat16).astype(mx.float32))).view(
            np.uint32
        ),
    )


@pytest.mark.parametrize("width,heads,kv_heads,causal", [(72, 3, 3, False), (256, 4, 1, True)])
def test_native_attention_preserves_uniform_masked_value_means(
    width: int, heads: int, kv_heads: int, causal: bool
) -> None:
    from mlx_reason_ct import source_arithmetic

    mx = source_arithmetic.mx
    query_length, key_length, offset = 129, 137, 8
    channels = np.arange(width)[None, None, :]
    keys = np.arange(key_length)[None, :, None]
    groups = np.arange(kv_heads)[:, None, None]
    values = (((channels + keys * 3 + groups * 5) % 17 - 8) / 8).astype(np.float32)
    # Zero Q/K makes visible probabilities uniform. Dyadic values sum exactly;
    # their masked means independently check key, channel and grouped-head mapping
    # across ragged key tiles and multiple query launches.
    expected = np.empty((1, heads, query_length, width), dtype=np.float32)
    for head in range(heads):
        for row in range(query_length):
            count = min(key_length, offset + row + 1) if causal else key_length
            expected[0, head, row] = values[head // (heads // kv_heads), :count].mean(
                axis=0, dtype=np.float64
            )
    bits = expected.view(np.uint32)
    expected = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(
        np.float32
    )
    actual = source_arithmetic.attention(
        mx.zeros((1, heads, query_length, width), dtype=mx.bfloat16),
        mx.zeros((1, kv_heads, key_length, width), dtype=mx.bfloat16),
        mx.array(values[None]).astype(mx.bfloat16),
        causal=causal,
        offset=offset,
    )
    assert actual.dtype == mx.bfloat16
    np.testing.assert_array_equal(cast(FloatArray, np.array(actual.astype(mx.float32))), expected)


def test_native_attention_k8_preserves_reference_carries() -> None:
    from mlx_reason_ct import source_arithmetic

    mx = source_arithmetic.mx
    specs = cast(
        dict[str, KernelSpec], json.loads((source_arithmetic.ASSETS / "kernels.json").read_text())
    )
    rng = np.random.default_rng(31)
    count = 4096
    operands = rng.normal(size=(2, count, 8)) * np.exp2(rng.integers(-126, 127, (2, count, 8)))
    # Attention consumes BF16 operands, including probabilities constructed in
    # the kernel. Truncating the host bits retains BF16 subnormals on upload.
    rounded = operands.astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)
    x, weight = rounded.view(np.float32)
    carry = (rng.normal(size=count) * np.exp2(rng.integers(-126, 127, count))).astype(np.float32)
    x[:8], weight[:8], carry[:8] = 0, 0, 0
    carry[0], x[0, :3], weight[0, :3] = 1, 2**-12, 2**-12
    carry[1] = np.nextafter(np.float32(-(2**-100)), np.float32(0))
    x[1, 0], weight[1, 0] = 2**-50, 2**-50
    x[2], weight[2] = 2**63, 2**62
    x[3, 0], weight[3, 0] = np.inf, 2**-100
    x[4, 0], weight[4, 0] = np.nan, 2**-100
    x[5, 0], weight[5, 0] = 2**-133, 2**100
    # A normal carry exposes a missed subnormal guard inside the fast window.
    carry[5] = 2**-100
    carry[6] = -np.inf
    carry[7] = np.nan
    outputs: list[FloatArray] = []
    # Projection retains the original bit-shift block as its slow path in the
    # package. Compare that independent implementation with attention's local guard.
    for owner in ("projection", "attention"):
        block = mx.fast.metal_kernel(
            name="test_k8_" + owner,
            input_names=["X", "W", "C"],
            output_names=["Y"],
            header=specs[owner]["header"],
            source="""
                uint row = thread_position_in_grid.x;
                if (row >= Count) return;
                float x[8], w[8];
                for (int k = 0; k < 8; ++k) {
                    x[k] = X[row * 8 + k]; w[k] = W[row * 8 + k];
                }
                Y[row] = nv_block(C[row], x, w);
            """,
        )
        value = block(
            inputs=[mx.array(x), mx.array(weight), mx.array(carry)],
            template=[("Count", count)],
            grid=(count, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(count,)],
            output_dtypes=[mx.float32],
        )[0]
        outputs.append(cast(FloatArray, np.array(value)))
    np.testing.assert_array_equal(outputs[0].view(np.uint32), outputs[1].view(np.uint32))
    # Analytic anchors distinguish truncated carries from ordinary RNE sums,
    # cancellation at the lower fast-window edge, and overflow at the upper edge.
    np.testing.assert_array_equal(outputs[1][:3], [1 + 2**-23, 2**-124, np.inf])
