"""MONAI 1.6 / PyTorch 2.13 CPU float32 CT resampling, without Torch.

Keep normalized affine-grid rounding and border-clamped trilinear corner order.
Direct interpolation in float64 voxel coordinates changes BF16 input rounding.
Source operations are recorded in docs/development.md; licenses in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import numpy as np


def _multiply4(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Keep the four-term FP32 stores of the source's small affine products."""
    result = np.zeros((4, 4), dtype=np.float32)
    for k in range(4):
        result = result + left[:, k, None] * right[k]
    return result


def _normalize(shape: tuple[int, ...]) -> np.ndarray:
    """4x4 FP32 map from voxel-center indices to normalized [-1, 1] (align_corners=False)."""

    sizes = np.asarray(shape, dtype=np.float64)
    result = np.diag(np.concatenate((2 / sizes, [1])))
    result[:3, 3] = 1 / sizes - 1
    return result.astype(np.float32)


def _axis(length: int) -> np.ndarray:
    """Normalized FP32 grid coordinates of one output axis, rounded as ``affine_grid`` does."""

    if length == 1:
        return np.zeros(1, dtype=np.float32)
    # ATen linspace uses a rounded FP32 step and fused endpoint arithmetic,
    # starting from each end. affine_grid then multiplies and divides separately.
    step = float(np.float32(2 / (length - 1)))
    indices = np.arange(length, dtype=np.float64)
    values = np.where(
        indices < length // 2, -1 + indices * step, 1 - (length - indices - 1) * step
    ).astype(np.float32)
    return (values * np.float32(length - 1)) / np.float32(length)


def _sample(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Trilinearly sample ``values`` at normalized ``grid`` points ``(..., 3)``; border-clamped."""

    coordinates = [
        np.clip(((grid[..., axis] + 1) * size - 1) / 2, 0, size - 1)
        for axis, size in enumerate(values.shape)
    ]
    lower = [np.floor(c).astype(np.intp) for c in coordinates]
    upper = [np.minimum(lo + 1, size - 1) for lo, size in zip(lower, values.shape, strict=True)]
    fractions = [c - lo.astype(np.float32) for c, lo in zip(coordinates, lower, strict=True)]
    result = np.zeros(grid.shape[:-1], dtype=np.float32)
    # Source W corresponds to array axis 2: its weight and corner vary first.
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                bits = (ix, iy, iz)
                indices = tuple(upper[a] if bits[a] else lower[a] for a in range(3))
                weights = [fractions[a] if bits[a] else 1 - fractions[a] for a in range(3)]
                weight = (weights[2] * weights[1]) * weights[0]
                result += values[indices] * weight
    return result


def resample_ct(values: np.ndarray, mapping: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sample a finite XYZ FP32 HU volume using the source output-to-input map.

    ``mapping`` is a 4x4 matrix mapping output voxel-center indices to input
    voxel-center indices. ``shape`` is the output ``(X, Y, Z)`` shape; the result is
    a float32 array of that shape. Normalization uses align_corners=False;
    out-of-bounds locations clamp to the input border.
    Eight output planes bound temporary coordinate and interpolation storage.
    """
    theta = _multiply4(
        _multiply4(_normalize(values.shape), mapping.astype(np.float32)),
        np.linalg.inv(_normalize(shape)),
    )[:3]
    axes = [_axis(n) for n in shape]
    result = np.empty(shape, dtype=np.float32)
    small_grid = int(np.prod(shape)) * 12 < 400
    for start in range(0, shape[0], 8):
        x = axes[0][start : start + 8, None, None, None]
        y = axes[1][None, :, None, None]
        z = axes[2][None, None, :, None]
        if small_grid:
            # The source's small bmm kernel stores each scalar product and sum.
            grid = ((z * theta[:, 2] + y * theta[:, 1]) + x * theta[:, 0]) + theta[:, 3]
        else:
            # The pinned CPU K4 grid product carries terms (0,2) and (1,3).
            # W/Y/X indexing makes Z the first product and X its fused addend.
            carry = z * theta[:, 2]
            even = (
                x.astype(np.float64) * theta[:, 0].astype(np.float64) + carry.astype(np.float64)
            ).astype(np.float32)
            odd = y * theta[:, 1] + theta[:, 3]
            grid = even + odd
        result[start : start + 8] = _sample(values, grid)
    return result
