"""Recorded MONAI resampling and an independent oblique geometry expectation."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from mlx_reason_ct._host_types import Nibabel
from mlx_reason_ct.ct_crop import load_anatomy_ct
from mlx_reason_ct.ct_resample import resample_ct

nib = cast(Nibabel, import_module("nibabel"))


@pytest.mark.parametrize("case", [0, 1, 2, 3])
def test_resampling_matches_original_float32_processor_grid(case: int) -> None:
    fixture = Path(__file__).parent / "fixtures/nv_reason_ct" / f"ct_resampling-{case}.npz"
    with np.load(fixture) as data:
        expected = cast(np.ndarray, data["expected"])
        actual = resample_ct(
            cast(np.ndarray, data["values"]), cast(np.ndarray, data["mapping"]), expected.shape
        )
    np.testing.assert_array_equal(actual, expected)


def test_oblique_anisotropic_resampling_obeys_independent_linear_ramp(tmp_path: Path) -> None:
    x, y, z = np.indices((9, 7, 11))
    values = (x + 10 * y + 100 * z).astype(np.float32)
    angle = 0.17
    rotation = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([-3.0, -2.4, 4.0])
    affine[:3, 3] = (12.0, -30.0, 4.0)
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm")
    path = tmp_path / "oblique.nii.gz"
    nib.save(image, path)
    with pytest.warns(RuntimeWarning, match="superior-edge"):
        result = load_anatomy_ct(path, "abdomen", roi=(13, 8, 21))
    assert result.resampled_shape == (13, 8, 21)
    assert result.start == (0, 0, 0)
    np.testing.assert_allclose(
        result.affine[:3, :3], rotation @ np.diag([-2.0, -2.0, 2.0]), atol=1e-6, rtol=0
    )
    # Output index (4, 3, 8) samples input (8/3, 6/2.4, 16/4).
    assert float(result.values[4, 3, 8]) == pytest.approx(8 / 3 + 25 + 400, abs=0.0001)
    np.testing.assert_allclose(result.affine[:3, 3], [12.0, -30.0, 4.0], atol=1e-6, rtol=0)
