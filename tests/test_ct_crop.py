"""Recorded MONAI resampling and oblique geometry on HCC-TACE CT."""

from __future__ import annotations

import os
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
    default = Path(__file__).resolve().parents[1] / "work/real-reference"
    directory = Path(os.environ.get("MEDMLX_TEST_REFERENCES", str(default)))
    with np.load(fixture) as config, np.load(directory / fixture.name) as data:
        expected = cast(np.ndarray, data["expected"])
        actual = resample_ct(
            cast(np.ndarray, data["values"]), cast(np.ndarray, config["mapping"]), expected.shape
        )
    # CPU affine-grid rounding differs between the Linux reference and macOS.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.0012)


def test_oblique_anisotropic_resampling_obeys_geometry(tmp_path: Path) -> None:
    default = Path(__file__).resolve().parents[1] / "work/real-reference"
    directory = Path(os.environ.get("MEDMLX_TEST_REFERENCES", str(default)))
    with np.load(directory / "ct_resampling-0.npz") as data:
        values = cast(np.ndarray, data["values"])
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
    # Output index (4, 3, 8) samples input (8/3, 6/2.4, 4).
    expected = values[2, 2, 4] / 6 + values[3, 2, 4] / 3 + values[2, 3, 4] / 6 + values[3, 3, 4] / 3
    assert float(result.values[4, 3, 8]) == pytest.approx(float(expected), abs=0.001)
    np.testing.assert_allclose(result.affine[:3, 3], [12.0, -30.0, 4.0], atol=1e-6, rtol=0)
