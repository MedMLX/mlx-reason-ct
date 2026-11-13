"""Reject unsupported CT data before coercion or voxel loading."""

from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from mlx_reason_ct._host_types import Nibabel
from mlx_reason_ct.errors import InvalidInputError
from mlx_reason_ct.nifti import load_ct

nib = cast(Nibabel, import_module("nibabel"))


def test_complex_voxels_are_not_silently_converted_to_hu(tmp_path: Path) -> None:
    path = tmp_path / "complex.nii"
    source = Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz"
    ct = np.asarray(nib.load(str(source)).dataobj, dtype=np.float32)
    values = ct[100:102, 210:212, 55:57].astype(np.complex64)
    image = nib.Nifti1Image(values + np.complex64(99j), np.eye(4))
    nib.save(image, path)
    with pytest.raises(InvalidInputError):
        load_ct(path)
