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
    image = nib.Nifti1Image(np.full((2, 2, 2), 1 + 99j, dtype=np.complex64), np.eye(4))
    nib.save(image, path)
    with pytest.raises(InvalidInputError):
        load_ct(path)
