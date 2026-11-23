"""Read scalar HU voxels with an unambiguous coded NIfTI transform."""

from pathlib import Path

import numpy as np
from medmlx_core.volumes import read_volume

from mlx_reason_ct._host_types import Affine, FloatArray
from mlx_reason_ct.errors import InvalidInputError


def load_ct(path: Path) -> tuple[FloatArray, Affine]:
    """Load one scalar 3D CT volume and its voxel-to-world affine.

    Args:
        path: NIfTI file (``.nii`` or ``.nii.gz``) holding Hounsfield-unit values.

    Returns:
        ``(values, affine)``. ``values`` is a finite float32 array in the file's
        storage axis order ``(i, j, k)``, with the header's slope/intercept scaling
        applied and no reorientation. ``affine`` is a float64 4x4 matrix mapping
        voxel indices to NIfTI RAS world coordinates in millimeters. The sform is
        used when coded, otherwise the qform. Meter and micron headers are converted to
        millimeters and unspecified units are taken as millimeters. The contents are
        expected to be HU but cannot be verified as such from the NIfTI metadata.

    Raises:
        MissingDependencyError: nibabel is not installed.
        InvalidInputError: the file is unreadable, has no coded qform or sform,
            has qform and sform that disagree, has a non-finite, non-affine or singular
            transform or a spatial unit other than unspecified, meter, millimeter or
            micron, or is not a finite nonempty 3D volume with real numeric voxels.
    """
    volume = read_volume(path, role="CT input")
    if not volume.qform_code and not volume.sform_code:
        raise InvalidInputError("NIfTI image has no coded qform or sform spatial transform")
    if (
        volume.qform_code
        and volume.sform_code
        and not np.allclose(volume.qform[:3], volume.sform[:3], atol=1e-5, rtol=0)
    ):
        raise InvalidInputError("NIfTI qform and sform encode different spatial grids")
    return volume.data, volume.affine
