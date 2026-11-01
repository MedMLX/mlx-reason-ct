"""Read scalar HU voxels with an unambiguous coded NIfTI transform."""

from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np

from mlx_reason_ct._host_types import Affine, FloatArray, Nibabel
from mlx_reason_ct.errors import InvalidInputError, MissingDependencyError


def load_ct(path: Path) -> tuple[FloatArray, Affine]:
    """Load one scalar 3D CT volume and its voxel-to-world affine.

    Args:
        path: NIfTI file (``.nii`` or ``.nii.gz``) holding Hounsfield-unit values.

    Returns:
        ``(values, affine)``. ``values`` is a finite float32 array in the file's
        storage axis order ``(i, j, k)``, with nibabel's slope/intercept scaling
        applied and no reorientation. ``affine`` is a float64 4x4 matrix mapping
        voxel indices to NIfTI RAS world coordinates in millimeters. The sform is
        used when coded, otherwise the qform. Unknown spatial units are assumed to
        be millimeters; the contents are expected to be HU but cannot be verified as
        such from the NIfTI metadata.

    Raises:
        MissingDependencyError: nibabel is not installed.
        InvalidInputError: the file is unreadable, has no coded qform or sform,
            has qform and sform that disagree (after unit scaling), has a
            non-finite, non-affine or singular transform, or is not a finite
            3D volume.
    """
    try:
        nib = cast(Nibabel, import_module("nibabel"))
    except ImportError as error:
        raise MissingDependencyError(
            "Reading CT NIfTI inputs requires nibabel; install mlx-reason-ct.",
        ) from error
    try:
        image = nib.load(str(path), mmap=True)
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
        raise InvalidInputError(f"Could not read NIfTI CT {path}: {error}") from error
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if not qcode and not scode:
        raise InvalidInputError("NIfTI image has no coded qform or sform spatial transform")
    # Normalize physical units before comparing forms and resampling in mm. The
    # scale applies to the affine's first three rows, translation included.
    scale = {"unknown": 1.0, "mm": 1.0, "meter": 1000.0, "micron": 0.001}[
        image.header.get_xyzt_units()[0]
    ]
    if (
        qcode
        and scode
        and not np.allclose(
            np.asarray(qform)[:3] * scale,
            np.asarray(sform)[:3] * scale,
            atol=1e-5,
            rtol=0,
        )
    ):
        raise InvalidInputError("NIfTI qform and sform encode different spatial grids")
    affine = np.array(sform if scode else qform, dtype=np.float64)
    affine[:3] *= scale
    if (
        not np.isfinite(affine).all()
        or not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or abs(np.linalg.det(affine[:3, :3])) < 1e-12
    ):
        raise InvalidInputError("NIfTI spatial transform must be finite and invertible")
    try:
        values = np.asarray(image.dataobj, dtype=np.float32)
    except (OSError, ValueError) as error:
        raise InvalidInputError(f"Could not read CT voxels from {path}: {error}") from error
    if values.ndim != 3 or not np.isfinite(values).all():
        raise InvalidInputError("CT input must be a finite scalar 3D HU volume")
    return values, affine
