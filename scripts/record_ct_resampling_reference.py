"""Record MONAI CT resampling outputs from an HCC-TACE scan."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.networks.layers import AffineTransform


def main() -> None:
    """Use the pinned CPU reference on fixed crops and voxel-coordinate maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ct", type=Path, default=Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz"
    )
    parser.add_argument("--output", type=Path, default=Path("work/real-reference"))
    args = parser.parse_args()
    expected_versions = {"torch": "2.13.0", "monai": "1.6.0", "numpy": "2.3.5"}
    for package, expected in expected_versions.items():
        if version(package) != expected:
            raise ValueError(f"{package} must be {expected}, got {version(package)}")
    torch.set_num_threads(1)
    ct = np.asarray(nib.load(args.ct).dataobj, dtype=np.float32)
    fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/nv_reason_ct"
    args.output.mkdir(parents=True, exist_ok=True)
    transform = AffineTransform(
        normalized=False, mode="bilinear", padding_mode="border", align_corners=False
    )
    for case in range(4):
        path = fixtures / f"ct_resampling-{case}.npz"
        with np.load(path, allow_pickle=False) as archive:
            shape = (
                tuple(archive["input_shape"].tolist())
                if "input_shape" in archive
                else archive["values"].shape
            )
            output_shape = (
                tuple(archive["output_shape"].tolist())
                if "output_shape" in archive
                else archive["expected"].shape
            )
            mapping = archive["mapping"]
        crop = ct[
            tuple(
                slice(start, start + size)
                for start, size in zip((100, 210, 55), shape, strict=True)
            )
        ]
        if crop.shape != shape:
            raise ValueError(f"CT crop {crop.shape} does not match {shape}")
        with torch.no_grad():
            expected = transform(
                torch.from_numpy(crop.copy())[None, None],
                torch.as_tensor(mapping[None], dtype=torch.float32),
                output_shape,
            )
        np.savez_compressed(
            args.output / path.name, values=crop, expected=expected.numpy().squeeze((0, 1))
        )
        np.savez_compressed(path, mapping=mapping, input_shape=shape, output_shape=output_shape)


if __name__ == "__main__":
    main()
