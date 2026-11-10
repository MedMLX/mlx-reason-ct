"""Host array and installed dependency interfaces used by the source pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

type HostArray = NDArray[np.generic]
type FloatArray = NDArray[np.float32]
type Affine = NDArray[np.float64]


class NiftiHeader(Protocol):
    def get_data_shape(self) -> tuple[int, ...]: ...
    def get_data_dtype(self) -> np.dtype[np.generic]: ...
    def get_xyzt_units(self) -> tuple[str, str]: ...
    def set_xyzt_units(self, xyz: str) -> None: ...


class ArrayBuffer(Protocol):
    def __array__(
        self,
        dtype: np.dtype[np.generic] | None = None,
        copy: bool | None = None,
    ) -> HostArray: ...


class NiftiImage(Protocol):
    header: NiftiHeader
    dataobj: HostArray | ArrayBuffer

    def get_qform(self, *, coded: Literal[True]) -> tuple[Affine | None, int]: ...
    def get_sform(self, *, coded: Literal[True]) -> tuple[Affine | None, int]: ...


class Orientations(Protocol):
    def ornt_transform(self, start_ornt: Affine, end_ornt: Affine) -> Affine: ...
    def io_orientation(self, affine: Affine) -> Affine: ...
    def axcodes2ornt(self, axcodes: Sequence[str]) -> Affine: ...
    def inv_ornt_aff(self, ornt: Affine, shape: tuple[int, ...]) -> Affine: ...
    def apply_orientation[T: np.generic](self, arr: NDArray[T], ornt: Affine) -> NDArray[T]: ...


class FileImages(Protocol):
    ImageFileError: type[Exception]


class Nibabel(Protocol):
    orientations: Orientations
    filebasedimages: FileImages

    def load(self, filename: str, *, mmap: bool = True) -> NiftiImage: ...
    def Nifti1Image(self, dataobj: HostArray, affine: HostArray) -> NiftiImage: ...
    def save(self, image: NiftiImage, filename: str | Path) -> None: ...


class Ndimage(Protocol):
    def binary_fill_holes(self, input: NDArray[np.bool_]) -> NDArray[np.bool_]: ...
    def label(self, input: NDArray[np.bool_]) -> tuple[NDArray[np.int32], int]: ...


class Encoding(Protocol):
    @property
    def ids(self) -> list[int]: ...


class TokenizerContract(Protocol):
    def no_padding(self) -> None: ...
    def no_truncation(self) -> None: ...
    def token_to_id(self, token: str) -> int | None: ...
    def encode(self, sequence: str, *, add_special_tokens: bool) -> Encoding: ...
    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str: ...


class TokenizerFactory(Protocol):
    def from_file(self, path: str) -> TokenizerContract: ...


class Tokenizers(Protocol):
    Tokenizer: TokenizerFactory
