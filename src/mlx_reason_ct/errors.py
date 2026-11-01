"""Shared MedMLX errors at the input, asset and execution boundaries."""

from typing import cast

from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)

__all__ = [
    "AssetNotReadyError",
    "InvalidInputError",
    "InvalidPromptError",
    "MissingDependencyError",
    "ModelExecutionError",
]

# The pinned core has no typing marker; its InvalidInputError inherits ValueError.
_PromptErrorBase = cast(type[ValueError], InvalidInputError)


class InvalidPromptError(_PromptErrorBase):
    """A prompt cannot represent exactly one CT volume."""
