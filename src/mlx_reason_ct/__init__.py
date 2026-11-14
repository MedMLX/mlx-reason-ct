"""MLX-Reason-CT: CT reporting on Apple Silicon with lazy public exports."""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.6.0"

if TYPE_CHECKING:
    from mlx_reason_ct.medmlx import generate_report, load_model

__all__ = ["generate_report", "load_model"]


def __getattr__(name: str) -> object:
    """Import a public name on first access and cache it in the package."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.medmlx"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List loaded and lazily exported names."""
    return sorted({*globals(), *__all__})
