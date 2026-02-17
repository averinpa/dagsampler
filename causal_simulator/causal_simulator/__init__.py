"""Top-level package for causal_simulator.

Provides the main public API surface for consumers of the package.
"""

from importlib.metadata import PackageNotFoundError, version

# Public exports
from .causal_sim import CausalDataGenerator

__all__ = [
    "CausalDataGenerator",
]

# Package version (matches distribution name in pyproject.toml)
try:
    __version__ = version("simulation")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


