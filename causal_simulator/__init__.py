"""Compatibility shim for local development.

Allows `from causal_simulator import CausalDataGenerator` when running from the
repository root, where this top-level `causal_simulator/` folder would otherwise
shadow the installed package on sys.path.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("simulation")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .causal_simulator import CausalDataGenerator

__all__ = ["CausalDataGenerator"]


