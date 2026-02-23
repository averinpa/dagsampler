"""Top-level package for causal_simulator.

Provides the main public API surface for consumers of the package.
"""

from importlib.metadata import PackageNotFoundError, version

from .causal_sim import CausalDataGenerator
from .templates import (
    chain_config,
    collider_config,
    fork_config,
    indep_config,
    independence_config,
)

__all__ = [
    "CausalDataGenerator",
    "indep_config",
    "independence_config",
    "chain_config",
    "fork_config",
    "collider_config",
]

try:
    __version__ = version("dagsampler")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
