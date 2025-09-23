"""
Public package facade for separatrix_locator.

This lightweight module re-exports the primary user-facing classes so that
`import separatrix_locator` works and users can access key APIs via
`separatrix_locator.SeparatrixLocator`, etc.
"""
from separatrix_locator.core.separatrix_locator import SeparatrixLocator

# Optional convenience exports (keep surface minimal and stable)
from separatrix_locator.dynamics.base import DynamicalSystem  # noqa: F401

__all__ = [
    "SeparatrixLocator",
    "DynamicalSystem",
]


