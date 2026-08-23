"""Feature extraction package for the bandpass classifier.

This sub-package contains registry setups, basic feature calculations,
and specialized scan statistics calculations. Importing this package boots up
the extractor registry automatically.
"""

# This single explicit import boots up the registry,
# which triggers all downstream decorators.
from . import registry  # noqa: F401
from .extractor import Extractor

__all__ = ["Extractor"]
