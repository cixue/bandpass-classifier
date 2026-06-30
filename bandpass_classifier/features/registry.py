"""Feature registration module.

This module imports all feature modules to ensure that decorators decorate classes
and register functions correctly when the features package is loaded.
"""

from . import basic
from .scan_statistics import features as scan_statistics_features

