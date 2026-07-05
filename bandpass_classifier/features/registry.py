"""Central feature registration module.

This module imports all feature modules to ensure that decorators decorate classes
and register functions correctly when the features package is loaded.

To add a new feature set, implement the related feature construction process in a
separate module and import it here.
"""

from . import basic_features
from .scan_statistics import features

