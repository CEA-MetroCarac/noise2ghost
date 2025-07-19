"""Noise2Ghost package.

Self-supervised deep convolutional reconstructions for ghost imaging.
"""

from __future__ import annotations

from noise2ghost.debug import get_version

__author__ = """Nicola Vigano & Mathieu Manni"""
__email__ = "nicola.vigano@cea.fr"
__version__ = get_version()

from . import algos
from . import models
from . import reconstructions
from . import testing

# __all__: list[str] = []
