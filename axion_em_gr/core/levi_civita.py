"""
Levi-Civita utilities.

We distinguish between:
    [ijk]             = antisymmetric symbol, values 0, +/-1
    epsilon^{ijk}    = [ijk] / sqrt(gamma)
    epsilon_{ijk}    = sqrt(gamma) [ijk]

The current code uses epsilon^{ijk} for spatial curls in the 3+1 Maxwell
equations.
"""

from __future__ import annotations

import numpy as np


def levi_civita_symbol_3d() -> np.ndarray:
    """
    Return the 3D Levi-Civita antisymmetric symbol [ijk].
    """
    eps = np.zeros((3, 3, 3), dtype=float)

    eps[0, 1, 2] = +1.0
    eps[1, 2, 0] = +1.0
    eps[2, 0, 1] = +1.0

    eps[0, 2, 1] = -1.0
    eps[2, 1, 0] = -1.0
    eps[1, 0, 2] = -1.0

    return eps


LEVI_CIVITA_SYMBOL_3D = levi_civita_symbol_3d()