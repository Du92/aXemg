"""
Flat spacetime in Cartesian coordinates.

N = 1
beta^i = 0
gamma_ij = delta_ij
gamma^ij = delta^ij
sqrt(gamma) = 1
K = 0
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.geometry.base_metric import BaseMetric, GeometryFields


class FlatMetric(BaseMetric):
    """
    Minkowski spacetime in Cartesian 3+1 form.
    """

    def evaluate(self, t: float, grid: Grid) -> GeometryFields:
        shape = grid.shape_full

        lapse = np.ones(shape, dtype=float)
        shift = np.zeros((3, *shape), dtype=float)

        gamma_down = np.zeros((3, 3, *shape), dtype=float)
        gamma_up = np.zeros((3, 3, *shape), dtype=float)

        for i in range(3):
            gamma_down[i, i] = 1.0
            gamma_up[i, i] = 1.0

        sqrt_gamma = np.ones(shape, dtype=float)
        K = np.zeros(shape, dtype=float)

        return GeometryFields(
            lapse=lapse,
            shift=shift,
            gamma_down=gamma_down,
            gamma_up=gamma_up,
            sqrt_gamma=sqrt_gamma,
            K=K,
        )
