"""
Diagnostics for Maxwell operators.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.covariant_derivatives import (
    curl_lapse_weighted_covector_curved,
)
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.tensors import lower_vector
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.core.state import State


def curl_E_curved(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Compute epsilon^{ijk}D_j E_k.
    """
    if state.E is None:
        return grid.zeros_vector()

    E_down = lower_vector(
        state.E,
        geom.gamma_down,
    )

    return curl_lapse_weighted_covector_curved(
        covector_down=E_down,
        grid=grid,
        geom=geom,
        order=order,
    ) / geom.lapse


def curl_B_curved(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Compute epsilon^{ijk}D_j B_k.
    """
    if state.B is None:
        return grid.zeros_vector()

    B_down = lower_vector(
        state.B,
        geom.gamma_down,
    )

    return curl_lapse_weighted_covector_curved(
        covector_down=B_down,
        grid=grid,
        geom=geom,
        order=order,
    ) / geom.lapse