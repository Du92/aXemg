"""
Basic tensor operations.

Phase 2 mostly uses flat Cartesian geometry, but we already introduce
index operations in a geometry-aware way.
"""

from __future__ import annotations

import numpy as np


def lower_vector(
    vector_up: np.ndarray,
    gamma_down: np.ndarray,
) -> np.ndarray:
    """
    Lower a spatial vector index:

        V_i = gamma_ij V^j

    Parameters
    ----------
    vector_up:
        Array with shape (3, ...).
    gamma_down:
        Spatial metric with shape (3, 3, ...).

    Returns
    -------
    vector_down:
        Array with shape (3, ...).
    """
    vector_down = np.zeros_like(vector_up)

    for i in range(3):
        for j in range(3):
            vector_down[i] += gamma_down[i, j] * vector_up[j]

    return vector_down


def raise_vector(
    vector_down: np.ndarray,
    gamma_up: np.ndarray,
) -> np.ndarray:
    """
    Raise a spatial vector index:

        V^i = gamma^ij V_j
    """
    vector_up = np.zeros_like(vector_down)

    for i in range(3):
        for j in range(3):
            vector_up[i] += gamma_up[i, j] * vector_down[j]

    return vector_up


def contract_cov_contra(
    covector_down: np.ndarray,
    vector_up: np.ndarray,
) -> np.ndarray:
    """
    Contract a covector with a vector:

        A_i V^i
    """
    result = np.zeros_like(vector_up[0])

    for i in range(3):
        result += covector_down[i] * vector_up[i]

    return result


def vector_norm_squared(
    vector_up: np.ndarray,
    gamma_down: np.ndarray,
) -> np.ndarray:
    """
    Compute V_i V^i.
    """
    vector_down = lower_vector(vector_up, gamma_down)
    return contract_cov_contra(vector_down, vector_up)
