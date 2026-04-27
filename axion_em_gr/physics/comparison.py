"""
Comparison utilities for pairs of evolutions.

These functions are useful for comparing:
- flat background vs GW background,
- weak coupling vs strong coupling,
- different metric amplitudes,
- different axion masses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State
from axion_em_gr.core.tensors import contract_cov_contra, lower_vector
from axion_em_gr.geometry.base_metric import GeometryFields


@dataclass
class DifferenceNorms:
    """
    Norms of field differences between two states.
    """

    time: float

    max_delta_a: float
    l2_delta_a: float
    rel_l2_delta_a: float

    max_delta_Pi: float
    l2_delta_Pi: float
    rel_l2_delta_Pi: float

    max_delta_E: float
    l2_delta_E: float
    rel_l2_delta_E: float

    max_delta_B: float
    l2_delta_B: float
    rel_l2_delta_B: float

    max_delta_EdotB: float
    l2_delta_EdotB: float
    rel_l2_delta_EdotB: float


def _l2_scalar(field: np.ndarray, grid: Grid) -> float:
    interior = grid.interior_slices
    dx = grid.dx[0]
    return float(np.sqrt(np.sum(field[interior] ** 2) * dx))


def _l2_vector(field: np.ndarray, grid: Grid) -> float:
    interior = grid.interior_slices
    dx = grid.dx[0]
    data = field[(slice(None), *interior)]
    return float(np.sqrt(np.sum(data**2) * dx))


def _max_scalar(field: np.ndarray, grid: Grid) -> float:
    interior = grid.interior_slices
    return float(np.max(np.abs(field[interior])))


def _max_vector(field: np.ndarray, grid: Grid) -> float:
    interior = grid.interior_slices
    data = field[(slice(None), *interior)]
    return float(np.max(np.sqrt(np.sum(data**2, axis=0))))


def _relative(numerator: float, denominator: float, eps: float = 1.0e-14) -> float:
    return float(numerator / max(denominator, eps))


def edotb_profile(
    state: State,
    geom: GeometryFields,
) -> np.ndarray:
    """
    Compute E_i B^i.
    """
    if state.E is None or state.B is None:
        return np.zeros_like(state.a)

    E_down = lower_vector(state.E, geom.gamma_down)
    return contract_cov_contra(E_down, state.B)


def difference_norms(
    state_reference: State,
    state_test: State,
    geom_reference: GeometryFields,
    geom_test: GeometryFields,
    grid: Grid,
    time: float,
) -> DifferenceNorms:
    """
    Compare a reference state against a test state.

    Usually:
        reference = flat
        test      = GW
    """
    delta_a = state_test.a - state_reference.a
    delta_Pi = state_test.Pi - state_reference.Pi

    max_delta_a = _max_scalar(delta_a, grid)
    l2_delta_a = _l2_scalar(delta_a, grid)
    rel_l2_delta_a = _relative(l2_delta_a, _l2_scalar(state_reference.a, grid))

    max_delta_Pi = _max_scalar(delta_Pi, grid)
    l2_delta_Pi = _l2_scalar(delta_Pi, grid)
    rel_l2_delta_Pi = _relative(l2_delta_Pi, _l2_scalar(state_reference.Pi, grid))

    if state_reference.E is not None and state_test.E is not None:
        delta_E = state_test.E - state_reference.E
        max_delta_E = _max_vector(delta_E, grid)
        l2_delta_E = _l2_vector(delta_E, grid)
        rel_l2_delta_E = _relative(l2_delta_E, _l2_vector(state_reference.E, grid))
    else:
        max_delta_E = 0.0
        l2_delta_E = 0.0
        rel_l2_delta_E = 0.0

    if state_reference.B is not None and state_test.B is not None:
        delta_B = state_test.B - state_reference.B
        max_delta_B = _max_vector(delta_B, grid)
        l2_delta_B = _l2_vector(delta_B, grid)
        rel_l2_delta_B = _relative(l2_delta_B, _l2_vector(state_reference.B, grid))
    else:
        max_delta_B = 0.0
        l2_delta_B = 0.0
        rel_l2_delta_B = 0.0

    edotb_reference = edotb_profile(state_reference, geom_reference)
    edotb_test = edotb_profile(state_test, geom_test)
    delta_EdotB = edotb_test - edotb_reference

    max_delta_EdotB = _max_scalar(delta_EdotB, grid)
    l2_delta_EdotB = _l2_scalar(delta_EdotB, grid)
    rel_l2_delta_EdotB = _relative(
        l2_delta_EdotB,
        _l2_scalar(edotb_reference, grid),
    )

    return DifferenceNorms(
        time=time,
        max_delta_a=max_delta_a,
        l2_delta_a=l2_delta_a,
        rel_l2_delta_a=rel_l2_delta_a,
        max_delta_Pi=max_delta_Pi,
        l2_delta_Pi=l2_delta_Pi,
        rel_l2_delta_Pi=rel_l2_delta_Pi,
        max_delta_E=max_delta_E,
        l2_delta_E=l2_delta_E,
        rel_l2_delta_E=rel_l2_delta_E,
        max_delta_B=max_delta_B,
        l2_delta_B=l2_delta_B,
        rel_l2_delta_B=rel_l2_delta_B,
        max_delta_EdotB=max_delta_EdotB,
        l2_delta_EdotB=l2_delta_EdotB,
        rel_l2_delta_EdotB=rel_l2_delta_EdotB,
    )