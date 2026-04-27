"""
Axion evolution equations.

The 3+1 axion system is

    partial_t a = N Pi + beta^i partial_i a,

    partial_t Pi =
        beta^i partial_i Pi
        + N [
            D_i D^i a
            + K Pi
            - D_i(ln N) D^i a
            - dV/da
            - g_agamma E_i B^i
        ].

Phase 19 uses covariant spatial operators in both 1D and 2D.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.covariant_derivatives import (
    lapse_gradient_term,
    scalar_laplacian_covariant,
)
from axion_em_gr.core.derivatives import partial_derivative
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.state import State
from axion_em_gr.core.tensors import contract_cov_contra, lower_vector
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.physics.potentials import AxionPotential


def advective_derivative_scalar(
    scalar: np.ndarray,
    beta: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Compute beta^i partial_i scalar.
    """
    result = np.zeros_like(scalar)

    for i in range(grid.ndim):
        result += beta[i] * partial_derivative(
            scalar,
            grid=grid,
            axis=i,
            order=order,
        )

    return result


def axion_em_source_EdotB(
    state: State,
    geom: GeometryFields,
) -> np.ndarray:
    """
    Compute E_i B^i.
    """
    if state.E is None or state.B is None:
        return np.zeros_like(state.a)

    E_down = lower_vector(
        state.E,
        geom.gamma_down,
    )

    return contract_cov_contra(
        E_down,
        state.B,
    )


def compute_axion_rhs(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    potential: AxionPotential,
    numerics: NumericalParameters,
    physical: PhysicalParameters,
    include_em_coupling: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RHS for the axion field.

    Returns
    -------
    rhs_a:
        partial_t a

    rhs_Pi:
        partial_t Pi
    """
    order = numerics.derivative_order

    adv_a = advective_derivative_scalar(
        scalar=state.a,
        beta=geom.shift,
        grid=grid,
        order=order,
    )

    adv_Pi = advective_derivative_scalar(
        scalar=state.Pi,
        beta=geom.shift,
        grid=grid,
        order=order,
    )

    rhs_a = geom.lapse * state.Pi + adv_a

    lap_a = scalar_laplacian_covariant(
        scalar=state.a,
        grid=grid,
        geom=geom,
        order=order,
    )

    lapse_term = lapse_gradient_term(
        scalar=state.a,
        grid=grid,
        geom=geom,
        order=order,
    )

    source = np.zeros_like(state.a)

    if include_em_coupling:
        source = physical.g_agamma * axion_em_source_EdotB(
            state=state,
            geom=geom,
        )

    rhs_Pi = (
        adv_Pi
        +
        geom.lapse
        *
        (
            lap_a
            +
            geom.K * state.Pi
            -
            lapse_term
            -
            potential.dV_da(state.a)
            -
            source
        )
    )

    return rhs_a, rhs_Pi