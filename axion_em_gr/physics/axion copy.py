"""
Axion evolution equations.

Supports:
- 1D 3+1 geometric RHS from previous phases,
- 2D flat Cartesian RHS for Phase 7.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.derivatives import (
    covariant_scalar_laplacian_1d,
    laplacian_flat,
    partial_derivative,
)
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.state import State
from axion_em_gr.core.tensors import contract_cov_contra, lower_vector
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.physics.potentials import AxionPotential


def compute_axion_rhs_flat_nd(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    potential: AxionPotential,
    numerics: NumericalParameters,
    physical: PhysicalParameters | None = None,
    include_em_coupling: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flat Cartesian axion RHS in 1D or 2D:

        ∂_t a = Pi

        ∂_t Pi = ∇² a - dV/da - g E_i B^i.
    """
    a = state.a
    Pi = state.Pi

    lap_a = laplacian_flat(
        a,
        grid=grid,
        order=numerics.derivative_order,
    )

    rhs_a = Pi.copy()
    rhs_Pi = lap_a - potential.dV_da(a)

    if include_em_coupling:
        if physical is None:
            raise ValueError("Physical parameters are required for EM coupling.")

        if state.E is None or state.B is None:
            raise ValueError("Axion-EM coupling requires E and B fields.")

        E_down = lower_vector(state.E, geom.gamma_down)
        E_dot_B = contract_cov_contra(E_down, state.B)

        rhs_Pi += -physical.g_agamma * E_dot_B

    return rhs_a, rhs_Pi


def compute_axion_rhs_3p1_1d(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    potential: AxionPotential,
    numerics: NumericalParameters,
    physical: PhysicalParameters | None = None,
    include_em_coupling: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 1D 3+1 axion RHS:

        ∂_t a = N Pi + beta^x ∂_x a

        ∂_t Pi =
            beta^x ∂_x Pi
            + N [
                D_i D^i a
                + K Pi
                - D_i(ln N)D^i a
                - dV/da
                - g E_i B^i
            ].
    """
    if grid.ndim != 1:
        raise NotImplementedError("compute_axion_rhs_3p1_1d only supports 1D.")

    a = state.a
    Pi = state.Pi

    N = geom.lapse
    beta = geom.shift
    gamma_up = geom.gamma_up

    da_dx = partial_derivative(
        a,
        grid=grid,
        axis=0,
        order=numerics.derivative_order,
    )

    dPi_dx = partial_derivative(
        Pi,
        grid=grid,
        axis=0,
        order=numerics.derivative_order,
    )

    lap_a = covariant_scalar_laplacian_1d(
        a,
        grid=grid,
        geom=geom,
        order=numerics.derivative_order,
    )

    d_lnN_dx = partial_derivative(
        np.log(N),
        grid=grid,
        axis=0,
        order=numerics.derivative_order,
    )

    lapse_gradient_term = d_lnN_dx * gamma_up[0, 0] * da_dx

    rhs_a = N * Pi + beta[0] * da_dx

    rhs_Pi = beta[0] * dPi_dx + N * (
        lap_a
        + geom.K * Pi
        - lapse_gradient_term
        - potential.dV_da(a)
    )

    if include_em_coupling:
        if physical is None:
            raise ValueError("Physical parameters are required for EM coupling.")

        if state.E is None or state.B is None:
            raise ValueError("Axion-EM coupling requires E and B fields.")

        E_down = lower_vector(state.E, geom.gamma_down)
        E_dot_B = contract_cov_contra(E_down, state.B)

        rhs_Pi += -N * physical.g_agamma * E_dot_B

    return rhs_a, rhs_Pi


def compute_axion_rhs(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    potential: AxionPotential,
    numerics: NumericalParameters,
    physical: PhysicalParameters | None = None,
    include_em_coupling: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch axion RHS depending on grid dimensionality.

    1D:
        use 3+1 geometric implementation.

    2D:
        use flat Cartesian implementation for Phase 7.
    """
    if grid.ndim == 1:
        return compute_axion_rhs_3p1_1d(
            state=state,
            grid=grid,
            geom=geom,
            potential=potential,
            numerics=numerics,
            physical=physical,
            include_em_coupling=include_em_coupling,
        )

    if grid.ndim == 2:
        return compute_axion_rhs_flat_nd(
            state=state,
            grid=grid,
            geom=geom,
            potential=potential,
            numerics=numerics,
            physical=physical,
            include_em_coupling=include_em_coupling,
        )

    raise NotImplementedError("Axion RHS currently supports 1D and 2D.")


# Backward-compatible aliases.
compute_axion_rhs_flat_1d = compute_axion_rhs_3p1_1d