"""
Maxwell evolution equations.

Supports:
- 1D 3+1 Maxwell from previous phases,
- 2D flat Cartesian Maxwell for Phase 7.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.derivatives import (
    curl_covector_curved_1d,
    curl_covector_flat,
    gradient_scalar_flat,
    lie_derivative_vector_1d,
    partial_derivative,
)
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.state import State
from axion_em_gr.core.tensors import lower_vector
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.physics.sources import SourceModel


def axion_em_coupling_term_flat_nd(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    physical: PhysicalParameters,
    numerics: NumericalParameters,
) -> np.ndarray:
    """
    Flat Cartesian axion-induced Maxwell source in 1D or 2D:

        g epsilon^{ijk} E_k ∂_j a - g Pi B^i.

    In 2D, ∂_z a = 0.
    """
    if state.E is None or state.B is None:
        raise ValueError("Axion-EM Maxwell coupling requires E and B fields.")

    g_agamma = physical.g_agamma

    E_down = lower_vector(state.E, geom.gamma_down)
    grad_a = gradient_scalar_flat(
        state.a,
        grid=grid,
        order=numerics.derivative_order,
    )

    coupling = np.zeros_like(state.E)

    if grid.ndim == 1:
        da_dx = grad_a[0]

        coupling[0] = -g_agamma * state.Pi * state.B[0]
        coupling[1] = -g_agamma * E_down[2] * da_dx - g_agamma * state.Pi * state.B[1]
        coupling[2] = +g_agamma * E_down[1] * da_dx - g_agamma * state.Pi * state.B[2]

        return coupling

    if grid.ndim == 2:
        da_dx = grad_a[0]
        da_dy = grad_a[1]

        # epsilon^{ijk} E_k ∂_j a
        #
        # C^x = epsilon^{x y z} E_z ∂_y a
        #     = E_z ∂_y a
        #
        # C^y = epsilon^{y x z} E_z ∂_x a
        #     = - E_z ∂_x a
        #
        # C^z = epsilon^{z x y} E_y ∂_x a
        #     + epsilon^{z y x} E_x ∂_y a
        #     = E_y ∂_x a - E_x ∂_y a

        coupling[0] = (
            +g_agamma * E_down[2] * da_dy
            -g_agamma * state.Pi * state.B[0]
        )

        coupling[1] = (
            -g_agamma * E_down[2] * da_dx
            -g_agamma * state.Pi * state.B[1]
        )

        coupling[2] = (
            +g_agamma * E_down[1] * da_dx
            -g_agamma * E_down[0] * da_dy
            -g_agamma * state.Pi * state.B[2]
        )

        return coupling

    raise NotImplementedError("Flat axion-EM coupling supports 1D and 2D.")


def axion_em_coupling_term_3p1_1d(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    physical: PhysicalParameters,
    numerics: NumericalParameters,
) -> np.ndarray:
    """
    1D 3+1 axion-induced Maxwell source:

        g N epsilon^{ijk} E_k D_j a - g N Pi B^i.
    """
    if state.E is None or state.B is None:
        raise ValueError("Axion-EM Maxwell coupling requires E and B fields.")

    if grid.ndim != 1:
        raise NotImplementedError("3+1 coupling currently supports only 1D.")

    g_agamma = physical.g_agamma
    N = geom.lapse
    inv_sqrt_gamma = 1.0 / geom.sqrt_gamma

    E_down = lower_vector(state.E, geom.gamma_down)

    da_dx = partial_derivative(
        state.a,
        grid=grid,
        axis=0,
        order=numerics.derivative_order,
    )

    coupling = np.zeros_like(state.E)

    coupling[0] = -g_agamma * N * state.Pi * state.B[0]

    coupling[1] = (
        -g_agamma * N * inv_sqrt_gamma * E_down[2] * da_dx
        -g_agamma * N * state.Pi * state.B[1]
    )

    coupling[2] = (
        +g_agamma * N * inv_sqrt_gamma * E_down[1] * da_dx
        -g_agamma * N * state.Pi * state.B[2]
    )

    return coupling


def compute_maxwell_rhs_flat_nd(
    state: State,
    t: float,
    grid: Grid,
    geom: GeometryFields,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters | None = None,
    include_axion_coupling: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flat Cartesian Maxwell RHS in 1D or 2D.

    Stable sign convention:

        ∂_t B = - curl E
        ∂_t E = + curl B + j + axion terms
    """
    if state.E is None or state.B is None:
        raise ValueError("Maxwell RHS requires E and B fields in the state.")

    E_down = lower_vector(state.E, geom.gamma_down)
    B_down = lower_vector(state.B, geom.gamma_down)

    curl_E = curl_covector_flat(
        E_down,
        grid=grid,
        order=numerics.derivative_order,
    )

    curl_B = curl_covector_flat(
        B_down,
        grid=grid,
        order=numerics.derivative_order,
    )

    j_up = sources.current(
        state=state,
        t=t,
        grid=grid,
        geom=geom,
    )

    rhs_B = -curl_E
    rhs_E = +curl_B + geom.lapse[None, ...] * j_up

    if include_axion_coupling:
        if physical is None:
            raise ValueError("Physical parameters are required for axion coupling.")

        rhs_E += axion_em_coupling_term_flat_nd(
            state=state,
            grid=grid,
            geom=geom,
            physical=physical,
            numerics=numerics,
        )

    return rhs_E, rhs_B


def compute_maxwell_rhs_3p1_1d(
    state: State,
    t: float,
    grid: Grid,
    geom: GeometryFields,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters | None = None,
    include_axion_coupling: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 1D 3+1 Maxwell RHS.
    """
    if grid.ndim != 1:
        raise NotImplementedError("compute_maxwell_rhs_3p1_1d only supports 1D.")

    if state.E is None or state.B is None:
        raise ValueError("Maxwell RHS requires E and B fields in the state.")

    N = geom.lapse
    beta = geom.shift

    E_up = state.E
    B_up = state.B

    E_down = lower_vector(E_up, geom.gamma_down)
    B_down = lower_vector(B_up, geom.gamma_down)

    N_E_down = N[None, ...] * E_down
    N_B_down = N[None, ...] * B_down

    curl_NE = curl_covector_curved_1d(
        N_E_down,
        grid=grid,
        geom=geom,
        order=numerics.derivative_order,
    )

    curl_NB = curl_covector_curved_1d(
        N_B_down,
        grid=grid,
        geom=geom,
        order=numerics.derivative_order,
    )

    lie_B = lie_derivative_vector_1d(
        B_up,
        beta,
        grid=grid,
        order=numerics.derivative_order,
    )

    lie_E = lie_derivative_vector_1d(
        E_up,
        beta,
        grid=grid,
        order=numerics.derivative_order,
    )

    j_up = sources.current(
        state=state,
        t=t,
        grid=grid,
        geom=geom,
    )

    rhs_B = (
        lie_B
        + N[None, ...] * geom.K[None, ...] * B_up
        - curl_NE
    )

    rhs_E = (
        lie_E
        + N[None, ...] * geom.K[None, ...] * E_up
        + curl_NB
        + N[None, ...] * j_up
    )

    if include_axion_coupling:
        if physical is None:
            raise ValueError("Physical parameters are required for axion coupling.")

        rhs_E += axion_em_coupling_term_3p1_1d(
            state=state,
            grid=grid,
            geom=geom,
            physical=physical,
            numerics=numerics,
        )

    return rhs_E, rhs_B


def compute_maxwell_rhs(
    state: State,
    t: float,
    grid: Grid,
    geom: GeometryFields,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters | None = None,
    include_axion_coupling: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch Maxwell RHS.

    1D:
        use 3+1 implementation.

    2D:
        use flat Cartesian implementation for Phase 7.
    """
    if grid.ndim == 1:
        return compute_maxwell_rhs_3p1_1d(
            state=state,
            t=t,
            grid=grid,
            geom=geom,
            sources=sources,
            numerics=numerics,
            physical=physical,
            include_axion_coupling=include_axion_coupling,
        )

    if grid.ndim == 2:
        return compute_maxwell_rhs_flat_nd(
            state=state,
            t=t,
            grid=grid,
            geom=geom,
            sources=sources,
            numerics=numerics,
            physical=physical,
            include_axion_coupling=include_axion_coupling,
        )

    raise NotImplementedError("Maxwell RHS currently supports 1D and 2D.")


# Backward-compatible aliases.
compute_maxwell_rhs_flat_1d = compute_maxwell_rhs_3p1_1d