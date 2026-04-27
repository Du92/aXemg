"""
Covariant finite-difference operators for spatial 3+1 geometry.

Phase 19 focuses on 1D/2D Cartesian-like grids with a general spatial metric
represented by gamma_ij, gamma^ij and sqrt(gamma).

Implemented operators:
    - scalar_gradient_covariant
    - scalar_gradient_contravariant
    - scalar_laplacian_covariant
    - divergence_vector_covariant
    - lapse_gradient_term

The main scalar Laplacian is

    D_i D^i a =
        1/sqrt(gamma) partial_i[
            sqrt(gamma) gamma^{ij} partial_j a
        ].

The vector divergence is

    D_i V^i =
        1/sqrt(gamma) partial_i[
            sqrt(gamma) V^i
        ].

These operators are designed for reduced 1D/2D simulations.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.derivatives import partial_derivative
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.tensors import raise_vector
from axion_em_gr.geometry.base_metric import GeometryFields

from axion_em_gr.core.levi_civita import LEVI_CIVITA_SYMBOL_3D
from axion_em_gr.core.tensors import lower_vector
from axion_em_gr.core.state import State


def scalar_gradient_covariant(
    scalar: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Compute covariant components of the scalar gradient:

        D_i scalar = partial_i scalar.

    Returns an array with shape:

        (3, *grid.shape_full)

    Only the first grid.ndim components are filled.
    """
    grad = grid.zeros_vector()

    for i in range(grid.ndim):
        grad[i] = partial_derivative(
            scalar,
            grid=grid,
            axis=i,
            order=order,
        )

    return grad


def scalar_gradient_contravariant(
    scalar: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Compute D^i scalar = gamma^{ij} partial_j scalar.
    """
    grad_down = scalar_gradient_covariant(
        scalar=scalar,
        grid=grid,
        order=order,
    )

    return raise_vector(
        grad_down,
        geom.gamma_up,
    )


def scalar_laplacian_covariant(
    scalar: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Compute the covariant scalar Laplacian:

        D_i D^i scalar =
            1/sqrt(gamma) partial_i[
                sqrt(gamma) gamma^{ij} partial_j scalar
            ].

    Works for 1D and 2D grids. The metric itself is treated as a prescribed
    field on the grid.
    """
    grad_down = scalar_gradient_covariant(
        scalar=scalar,
        grid=grid,
        order=order,
    )

    flux = grid.zeros_vector()

    for i in range(grid.ndim):
        tmp = np.zeros_like(scalar)

        for j in range(grid.ndim):
            tmp += geom.gamma_up[i, j] * grad_down[j]

        flux[i] = geom.sqrt_gamma * tmp

    div_flux = np.zeros_like(scalar)

    for i in range(grid.ndim):
        div_flux += partial_derivative(
            flux[i],
            grid=grid,
            axis=i,
            order=order,
        )

    return div_flux / geom.sqrt_gamma


def divergence_vector_covariant(
    vector_up: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Compute the spatial covariant divergence of a contravariant vector:

        D_i V^i =
            1/sqrt(gamma) partial_i[
                sqrt(gamma) V^i
            ].

    Only the first grid.ndim components contribute to the divergence.
    """
    div = np.zeros(grid.shape_full, dtype=float)

    for i in range(grid.ndim):
        flux_i = geom.sqrt_gamma * vector_up[i]

        div += partial_derivative(
            flux_i,
            grid=grid,
            axis=i,
            order=order,
        )

    return div / geom.sqrt_gamma


def lapse_gradient_term(
    scalar: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Compute

        D_i(ln N) D^i scalar.

    Since ln N is a scalar,

        D_i(ln N) = partial_i(ln N),

    and

        D^i scalar = gamma^{ij} partial_j scalar.

    The returned scalar is:

        partial_i(ln N) gamma^{ij} partial_j scalar.
    """
    eps = 1.0e-14

    ln_lapse = np.log(np.maximum(geom.lapse, eps))

    grad_lnN = scalar_gradient_covariant(
        scalar=ln_lapse,
        grid=grid,
        order=order,
    )

    grad_scalar_down = scalar_gradient_covariant(
        scalar=scalar,
        grid=grid,
        order=order,
    )

    grad_scalar_up = raise_vector(
        grad_scalar_down,
        geom.gamma_up,
    )

    result = np.zeros_like(scalar)

    for i in range(grid.ndim):
        result += grad_lnN[i] * grad_scalar_up[i]

    return result

def lie_derivative_vector(
    vector_up: np.ndarray,
    beta: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Compute the spatial Lie derivative of a contravariant vector:

        (L_beta V)^i =
            beta^j partial_j V^i
            - V^j partial_j beta^i.

    Works for 1D/2D reduced grids with 3 vector components.
    """
    result = np.zeros_like(vector_up)

    for i in range(3):
        advective = np.zeros_like(vector_up[i])
        stretching = np.zeros_like(vector_up[i])

        for j in range(grid.ndim):
            advective += beta[j] * partial_derivative(
                vector_up[i],
                grid=grid,
                axis=j,
                order=order,
            )

            stretching += vector_up[j] * partial_derivative(
                beta[i],
                grid=grid,
                axis=j,
                order=order,
            )

        result[i] = advective - stretching

    return result


def curl_covector_curved(
    covector_down: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Compute

        (curl A)^i = epsilon^{ijk} D_j A_k.

    For a covector A_k,

        epsilon^{ijk} D_j A_k = epsilon^{ijk} partial_j A_k,

    because the Christoffel term is symmetric in (j,k) and cancels against
    epsilon^{ijk}.

    In components:

        epsilon^{ijk} = [ijk] / sqrt(gamma).

    This supports 1D and 2D grids while keeping all three vector components.
    """
    curl = np.zeros_like(covector_down)
    eps_symbol = LEVI_CIVITA_SYMBOL_3D

    for i in range(3):
        component = np.zeros_like(covector_down[0])

        for j in range(grid.ndim):
            for k in range(3):
                eps_ijk = eps_symbol[i, j, k]

                if eps_ijk == 0.0:
                    continue

                d_j_A_k = partial_derivative(
                    covector_down[k],
                    grid=grid,
                    axis=j,
                    order=order,
                )

                component += eps_ijk * d_j_A_k

        curl[i] = component / geom.sqrt_gamma

    return curl


def curl_lapse_weighted_covector_curved(
    covector_down: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Compute

        epsilon^{ijk} D_j(N A_k)
        =
        epsilon^{ijk} partial_j(N A_k).

    This is the operator appearing in the 3+1 Maxwell equations.
    """
    weighted = np.zeros_like(covector_down)

    for k in range(3):
        weighted[k] = geom.lapse * covector_down[k]

    return curl_covector_curved(
        covector_down=weighted,
        grid=grid,
        geom=geom,
        order=order,
    )


def axion_gradient_cross_E_curved(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Compute the axion-Maxwell term:

        epsilon^{ijk} E_k D_j a.

    Here E_k is the covariant electric field.
    """
    if state.E is None:
        return np.zeros((3, *grid.shape_full), dtype=float)

    E_down = lower_vector(
        state.E,
        geom.gamma_down,
    )

    grad_a = scalar_gradient_covariant(
        scalar=state.a,
        grid=grid,
        order=order,
    )

    result = np.zeros_like(state.E)
    eps_symbol = LEVI_CIVITA_SYMBOL_3D

    for i in range(3):
        component = np.zeros_like(state.a)

        for j in range(grid.ndim):
            for k in range(3):
                eps_ijk = eps_symbol[i, j, k]

                if eps_ijk == 0.0:
                    continue

                component += eps_ijk * E_down[k] * grad_a[j]

        result[i] = component / geom.sqrt_gamma

    return result