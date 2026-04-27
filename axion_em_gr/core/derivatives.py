"""
Finite-difference derivative operators.

Phase 7 supports:
- 1D and 2D centered derivatives,
- flat divergence and curl in 1D/2D,
- flat scalar Laplacian in 1D/2D,
- selected 1D 3+1 geometric operators used by previous phases.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.geometry.base_metric import GeometryFields


def _axis_slices_for_centered_derivative(
    ndim: int,
    axis: int,
    g: int,
    shape: tuple[int, ...],
) -> tuple[tuple[slice, ...], tuple[slice, ...], tuple[slice, ...]]:
    """
    Build interior, plus and minus slices for a centered first derivative.
    """
    interior = [slice(g, g + n) for n in shape]
    plus = interior.copy()
    minus = interior.copy()

    plus[axis] = slice(g + 1, g + shape[axis] + 1)
    minus[axis] = slice(g - 1, g + shape[axis] - 1)

    return tuple(interior), tuple(plus), tuple(minus)


def partial_derivative(
    field: np.ndarray,
    grid: Grid,
    axis: int = 0,
    order: int = 2,
) -> np.ndarray:
    """
    Compute partial derivative ∂_axis field.

    Supports 1D and 2D.
    """
    if grid.ndim not in (1, 2):
        raise NotImplementedError("partial_derivative currently supports 1D and 2D.")

    if axis < 0 or axis >= grid.ndim:
        raise ValueError("Invalid derivative axis.")

    if order != 2:
        raise NotImplementedError("Only second-order derivatives are implemented.")

    dx = grid.dx[axis]
    g = grid.nghost

    derivative = np.zeros_like(field)

    interior, plus, minus = _axis_slices_for_centered_derivative(
        ndim=grid.ndim,
        axis=axis,
        g=g,
        shape=grid.shape,
    )

    derivative[interior] = (field[plus] - field[minus]) / (2.0 * dx)

    return derivative


def second_partial_derivative(
    field: np.ndarray,
    grid: Grid,
    axis: int = 0,
    order: int = 2,
) -> np.ndarray:
    """
    Compute second partial derivative ∂_axis∂_axis field.

    Supports 1D and 2D.
    """
    if grid.ndim not in (1, 2):
        raise NotImplementedError(
            "second_partial_derivative currently supports 1D and 2D."
        )

    if axis < 0 or axis >= grid.ndim:
        raise ValueError("Invalid derivative axis.")

    if order != 2:
        raise NotImplementedError("Only second-order derivatives are implemented.")

    dx = grid.dx[axis]
    g = grid.nghost

    derivative = np.zeros_like(field)

    interior, plus, minus = _axis_slices_for_centered_derivative(
        ndim=grid.ndim,
        axis=axis,
        g=g,
        shape=grid.shape,
    )

    derivative[interior] = (
        field[plus] - 2.0 * field[interior] + field[minus]
    ) / dx**2

    return derivative


def gradient_scalar_flat(
    scalar: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Compute flat gradient of a scalar.

    Returns a 3-component covector-like array:

        grad[0] = ∂_x scalar
        grad[1] = ∂_y scalar if ndim >= 2 else 0
        grad[2] = 0 for 1D/2D
    """
    grad = np.zeros((3, *grid.shape_full), dtype=float)

    for axis in range(grid.ndim):
        grad[axis] = partial_derivative(
            scalar,
            grid=grid,
            axis=axis,
            order=order,
        )

    return grad


def laplacian_flat(
    field: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Flat-space Laplacian in 1D or 2D.

    1D:
        ∇² f = ∂_x² f

    2D:
        ∇² f = ∂_x² f + ∂_y² f
    """
    lap = np.zeros_like(field)

    for axis in range(grid.ndim):
        lap += second_partial_derivative(
            field,
            grid=grid,
            axis=axis,
            order=order,
        )

    return lap


def laplacian_flat_1d(
    field: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Backward-compatible flat 1D Laplacian.
    """
    if grid.ndim != 1:
        raise ValueError("laplacian_flat_1d requires grid.ndim == 1.")
    return laplacian_flat(field, grid, order=order)


def divergence_flat(
    vector_up: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Flat-space divergence of a contravariant vector.

    1D:
        ∂_x V^x

    2D:
        ∂_x V^x + ∂_y V^y
    """
    div = np.zeros(grid.shape_full, dtype=float)

    for axis in range(grid.ndim):
        div += partial_derivative(
            vector_up[axis],
            grid=grid,
            axis=axis,
            order=order,
        )

    return div


def divergence_flat_1d(
    vector_up: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Backward-compatible flat 1D divergence.
    """
    if grid.ndim != 1:
        raise ValueError("divergence_flat_1d requires grid.ndim == 1.")
    return divergence_flat(vector_up, grid, order=order)


def curl_covector_flat(
    covector_down: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Flat-space curl of a covector A_i in 1D or 2D.

    1D, dependence only on x:
        (curl A)^x = 0
        (curl A)^y = -∂_x A_z
        (curl A)^z =  ∂_x A_y

    2D, dependence on x,y and ∂_z=0:
        (curl A)^x =  ∂_y A_z
        (curl A)^y = -∂_x A_z
        (curl A)^z =  ∂_x A_y - ∂_y A_x
    """
    curl = np.zeros_like(covector_down)

    d_Ax_dx = partial_derivative(covector_down[0], grid, axis=0, order=order)
    d_Ay_dx = partial_derivative(covector_down[1], grid, axis=0, order=order)
    d_Az_dx = partial_derivative(covector_down[2], grid, axis=0, order=order)

    if grid.ndim == 1:
        curl[0] = 0.0
        curl[1] = -d_Az_dx
        curl[2] = d_Ay_dx
        return curl

    if grid.ndim == 2:
        d_Ax_dy = partial_derivative(covector_down[0], grid, axis=1, order=order)
        d_Az_dy = partial_derivative(covector_down[2], grid, axis=1, order=order)

        curl[0] = d_Az_dy
        curl[1] = -d_Az_dx
        curl[2] = d_Ay_dx - d_Ax_dy
        return curl

    raise NotImplementedError("curl_covector_flat currently supports 1D and 2D.")


def curl_covector_flat_1d(
    covector_down: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Backward-compatible flat 1D curl.
    """
    if grid.ndim != 1:
        raise ValueError("curl_covector_flat_1d requires grid.ndim == 1.")
    return curl_covector_flat(covector_down, grid, order=order)


# ---------------------------------------------------------------------------
# 1D geometric operators from Phase 4/5.
# These remain 1D-only for now.
# ---------------------------------------------------------------------------

def covariant_scalar_laplacian_1d(
    scalar: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Compute the spatial scalar Laplacian in 1D:

        D_i D^i a =
        1/sqrt(gamma) ∂_x [sqrt(gamma) gamma^{xx} ∂_x a].
    """
    if grid.ndim != 1:
        raise NotImplementedError(
            "covariant_scalar_laplacian_1d is only implemented in 1D."
        )

    da_dx = partial_derivative(scalar, grid=grid, axis=0, order=order)
    flux = geom.sqrt_gamma * geom.gamma_up[0, 0] * da_dx
    d_flux_dx = partial_derivative(flux, grid=grid, axis=0, order=order)

    return d_flux_dx / geom.sqrt_gamma


def covariant_divergence_vector_1d(
    vector_up: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Compute the covariant divergence of a contravariant vector in 1D.
    """
    if grid.ndim != 1:
        raise NotImplementedError(
            "covariant_divergence_vector_1d is only implemented in 1D."
        )

    flux = geom.sqrt_gamma * vector_up[0]
    d_flux_dx = partial_derivative(flux, grid=grid, axis=0, order=order)

    return d_flux_dx / geom.sqrt_gamma


def curl_covector_curved_1d(
    covector_down: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    order: int = 2,
) -> np.ndarray:
    """
    Curved-space 1D curl of a covector A_i.
    """
    if grid.ndim != 1:
        raise NotImplementedError("curl_covector_curved_1d is only implemented in 1D.")

    curl = np.zeros_like(covector_down)

    d_Ay_dx = partial_derivative(covector_down[1], grid=grid, axis=0, order=order)
    d_Az_dx = partial_derivative(covector_down[2], grid=grid, axis=0, order=order)

    inv_sqrt_gamma = 1.0 / geom.sqrt_gamma

    curl[0] = 0.0
    curl[1] = -inv_sqrt_gamma * d_Az_dx
    curl[2] = inv_sqrt_gamma * d_Ay_dx

    return curl


def shift_advection_scalar_1d(
    scalar: np.ndarray,
    beta_up: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Compute beta^i ∂_i scalar in 1D.
    """
    if grid.ndim != 1:
        raise NotImplementedError("shift_advection_scalar_1d is only implemented in 1D.")

    d_scalar_dx = partial_derivative(scalar, grid=grid, axis=0, order=order)
    return beta_up[0] * d_scalar_dx


def lie_derivative_vector_1d(
    vector_up: np.ndarray,
    beta_up: np.ndarray,
    grid: Grid,
    order: int = 2,
) -> np.ndarray:
    """
    Compute the Lie derivative of a contravariant vector in 1D:

        (L_beta V)^i = beta^x ∂_x V^i - V^x ∂_x beta^i.
    """
    if grid.ndim != 1:
        raise NotImplementedError("lie_derivative_vector_1d is only implemented in 1D.")

    lie = np.zeros_like(vector_up)

    for i in range(3):
        d_Vi_dx = partial_derivative(vector_up[i], grid=grid, axis=0, order=order)
        d_betai_dx = partial_derivative(beta_up[i], grid=grid, axis=0, order=order)

        lie[i] = beta_up[0] * d_Vi_dx - vector_up[0] * d_betai_dx

    return lie