"""
Curved-space Poisson solvers.

Phase 20B implements iterative solvers for

    D_i D^i phi = rhs,

where

    D_i D^i phi =
        1/sqrt(gamma) partial_i[
            sqrt(gamma) gamma^{ij} partial_j phi
        ].

The implementation targets 2D diagonal / conformally flat metrics on Cartesian
grids. This covers the current smooth compact-object and isotropic
Schwarzschild-like 2D metrics.

Supported boundary choices:
    - dirichlet
    - neumann
    - outflow

The solver uses a finite-volume-like discretization for diagonal metrics:

    L phi =
        1/sqrt(gamma) [
            d_x(A_x d_x phi) + d_y(A_y d_y phi)
        ],

with

    A_x = sqrt(gamma) gamma^{xx},
    A_y = sqrt(gamma) gamma^{yy}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axion_em_gr.core.boundary import DirichletBoundary, NeumannBoundary, OutflowBoundary
from axion_em_gr.core.grid import Grid
from axion_em_gr.geometry.base_metric import GeometryFields


@dataclass
class CurvedPoissonReport:
    """
    Report returned by curved Poisson solvers.
    """

    method: str
    boundary: str
    iterations: int
    residual_linf: float
    converged: bool
    removed_mean: float


def _apply_phi_boundary(
    phi: np.ndarray,
    grid: Grid,
    boundary: str,
    dirichlet_value: float = 0.0,
) -> None:
    """
    Apply scalar ghost-zone boundary condition for phi.
    """
    if boundary == "dirichlet":
        DirichletBoundary(value=dirichlet_value).apply_array(phi, grid)
        return

    if boundary == "neumann":
        NeumannBoundary().apply_array(phi, grid)
        return

    if boundary == "outflow":
        OutflowBoundary().apply_array(phi, grid)
        return

    raise ValueError(f"Unknown curved Poisson boundary: {boundary!r}")


def mean_over_interior(field: np.ndarray, grid: Grid) -> float:
    return float(np.mean(field[grid.interior_slices]))


def curved_laplacian_diagonal_2d(
    phi: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
) -> np.ndarray:
    """
    Apply the curved scalar Laplacian to phi in 2D.

    Uses a conservative diagonal-metric discretization:

        L phi = 1/sqrt(gamma) [
            d_x(Ax d_x phi) + d_y(Ay d_y phi)
        ]

    with

        Ax = sqrt(gamma) gamma^{xx},
        Ay = sqrt(gamma) gamma^{yy}.

    This assumes gamma^{xy}=0. For conformally flat metrics this is exact.
    """
    if grid.ndim != 2:
        raise ValueError("curved_laplacian_diagonal_2d requires a 2D grid.")

    g = grid.nghost
    nx, ny = grid.shape
    dx, dy = grid.dx

    result = np.zeros_like(phi)

    i = slice(g, g + nx)
    j = slice(g, g + ny)

    # Interior central coefficients.
    sqrt_gamma = geom.sqrt_gamma

    Ax = sqrt_gamma * geom.gamma_up[0, 0]
    Ay = sqrt_gamma * geom.gamma_up[1, 1]

    # Face-centered A coefficients by arithmetic average.
    Ax_ip = 0.5 * (Ax[g + 1:g + nx + 1, g:g + ny] + Ax[g:g + nx, g:g + ny])
    Ax_im = 0.5 * (Ax[g:g + nx, g:g + ny] + Ax[g - 1:g + nx - 1, g:g + ny])

    Ay_jp = 0.5 * (Ay[g:g + nx, g + 1:g + ny + 1] + Ay[g:g + nx, g:g + ny])
    Ay_jm = 0.5 * (Ay[g:g + nx, g:g + ny] + Ay[g:g + nx, g - 1:g + ny - 1])

    phi_c = phi[g:g + nx, g:g + ny]

    phi_ip = phi[g + 1:g + nx + 1, g:g + ny]
    phi_im = phi[g - 1:g + nx - 1, g:g + ny]

    phi_jp = phi[g:g + nx, g + 1:g + ny + 1]
    phi_jm = phi[g:g + nx, g - 1:g + ny - 1]

    term_x = (Ax_ip * (phi_ip - phi_c) - Ax_im * (phi_c - phi_im)) / dx**2
    term_y = (Ay_jp * (phi_jp - phi_c) - Ay_jm * (phi_c - phi_jm)) / dy**2

    result[i, j] = (term_x + term_y) / sqrt_gamma[i, j]

    return result


def curved_poisson_residual_diagonal_2d(
    phi: np.ndarray,
    rhs: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
) -> np.ndarray:
    """
    Residual:

        R = D_iD^i phi - rhs.
    """
    return curved_laplacian_diagonal_2d(
        phi=phi,
        grid=grid,
        geom=geom,
    ) - rhs


def solve_curved_poisson_jacobi_diagonal_2d(
    rhs_full: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    boundary: str = "dirichlet",
    dirichlet_value: float = 0.0,
    max_iterations: int = 50_000,
    tolerance: float = 1.0e-8,
    omega: float = 2.0 / 3.0,
    remove_mean_for_neumann: bool = True,
) -> tuple[np.ndarray, CurvedPoissonReport]:
    """
    Weighted Jacobi solver for

        D_iD^i phi = rhs

    on a 2D diagonal metric.

    For Neumann/outflow boundaries, the mean RHS is removed by default because
    a pure Neumann elliptic problem has a solvability condition.
    """
    if grid.ndim != 2:
        raise ValueError("solve_curved_poisson_jacobi_diagonal_2d requires 2D.")

    g = grid.nghost
    nx, ny = grid.shape
    dx, dy = grid.dx

    rhs = rhs_full.copy()
    removed_mean = 0.0

    if boundary in ("neumann", "outflow") and remove_mean_for_neumann:
        removed_mean = mean_over_interior(rhs, grid)
        rhs[grid.interior_slices] -= removed_mean

    phi = np.zeros_like(rhs)
    phi_new = np.zeros_like(rhs)

    _apply_phi_boundary(phi, grid, boundary, dirichlet_value)
    _apply_phi_boundary(phi_new, grid, boundary, dirichlet_value)

    sqrt_gamma = geom.sqrt_gamma
    Ax = sqrt_gamma * geom.gamma_up[0, 0]
    Ay = sqrt_gamma * geom.gamma_up[1, 1]

    i = slice(g, g + nx)
    j = slice(g, g + ny)

    Ax_ip = 0.5 * (Ax[g + 1:g + nx + 1, g:g + ny] + Ax[g:g + nx, g:g + ny])
    Ax_im = 0.5 * (Ax[g:g + nx, g:g + ny] + Ax[g - 1:g + nx - 1, g:g + ny])

    Ay_jp = 0.5 * (Ay[g:g + nx, g + 1:g + ny + 1] + Ay[g:g + nx, g:g + ny])
    Ay_jm = 0.5 * (Ay[g:g + nx, g:g + ny] + Ay[g:g + nx, g - 1:g + ny - 1])

    denom = (Ax_ip + Ax_im) / dx**2 + (Ay_jp + Ay_jm) / dy**2

    # Avoid division by zero in pathological metrics.
    denom = np.maximum(denom, 1.0e-300)

    converged = False
    residual_linf = np.inf

    for iteration in range(1, max_iterations + 1):
        phi_ip = phi[g + 1:g + nx + 1, g:g + ny]
        phi_im = phi[g - 1:g + nx - 1, g:g + ny]
        phi_jp = phi[g:g + nx, g + 1:g + ny + 1]
        phi_jm = phi[g:g + nx, g - 1:g + ny - 1]

        rhs_eff = sqrt_gamma[i, j] * rhs[i, j]

        candidate = (
            Ax_ip * phi_ip / dx**2
            +
            Ax_im * phi_im / dx**2
            +
            Ay_jp * phi_jp / dy**2
            +
            Ay_jm * phi_jm / dy**2
            -
            rhs_eff
        ) / denom

        phi_new[i, j] = (1.0 - omega) * phi[i, j] + omega * candidate

        _apply_phi_boundary(phi_new, grid, boundary, dirichlet_value)

        if iteration % 50 == 0 or iteration == 1:
            residual = curved_poisson_residual_diagonal_2d(
                phi=phi_new,
                rhs=rhs,
                grid=grid,
                geom=geom,
            )
            residual_linf = float(np.max(np.abs(residual[i, j])))

            if residual_linf < tolerance:
                converged = True
                phi = phi_new.copy()
                break

        phi, phi_new = phi_new, phi

    if not converged:
        residual = curved_poisson_residual_diagonal_2d(
            phi=phi,
            rhs=rhs,
            grid=grid,
            geom=geom,
        )
        residual_linf = float(np.max(np.abs(residual[i, j])))

    _apply_phi_boundary(phi, grid, boundary, dirichlet_value)

    report = CurvedPoissonReport(
        method="jacobi_diagonal_2d",
        boundary=boundary,
        iterations=iteration,
        residual_linf=residual_linf,
        converged=converged,
        removed_mean=removed_mean,
    )

    return phi, report


def solve_curved_poisson_sor_diagonal_2d(
    rhs_full: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    boundary: str = "dirichlet",
    dirichlet_value: float = 0.0,
    max_iterations: int = 20_000,
    tolerance: float = 1.0e-8,
    omega: float = 1.6,
    remove_mean_for_neumann: bool = True,
) -> tuple[np.ndarray, CurvedPoissonReport]:
    """
    SOR solver for

        D_iD^i phi = rhs

    on a 2D diagonal metric.

    This uses Python loops and can be slower than vectorized Jacobi for large
    grids, but it often converges in fewer iterations.
    """
    if grid.ndim != 2:
        raise ValueError("solve_curved_poisson_sor_diagonal_2d requires 2D.")

    g = grid.nghost
    nx, ny = grid.shape
    dx, dy = grid.dx

    rhs = rhs_full.copy()
    removed_mean = 0.0

    if boundary in ("neumann", "outflow") and remove_mean_for_neumann:
        removed_mean = mean_over_interior(rhs, grid)
        rhs[grid.interior_slices] -= removed_mean

    phi = np.zeros_like(rhs)

    _apply_phi_boundary(phi, grid, boundary, dirichlet_value)

    sqrt_gamma = geom.sqrt_gamma
    Ax = sqrt_gamma * geom.gamma_up[0, 0]
    Ay = sqrt_gamma * geom.gamma_up[1, 1]

    converged = False
    residual_linf = np.inf

    for iteration in range(1, max_iterations + 1):
        for ii in range(g, g + nx):
            for jj in range(g, g + ny):
                Ax_ip = 0.5 * (Ax[ii + 1, jj] + Ax[ii, jj])
                Ax_im = 0.5 * (Ax[ii, jj] + Ax[ii - 1, jj])

                Ay_jp = 0.5 * (Ay[ii, jj + 1] + Ay[ii, jj])
                Ay_jm = 0.5 * (Ay[ii, jj] + Ay[ii, jj - 1])

                denom = (Ax_ip + Ax_im) / dx**2 + (Ay_jp + Ay_jm) / dy**2
                denom = max(denom, 1.0e-300)

                rhs_eff = sqrt_gamma[ii, jj] * rhs[ii, jj]

                candidate = (
                    Ax_ip * phi[ii + 1, jj] / dx**2
                    +
                    Ax_im * phi[ii - 1, jj] / dx**2
                    +
                    Ay_jp * phi[ii, jj + 1] / dy**2
                    +
                    Ay_jm * phi[ii, jj - 1] / dy**2
                    -
                    rhs_eff
                ) / denom

                phi[ii, jj] = (1.0 - omega) * phi[ii, jj] + omega * candidate

        _apply_phi_boundary(phi, grid, boundary, dirichlet_value)

        if iteration % 50 == 0 or iteration == 1:
            residual = curved_poisson_residual_diagonal_2d(
                phi=phi,
                rhs=rhs,
                grid=grid,
                geom=geom,
            )

            residual_linf = float(np.max(np.abs(residual[grid.interior_slices])))

            if residual_linf < tolerance:
                converged = True
                break

    report = CurvedPoissonReport(
        method="sor_diagonal_2d",
        boundary=boundary,
        iterations=iteration,
        residual_linf=residual_linf,
        converged=converged,
        removed_mean=removed_mean,
    )

    return phi, report


def solve_curved_poisson_diagonal_2d(
    rhs_full: np.ndarray,
    grid: Grid,
    geom: GeometryFields,
    method: str = "jacobi",
    boundary: str = "dirichlet",
    dirichlet_value: float = 0.0,
    max_iterations: int = 50_000,
    tolerance: float = 1.0e-8,
    omega: float | None = None,
    remove_mean_for_neumann: bool = True,
) -> tuple[np.ndarray, CurvedPoissonReport]:
    """
    Dispatch curved Poisson solver for 2D diagonal metrics.
    """
    if method == "jacobi":
        if omega is None:
            omega = 2.0 / 3.0

        return solve_curved_poisson_jacobi_diagonal_2d(
            rhs_full=rhs_full,
            grid=grid,
            geom=geom,
            boundary=boundary,
            dirichlet_value=dirichlet_value,
            max_iterations=max_iterations,
            tolerance=tolerance,
            omega=omega,
            remove_mean_for_neumann=remove_mean_for_neumann,
        )

    if method == "sor":
        if omega is None:
            omega = 1.6

        return solve_curved_poisson_sor_diagonal_2d(
            rhs_full=rhs_full,
            grid=grid,
            geom=geom,
            boundary=boundary,
            dirichlet_value=dirichlet_value,
            max_iterations=max_iterations,
            tolerance=tolerance,
            omega=omega,
            remove_mean_for_neumann=remove_mean_for_neumann,
        )

    raise ValueError(f"Unknown curved Poisson method: {method!r}")