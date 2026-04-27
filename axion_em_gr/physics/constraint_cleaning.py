"""
Constraint cleaning / projection utilities.

This module implements electrostatic projection for the flat-space
axion-modified Gauss constraint:

    C_E = div E - rho + g B^i partial_i a.

We correct the electric field by

    E^i_new = E^i - partial^i phi,

where phi solves

    Laplacian(phi) = C_E.

Supported Poisson solvers:
    - periodic_fft
    - jacobi_dirichlet
    - jacobi_neumann
    - sor_dirichlet
    - sor_neumann

The periodic FFT solver is fast but assumes periodic compatibility.
The Jacobi/SOR solvers are slower but compatible with non-periodic boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axion_em_gr.core.boundary import (
    DirichletBoundary,
    NeumannBoundary,
    OutflowBoundary,
    PeriodicBoundary,
)
from axion_em_gr.core.derivatives import gradient_scalar_flat
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.state import State
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.physics.constraints import (
    constraint_norms,
    electric_constraint,
)
from axion_em_gr.physics.sources import SourceModel
from axion_em_gr.core.covariant_derivatives import scalar_gradient_contravariant
from axion_em_gr.physics.curved_poisson import solve_curved_poisson_diagonal_2d


@dataclass
class CleaningReport:
    """
    Summary of an electric constraint-cleaning operation.
    """

    method: str

    mean_constraint_before: float
    l2_constraint_before: float
    linf_constraint_before: float

    mean_constraint_after: float
    l2_constraint_after: float
    linf_constraint_after: float

    poisson_zero_mode_removed: float

    poisson_iterations: int
    poisson_residual_linf: float
    poisson_converged: bool


def _volume_element(grid: Grid) -> float:
    if grid.ndim == 1:
        return grid.dx[0]

    if grid.ndim == 2:
        return grid.dx[0] * grid.dx[1]

    raise NotImplementedError("Only 1D and 2D grids are supported.")


def mean_over_interior(field: np.ndarray, grid: Grid) -> float:
    """
    Mean value over the physical interior.
    """
    return float(np.mean(field[grid.interior_slices]))


def _apply_phi_boundary(
    phi: np.ndarray,
    grid: Grid,
    poisson_boundary: str,
    dirichlet_value: float = 0.0,
) -> None:
    """
    Apply boundary condition to phi during a non-periodic Poisson iteration.
    """
    if poisson_boundary == "dirichlet":
        DirichletBoundary(value=dirichlet_value).apply_array(phi, grid)
        return

    if poisson_boundary in ("neumann", "outflow"):
        NeumannBoundary().apply_array(phi, grid)
        return

    if poisson_boundary == "periodic":
        PeriodicBoundary().apply_array(phi, grid)
        return

    raise ValueError(f"Unknown poisson boundary: {poisson_boundary!r}")


def _poisson_residual_flat(
    phi: np.ndarray,
    rhs: np.ndarray,
    grid: Grid,
) -> np.ndarray:
    """
    Compute finite-difference residual:

        R = Laplacian(phi) - rhs

    over the full grid, with meaningful values in the interior.
    """
    residual = np.zeros_like(rhs)
    g = grid.nghost

    if grid.ndim == 1:
        nx = grid.shape[0]
        dx = grid.dx[0]

        interior = slice(g, g + nx)
        plus = slice(g + 1, g + nx + 1)
        minus = slice(g - 1, g + nx - 1)

        lap_phi = (phi[plus] - 2.0 * phi[interior] + phi[minus]) / dx**2
        residual[interior] = lap_phi - rhs[interior]

        return residual

    if grid.ndim == 2:
        nx, ny = grid.shape
        dx, dy = grid.dx

        i = slice(g, g + nx)
        j = slice(g, g + ny)

        ip = slice(g + 1, g + nx + 1)
        im = slice(g - 1, g + nx - 1)

        jp = slice(g + 1, g + ny + 1)
        jm = slice(g - 1, g + ny - 1)

        lap_phi = (
            (phi[ip, j] - 2.0 * phi[i, j] + phi[im, j]) / dx**2
            +
            (phi[i, jp] - 2.0 * phi[i, j] + phi[i, jm]) / dy**2
        )

        residual[i, j] = lap_phi - rhs[i, j]

        return residual

    raise NotImplementedError("Poisson residual supports only 1D and 2D.")


# ---------------------------------------------------------------------------
# Periodic FFT solvers from previous phase
# ---------------------------------------------------------------------------

def solve_periodic_poisson_1d(
    rhs_full: np.ndarray,
    grid: Grid,
) -> tuple[np.ndarray, float]:
    """
    Solve d_x^2 phi = rhs on a periodic 1D grid using FFT.
    """
    if grid.ndim != 1:
        raise ValueError("solve_periodic_poisson_1d requires grid.ndim == 1.")

    boundary = PeriodicBoundary()

    g = grid.nghost
    nx = grid.shape[0]
    dx = grid.dx[0]

    rhs = rhs_full[g:g + nx].copy()

    removed_zero_mode = float(np.mean(rhs))
    rhs -= removed_zero_mode

    rhs_hat = np.fft.fft(rhs)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    k2 = kx**2

    phi_hat = np.zeros_like(rhs_hat, dtype=complex)

    nonzero = k2 > 0.0

    phi_hat[nonzero] = -rhs_hat[nonzero] / k2[nonzero]
    phi_hat[~nonzero] = 0.0

    phi = np.real(np.fft.ifft(phi_hat))

    phi_full = np.zeros_like(rhs_full)
    phi_full[g:g + nx] = phi

    boundary.apply_array(phi_full, grid)

    return phi_full, removed_zero_mode


def solve_periodic_poisson_2d(
    rhs_full: np.ndarray,
    grid: Grid,
) -> tuple[np.ndarray, float]:
    """
    Solve (d_x^2 + d_y^2) phi = rhs on a periodic 2D grid using FFT.
    """
    if grid.ndim != 2:
        raise ValueError("solve_periodic_poisson_2d requires grid.ndim == 2.")

    boundary = PeriodicBoundary()

    g = grid.nghost
    nx, ny = grid.shape
    dx, dy = grid.dx

    rhs = rhs_full[g:g + nx, g:g + ny].copy()

    removed_zero_mode = float(np.mean(rhs))
    rhs -= removed_zero_mode

    rhs_hat = np.fft.fftn(rhs)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)

    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX**2 + KY**2

    phi_hat = np.zeros_like(rhs_hat, dtype=complex)

    nonzero = k2 > 0.0

    phi_hat[nonzero] = -rhs_hat[nonzero] / k2[nonzero]
    phi_hat[~nonzero] = 0.0

    phi = np.real(np.fft.ifftn(phi_hat))

    phi_full = np.zeros_like(rhs_full)
    phi_full[g:g + nx, g:g + ny] = phi

    boundary.apply_array(phi_full, grid)

    return phi_full, removed_zero_mode


def solve_periodic_poisson_flat(
    rhs_full: np.ndarray,
    grid: Grid,
) -> tuple[np.ndarray, float]:
    """
    Dispatch periodic Poisson solver for 1D or 2D.
    """
    if grid.ndim == 1:
        return solve_periodic_poisson_1d(rhs_full, grid)

    if grid.ndim == 2:
        return solve_periodic_poisson_2d(rhs_full, grid)

    raise NotImplementedError("Periodic Poisson solve is implemented for 1D and 2D.")


# ---------------------------------------------------------------------------
# Non-periodic Jacobi/SOR solvers
# ---------------------------------------------------------------------------

def solve_poisson_jacobi_1d(
    rhs_full: np.ndarray,
    grid: Grid,
    poisson_boundary: str = "dirichlet",
    dirichlet_value: float = 0.0,
    max_iterations: int = 20_000,
    tolerance: float = 1.0e-10,
    omega: float = 2.0 / 3.0,
    remove_mean_for_neumann: bool = True,
) -> tuple[np.ndarray, int, float, bool, float]:
    """
    Solve d_x^2 phi = rhs with weighted Jacobi.

    Returns
    -------
    phi, iterations, residual_linf, converged, removed_zero_mode
    """
    if grid.ndim != 1:
        raise ValueError("solve_poisson_jacobi_1d requires grid.ndim == 1.")

    g = grid.nghost
    nx = grid.shape[0]
    dx = grid.dx[0]

    rhs = rhs_full.copy()
    removed_zero_mode = 0.0

    if poisson_boundary in ("neumann", "outflow") and remove_mean_for_neumann:
        removed_zero_mode = mean_over_interior(rhs, grid)
        rhs[grid.interior_slices] -= removed_zero_mode

    phi = np.zeros_like(rhs_full)
    phi_new = np.zeros_like(rhs_full)

    _apply_phi_boundary(phi, grid, poisson_boundary, dirichlet_value)
    _apply_phi_boundary(phi_new, grid, poisson_boundary, dirichlet_value)

    interior = slice(g, g + nx)
    plus = slice(g + 1, g + nx + 1)
    minus = slice(g - 1, g + nx - 1)

    converged = False
    residual_linf = np.inf

    for iteration in range(1, max_iterations + 1):
        phi_candidate = 0.5 * (
            phi[plus]
            +
            phi[minus]
            -
            dx**2 * rhs[interior]
        )

        phi_new[interior] = (1.0 - omega) * phi[interior] + omega * phi_candidate

        _apply_phi_boundary(phi_new, grid, poisson_boundary, dirichlet_value)

        if iteration % 25 == 0 or iteration == 1:
            residual = _poisson_residual_flat(phi_new, rhs, grid)
            residual_linf = float(np.max(np.abs(residual[interior])))

            if residual_linf < tolerance:
                converged = True
                phi = phi_new.copy()
                break

        phi, phi_new = phi_new, phi

    _apply_phi_boundary(phi, grid, poisson_boundary, dirichlet_value)

    if not converged:
        residual = _poisson_residual_flat(phi, rhs, grid)
        residual_linf = float(np.max(np.abs(residual[interior])))

    return phi, iteration, residual_linf, converged, removed_zero_mode


def solve_poisson_jacobi_2d(
    rhs_full: np.ndarray,
    grid: Grid,
    poisson_boundary: str = "dirichlet",
    dirichlet_value: float = 0.0,
    max_iterations: int = 50_000,
    tolerance: float = 1.0e-8,
    omega: float = 2.0 / 3.0,
    remove_mean_for_neumann: bool = True,
) -> tuple[np.ndarray, int, float, bool, float]:
    """
    Solve 2D Poisson equation with weighted Jacobi:

        (d_x^2 + d_y^2) phi = rhs.

    Returns
    -------
    phi, iterations, residual_linf, converged, removed_zero_mode
    """
    if grid.ndim != 2:
        raise ValueError("solve_poisson_jacobi_2d requires grid.ndim == 2.")

    g = grid.nghost
    nx, ny = grid.shape
    dx, dy = grid.dx

    rhs = rhs_full.copy()
    removed_zero_mode = 0.0

    if poisson_boundary in ("neumann", "outflow") and remove_mean_for_neumann:
        removed_zero_mode = mean_over_interior(rhs, grid)
        rhs[grid.interior_slices] -= removed_zero_mode

    phi = np.zeros_like(rhs_full)
    phi_new = np.zeros_like(rhs_full)

    _apply_phi_boundary(phi, grid, poisson_boundary, dirichlet_value)
    _apply_phi_boundary(phi_new, grid, poisson_boundary, dirichlet_value)

    i = slice(g, g + nx)
    j = slice(g, g + ny)

    ip = slice(g + 1, g + nx + 1)
    im = slice(g - 1, g + nx - 1)

    jp = slice(g + 1, g + ny + 1)
    jm = slice(g - 1, g + ny - 1)

    inv_denom = 1.0 / (2.0 / dx**2 + 2.0 / dy**2)

    converged = False
    residual_linf = np.inf

    for iteration in range(1, max_iterations + 1):
        phi_candidate = (
            (phi[ip, j] + phi[im, j]) / dx**2
            +
            (phi[i, jp] + phi[i, jm]) / dy**2
            -
            rhs[i, j]
        ) * inv_denom

        phi_new[i, j] = (1.0 - omega) * phi[i, j] + omega * phi_candidate

        _apply_phi_boundary(phi_new, grid, poisson_boundary, dirichlet_value)

        if iteration % 50 == 0 or iteration == 1:
            residual = _poisson_residual_flat(phi_new, rhs, grid)
            residual_linf = float(np.max(np.abs(residual[i, j])))

            if residual_linf < tolerance:
                converged = True
                phi = phi_new.copy()
                break

        phi, phi_new = phi_new, phi

    _apply_phi_boundary(phi, grid, poisson_boundary, dirichlet_value)

    if not converged:
        residual = _poisson_residual_flat(phi, rhs, grid)
        residual_linf = float(np.max(np.abs(residual[i, j])))

    return phi, iteration, residual_linf, converged, removed_zero_mode


def solve_poisson_sor_1d(
    rhs_full: np.ndarray,
    grid: Grid,
    poisson_boundary: str = "dirichlet",
    dirichlet_value: float = 0.0,
    max_iterations: int = 20_000,
    tolerance: float = 1.0e-10,
    omega: float = 1.6,
    remove_mean_for_neumann: bool = True,
) -> tuple[np.ndarray, int, float, bool, float]:
    """
    Solve 1D Poisson equation using Gauss-Seidel/SOR.
    """
    if grid.ndim != 1:
        raise ValueError("solve_poisson_sor_1d requires grid.ndim == 1.")

    g = grid.nghost
    nx = grid.shape[0]
    dx = grid.dx[0]

    rhs = rhs_full.copy()
    removed_zero_mode = 0.0

    if poisson_boundary in ("neumann", "outflow") and remove_mean_for_neumann:
        removed_zero_mode = mean_over_interior(rhs, grid)
        rhs[grid.interior_slices] -= removed_zero_mode

    phi = np.zeros_like(rhs_full)
    _apply_phi_boundary(phi, grid, poisson_boundary, dirichlet_value)

    converged = False
    residual_linf = np.inf

    for iteration in range(1, max_iterations + 1):
        for ii in range(g, g + nx):
            candidate = 0.5 * (
                phi[ii + 1]
                +
                phi[ii - 1]
                -
                dx**2 * rhs[ii]
            )
            phi[ii] = (1.0 - omega) * phi[ii] + omega * candidate

        _apply_phi_boundary(phi, grid, poisson_boundary, dirichlet_value)

        if iteration % 25 == 0 or iteration == 1:
            residual = _poisson_residual_flat(phi, rhs, grid)
            residual_linf = float(np.max(np.abs(residual[grid.interior_slices])))

            if residual_linf < tolerance:
                converged = True
                break

    return phi, iteration, residual_linf, converged, removed_zero_mode


def solve_poisson_sor_2d(
    rhs_full: np.ndarray,
    grid: Grid,
    poisson_boundary: str = "dirichlet",
    dirichlet_value: float = 0.0,
    max_iterations: int = 50_000,
    tolerance: float = 1.0e-8,
    omega: float = 1.7,
    remove_mean_for_neumann: bool = True,
) -> tuple[np.ndarray, int, float, bool, float]:
    """
    Solve 2D Poisson equation using Gauss-Seidel/SOR.

    This pure-Python loop is slower than Jacobi vectorization for large grids,
    but often converges in fewer iterations. Use moderate grid sizes.
    """
    if grid.ndim != 2:
        raise ValueError("solve_poisson_sor_2d requires grid.ndim == 2.")

    g = grid.nghost
    nx, ny = grid.shape
    dx, dy = grid.dx

    rhs = rhs_full.copy()
    removed_zero_mode = 0.0

    if poisson_boundary in ("neumann", "outflow") and remove_mean_for_neumann:
        removed_zero_mode = mean_over_interior(rhs, grid)
        rhs[grid.interior_slices] -= removed_zero_mode

    phi = np.zeros_like(rhs_full)
    _apply_phi_boundary(phi, grid, poisson_boundary, dirichlet_value)

    inv_denom = 1.0 / (2.0 / dx**2 + 2.0 / dy**2)

    converged = False
    residual_linf = np.inf

    i_start, i_end = g, g + nx
    j_start, j_end = g, g + ny

    for iteration in range(1, max_iterations + 1):
        for ii in range(i_start, i_end):
            for jj in range(j_start, j_end):
                candidate = (
                    (phi[ii + 1, jj] + phi[ii - 1, jj]) / dx**2
                    +
                    (phi[ii, jj + 1] + phi[ii, jj - 1]) / dy**2
                    -
                    rhs[ii, jj]
                ) * inv_denom

                phi[ii, jj] = (1.0 - omega) * phi[ii, jj] + omega * candidate

        _apply_phi_boundary(phi, grid, poisson_boundary, dirichlet_value)

        if iteration % 50 == 0 or iteration == 1:
            residual = _poisson_residual_flat(phi, rhs, grid)
            residual_linf = float(np.max(np.abs(residual[grid.interior_slices])))

            if residual_linf < tolerance:
                converged = True
                break

    return phi, iteration, residual_linf, converged, removed_zero_mode


def solve_poisson_nonperiodic_flat(
    rhs_full: np.ndarray,
    grid: Grid,
    method: str = "jacobi",
    poisson_boundary: str = "dirichlet",
    dirichlet_value: float = 0.0,
    max_iterations: int = 50_000,
    tolerance: float = 1.0e-8,
    omega: float | None = None,
    remove_mean_for_neumann: bool = True,
) -> tuple[np.ndarray, int, float, bool, float]:
    """
    Dispatch non-periodic Poisson solver.
    """
    if method == "jacobi":
        if omega is None:
            omega = 2.0 / 3.0

        if grid.ndim == 1:
            return solve_poisson_jacobi_1d(
                rhs_full=rhs_full,
                grid=grid,
                poisson_boundary=poisson_boundary,
                dirichlet_value=dirichlet_value,
                max_iterations=max_iterations,
                tolerance=tolerance,
                omega=omega,
                remove_mean_for_neumann=remove_mean_for_neumann,
            )

        if grid.ndim == 2:
            return solve_poisson_jacobi_2d(
                rhs_full=rhs_full,
                grid=grid,
                poisson_boundary=poisson_boundary,
                dirichlet_value=dirichlet_value,
                max_iterations=max_iterations,
                tolerance=tolerance,
                omega=omega,
                remove_mean_for_neumann=remove_mean_for_neumann,
            )

    if method == "sor":
        if omega is None:
            omega = 1.7

        if grid.ndim == 1:
            return solve_poisson_sor_1d(
                rhs_full=rhs_full,
                grid=grid,
                poisson_boundary=poisson_boundary,
                dirichlet_value=dirichlet_value,
                max_iterations=max_iterations,
                tolerance=tolerance,
                omega=omega,
                remove_mean_for_neumann=remove_mean_for_neumann,
            )

        if grid.ndim == 2:
            return solve_poisson_sor_2d(
                rhs_full=rhs_full,
                grid=grid,
                poisson_boundary=poisson_boundary,
                dirichlet_value=dirichlet_value,
                max_iterations=max_iterations,
                tolerance=tolerance,
                omega=omega,
                remove_mean_for_neumann=remove_mean_for_neumann,
            )

    raise ValueError(f"Unknown non-periodic Poisson method: {method!r}")


# ---------------------------------------------------------------------------
# Main cleaning interface
# ---------------------------------------------------------------------------

def clean_electric_constraint_flat(
    state: State,
    t: float,
    grid: Grid,
    geom: GeometryFields,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters,
    include_axion_coupling: bool = True,
    poisson_solver: str = "periodic_fft",
    poisson_boundary: str = "periodic",
    dirichlet_value: float = 0.0,
    max_iterations: int = 50_000,
    tolerance: float = 1.0e-8,
    omega: float | None = None,
    remove_mean_for_neumann: bool = True,
) -> tuple[State, CleaningReport]:
    """
    Project the electric field to reduce the flat-space Gauss constraint.

    Parameters
    ----------
    poisson_solver:
        One of:
            "periodic_fft"
            "jacobi"
            "sor"

    poisson_boundary:
        One of:
            "periodic"
            "dirichlet"
            "neumann"
            "outflow"

    Notes
    -----
    For periodic_fft, poisson_boundary is ignored and the periodic solver is
    used.

    For Neumann/outflow Poisson problems, the mean RHS is removed by default,
    because pure Neumann Poisson is solvable only if the source has zero mean.

        Notes
    -----
    This is a flat-space projection. For curved 2D geometries, it still helps
    reduce the coordinate-space Gauss residual, but it is not yet the fully
    covariant projection solving

        D_i D^i phi = C_E.

    A fully metric-compatible cleaner requires a curved Poisson solver.
    """
    if state.E is None or state.B is None:
        raise ValueError("Electric constraint cleaning requires E and B fields.")

    if grid.ndim not in (1, 2):
        raise NotImplementedError("Electric cleaning supports only 1D and 2D.")

    # Apply a boundary consistent with the Poisson problem before measuring C_E.
    if poisson_solver == "periodic_fft":
        pre_boundary = PeriodicBoundary()
    elif poisson_boundary == "dirichlet":
        pre_boundary = DirichletBoundary(value=dirichlet_value)
    elif poisson_boundary in ("neumann", "outflow"):
        pre_boundary = OutflowBoundary()
    else:
        raise ValueError(f"Unknown poisson_boundary: {poisson_boundary!r}")

    work = state.copy()
    pre_boundary.apply_state(work, grid)

    constraint_before = electric_constraint(
        state=work,
        t=t,
        grid=grid,
        geom=geom,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=include_axion_coupling,
    )

    l2_before, linf_before = constraint_norms(constraint_before, grid)
    mean_before = mean_over_interior(constraint_before, grid)

    if poisson_solver == "periodic_fft":
        phi, removed_zero_mode = solve_periodic_poisson_flat(
            rhs_full=constraint_before,
            grid=grid,
        )
        poisson_iterations = 1
        poisson_residual_linf = 0.0
        poisson_converged = True
        method_name = "periodic_fft"

    elif poisson_solver in ("jacobi", "sor"):
        (
            phi,
            poisson_iterations,
            poisson_residual_linf,
            poisson_converged,
            removed_zero_mode,
        ) = solve_poisson_nonperiodic_flat(
            rhs_full=constraint_before,
            grid=grid,
            method=poisson_solver,
            poisson_boundary=poisson_boundary,
            dirichlet_value=dirichlet_value,
            max_iterations=max_iterations,
            tolerance=tolerance,
            omega=omega,
            remove_mean_for_neumann=remove_mean_for_neumann,
        )
        method_name = f"{poisson_solver}_{poisson_boundary}"

    else:
        raise ValueError(f"Unknown poisson_solver: {poisson_solver!r}")

    grad_phi = gradient_scalar_flat(
        phi,
        grid=grid,
        order=numerics.derivative_order,
    )

    cleaned = work.copy()

    for i in range(grid.ndim):
        cleaned.E[i] -= grad_phi[i]

    pre_boundary.apply_state(cleaned, grid)

    constraint_after = electric_constraint(
        state=cleaned,
        t=t,
        grid=grid,
        geom=geom,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=include_axion_coupling,
    )

    l2_after, linf_after = constraint_norms(constraint_after, grid)
    mean_after = mean_over_interior(constraint_after, grid)

    report = CleaningReport(
        method=method_name,
        mean_constraint_before=mean_before,
        l2_constraint_before=l2_before,
        linf_constraint_before=linf_before,
        mean_constraint_after=mean_after,
        l2_constraint_after=l2_after,
        linf_constraint_after=linf_after,
        poisson_zero_mode_removed=removed_zero_mode,
        poisson_iterations=poisson_iterations,
        poisson_residual_linf=poisson_residual_linf,
        poisson_converged=poisson_converged,
    )

    return cleaned, report

def clean_electric_constraint_curved(
    state: State,
    t: float,
    grid: Grid,
    geom: GeometryFields,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters,
    include_axion_coupling: bool = True,
    poisson_method: str = "jacobi",
    poisson_boundary: str = "dirichlet",
    dirichlet_value: float = 0.0,
    max_iterations: int = 50_000,
    tolerance: float = 1.0e-8,
    omega: float | None = None,
    remove_mean_for_neumann: bool = True,
) -> tuple[State, CleaningReport]:
    """
    Curved-space electric constraint cleaning.

    This solves

        D_iD^i phi = C_E,

    where

        C_E = D_iE^i - rho + g B^i D_i a.

    Then corrects

        E^i_new = E^i - D^i phi
                = E^i - gamma^{ij} partial_j phi.

    Currently implemented for 2D diagonal/conformally-flat metrics.
    """
    if state.E is None or state.B is None:
        raise ValueError("Curved electric cleaning requires E and B fields.")

    if grid.ndim != 2:
        raise NotImplementedError("Curved electric cleaning currently supports 2D.")

    # Apply a compatible boundary before measuring the constraint.
    if poisson_boundary == "dirichlet":
        pre_boundary = DirichletBoundary(value=dirichlet_value)
    elif poisson_boundary == "neumann":
        pre_boundary = NeumannBoundary()
    elif poisson_boundary == "outflow":
        pre_boundary = OutflowBoundary()
    else:
        raise ValueError(f"Unknown poisson_boundary: {poisson_boundary!r}")

    work = state.copy()
    pre_boundary.apply_state(work, grid)

    constraint_before = electric_constraint(
        state=work,
        t=t,
        grid=grid,
        geom=geom,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=include_axion_coupling,
    )

    l2_before, linf_before = constraint_norms(constraint_before, grid)
    mean_before = mean_over_interior(constraint_before, grid)

    phi, poisson_report = solve_curved_poisson_diagonal_2d(
        rhs_full=constraint_before,
        grid=grid,
        geom=geom,
        method=poisson_method,
        boundary=poisson_boundary,
        dirichlet_value=dirichlet_value,
        max_iterations=max_iterations,
        tolerance=tolerance,
        omega=omega,
        remove_mean_for_neumann=remove_mean_for_neumann,
    )

    grad_phi_up = scalar_gradient_contravariant(
        scalar=phi,
        grid=grid,
        geom=geom,
        order=numerics.derivative_order,
    )

    cleaned = work.copy()

    for i in range(grid.ndim):
        cleaned.E[i] -= grad_phi_up[i]

    pre_boundary.apply_state(cleaned, grid)

    constraint_after = electric_constraint(
        state=cleaned,
        t=t,
        grid=grid,
        geom=geom,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=include_axion_coupling,
    )

    l2_after, linf_after = constraint_norms(constraint_after, grid)
    mean_after = mean_over_interior(constraint_after, grid)

    report = CleaningReport(
        method=f"curved_{poisson_report.method}_{poisson_boundary}",
        mean_constraint_before=mean_before,
        l2_constraint_before=l2_before,
        linf_constraint_before=linf_before,
        mean_constraint_after=mean_after,
        l2_constraint_after=l2_after,
        linf_constraint_after=linf_after,
        poisson_zero_mode_removed=poisson_report.removed_mean,
        poisson_iterations=poisson_report.iterations,
        poisson_residual_linf=poisson_report.residual_linf,
        poisson_converged=poisson_report.converged,
    )

    return cleaned, report