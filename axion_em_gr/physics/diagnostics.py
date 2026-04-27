"""
Diagnostics for axion and electromagnetic fields.

Supports:
- flat 1D diagnostics from earlier phases,
- flat 2D energy-density and Poynting diagnostics for Phase 8.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.derivatives import gradient_scalar_flat, partial_derivative
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State
from axion_em_gr.core.tensors import contract_cov_contra, lower_vector
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.physics.potentials import AxionPotential

from axion_em_gr.core.covariant_derivatives import scalar_gradient_covariant


def axion_energy_density_flat(
    state: State,
    grid: Grid,
    potential: AxionPotential,
) -> np.ndarray:
    """
    Flat-space axion energy density:

        rho_a = 1/2 Pi^2 + 1/2 |grad a|^2 + V(a).

    In 2D:

        |grad a|^2 = (partial_x a)^2 + (partial_y a)^2.
    """
    grad_a = gradient_scalar_flat(state.a, grid)

    grad_norm_sq = np.zeros_like(state.a)

    for i in range(grid.ndim):
        grad_norm_sq += grad_a[i] ** 2

    rho_a = 0.5 * state.Pi**2 + 0.5 * grad_norm_sq + potential.V(state.a)

    return rho_a


def electromagnetic_energy_density_flat(
    state: State,
    geom: GeometryFields,
) -> np.ndarray:
    """
    Flat or geometry-aware electromagnetic energy density:

        rho_EM = 1/2 (E_i E^i + B_i B^i).

    In flat space this is simply:

        rho_EM = 1/2 (|E|^2 + |B|^2).
    """
    if state.E is None or state.B is None:
        return np.zeros_like(state.a)

    E_down = lower_vector(state.E, geom.gamma_down)
    B_down = lower_vector(state.B, geom.gamma_down)

    E2 = contract_cov_contra(E_down, state.E)
    B2 = contract_cov_contra(B_down, state.B)

    return 0.5 * (E2 + B2)


def edotb_density(
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


def poynting_vector_flat(
    state: State,
) -> np.ndarray:
    """
    Flat-space Poynting vector:

        S^i = epsilon^{ijk} E_j B_k.

    In flat Cartesian coordinates E_j = E^j and B_k = B^k.

    Returns
    -------
    S:
        Array with shape (3, *grid.shape_full).
    """
    if state.E is None or state.B is None:
        raise ValueError("Poynting vector requires E and B fields.")

    E = state.E
    B = state.B

    S = np.zeros_like(E)

    S[0] = E[1] * B[2] - E[2] * B[1]
    S[1] = E[2] * B[0] - E[0] * B[2]
    S[2] = E[0] * B[1] - E[1] * B[0]

    return S


def poynting_magnitude_flat(
    state: State,
) -> np.ndarray:
    """
    Magnitude of the flat-space Poynting vector.
    """
    S = poynting_vector_flat(state)

    return np.sqrt(S[0] ** 2 + S[1] ** 2 + S[2] ** 2)


def integrate_scalar_density(
    density: np.ndarray,
    grid: Grid,
) -> float:
    """
    Integrate a scalar density over the physical domain.

    1D:
        integral density dx

    2D:
        integral density dx dy
    """
    interior = grid.interior_slices

    if grid.ndim == 1:
        volume_element = grid.dx[0]
    elif grid.ndim == 2:
        volume_element = grid.dx[0] * grid.dx[1]
    else:
        raise NotImplementedError("Only 1D and 2D integrations are supported.")

    return float(np.sum(density[interior]) * volume_element)


def axion_energy_flat(
    state: State,
    grid: Grid,
    potential: AxionPotential,
) -> float:
    """
    Integrated flat-space axion energy.
    """
    rho_a = axion_energy_density_flat(
        state=state,
        grid=grid,
        potential=potential,
    )

    return integrate_scalar_density(rho_a, grid)


def electromagnetic_energy_flat(
    state: State,
    grid: Grid,
    geom: GeometryFields,
) -> float:
    """
    Integrated electromagnetic energy.
    """
    rho_em = electromagnetic_energy_density_flat(
        state=state,
        geom=geom,
    )

    return integrate_scalar_density(rho_em, grid)


def radial_flux_density_2d(
    state: State,
    grid: Grid,
    center: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Approximate radial Poynting flux in 2D:

        S_r = S_x n_x + S_y n_y,

    where n is the radial unit vector in the x-y plane.

    This is useful for visualizing outward/inward EM energy transport.
    """
    if grid.ndim != 2:
        raise ValueError("radial_flux_density_2d requires a 2D grid.")

    X, Y = grid.coordinates_2d()
    x0, y0 = center

    rx = X - x0
    ry = Y - y0

    r = np.sqrt(rx**2 + ry**2)

    eps = 1.0e-14
    nx = rx / np.maximum(r, eps)
    ny = ry / np.maximum(r, eps)

    S = poynting_vector_flat(state)

    return S[0] * nx + S[1] * ny


# ---------------------------------------------------------------------------
# Backward-compatible names from earlier phases.
# ---------------------------------------------------------------------------

def electromagnetic_energy_flat_1d(
    state: State,
    grid: Grid,
) -> float:
    """
    Backward-compatible flat 1D electromagnetic energy proxy.
    """
    if state.E is None or state.B is None:
        return 0.0

    interior = grid.interior_slices
    dx = grid.dx[0]

    E_int = state.E[(slice(None), *interior)]
    B_int = state.B[(slice(None), *interior)]

    density = 0.5 * (
        np.sum(E_int**2, axis=0)
        +
        np.sum(B_int**2, axis=0)
    )

    return float(np.sum(density) * dx)


def axion_energy_flat_1d(
    state: State,
    grid: Grid,
    mass: float,
) -> float:
    """
    Backward-compatible flat-space axion energy proxy for massive potential.
    """
    interior = grid.interior_slices
    dx = grid.dx[0]

    da_dx = partial_derivative(state.a, grid)

    density = 0.5 * (
        state.Pi[interior] ** 2
        +
        da_dx[interior] ** 2
        +
        mass**2 * state.a[interior] ** 2
    )

    return float(np.sum(density) * dx)

def axion_energy_density_geometry(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    potential: AxionPotential,
) -> np.ndarray:
    """
    Geometry-aware axion energy density proxy:

        rho_a = 1/2 Pi^2
              + 1/2 gamma^{ij} partial_i a partial_j a
              + V(a).

    This is the natural Eulerian local energy density proxy for the scalar
    sector in the reduced code.
    """
    grad_a = scalar_gradient_covariant(
        scalar=state.a,
        grid=grid,
        order=2,
    )

    grad_norm = np.zeros_like(state.a)

    for i in range(grid.ndim):
        for j in range(grid.ndim):
            grad_norm += geom.gamma_up[i, j] * grad_a[i] * grad_a[j]

    return 0.5 * state.Pi**2 + 0.5 * grad_norm + potential.V(state.a)


def axion_energy_geometry(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    potential: AxionPotential,
) -> float:
    """
    Integrated geometry-aware axion energy.

    Uses sqrt(gamma) d^dx as volume weight over the computational domain.
    """
    rho = axion_energy_density_geometry(
        state=state,
        grid=grid,
        geom=geom,
        potential=potential,
    )

    interior = grid.interior_slices

    if grid.ndim == 1:
        coordinate_volume = grid.dx[0]
    elif grid.ndim == 2:
        coordinate_volume = grid.dx[0] * grid.dx[1]
    else:
        raise NotImplementedError("Only 1D/2D supported.")

    return float(np.sum(rho[interior] * geom.sqrt_gamma[interior]) * coordinate_volume)