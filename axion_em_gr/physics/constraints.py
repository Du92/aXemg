"""
Constraint equations.

Magnetic constraint:

    C_B = D_i B^i.

Electric axion-modified Gauss constraint:

    C_E = D_i E^i - rho + g B^i D_i a.

Equivalently, the physical constraint is

    D_i E^i = rho - g B^i D_i a.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.covariant_derivatives import (
    divergence_vector_covariant,
    scalar_gradient_covariant,
)
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.state import State
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.physics.sources import SourceModel


def magnetic_constraint(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    numerics: NumericalParameters,
) -> np.ndarray:
    """
    Compute magnetic constraint:

        C_B = D_i B^i.
    """
    if state.B is None:
        return np.zeros_like(state.a)

    return divergence_vector_covariant(
        vector_up=state.B,
        grid=grid,
        geom=geom,
        order=numerics.derivative_order,
    )


def electric_constraint(
    state: State,
    t: float,
    grid: Grid,
    geom: GeometryFields,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters,
    include_axion_coupling: bool = True,
) -> np.ndarray:
    """
    Compute electric Gauss constraint residual:

        C_E = D_i E^i - rho + g B^i D_i a.

    A perfectly constrained state has C_E = 0.
    """
    if state.E is None:
        return np.zeros_like(state.a)

    div_E = divergence_vector_covariant(
        vector_up=state.E,
        grid=grid,
        geom=geom,
        order=numerics.derivative_order,
    )

    rho = sources.charge_density(
        t=t,
        grid=grid,
        state=state,
        geom=geom,
    )

    constraint = div_E - rho

    if include_axion_coupling and state.B is not None:
        grad_a = scalar_gradient_covariant(
            scalar=state.a,
            grid=grid,
            order=numerics.derivative_order,
        )

        B_grad_a = np.zeros_like(state.a)

        for i in range(grid.ndim):
            B_grad_a += state.B[i] * grad_a[i]

        constraint += physical.g_agamma * B_grad_a

    return constraint


def constraint_norms(
    constraint: np.ndarray,
    grid: Grid,
) -> tuple[float, float]:
    """
    Return L2 and Linf norms over the physical interior.
    """
    interior = grid.interior_slices

    if grid.ndim == 1:
        volume_element = grid.dx[0]
    elif grid.ndim == 2:
        volume_element = grid.dx[0] * grid.dx[1]
    else:
        raise NotImplementedError("constraint_norms supports only 1D and 2D.")

    data = constraint[interior]

    l2 = float(np.sqrt(np.sum(data**2) * volume_element))
    linf = float(np.max(np.abs(data)))

    return l2, linf


# ---------------------------------------------------------------------------
# Backward-compatible aliases from older phases
# ---------------------------------------------------------------------------

def magnetic_constraint_3p1_1d(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    numerics: NumericalParameters,
) -> np.ndarray:
    return magnetic_constraint(
        state=state,
        grid=grid,
        geom=geom,
        numerics=numerics,
    )


def electric_constraint_3p1_1d(
    state: State,
    t: float,
    grid: Grid,
    geom: GeometryFields,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters,
    include_axion_coupling: bool = True,
) -> np.ndarray:
    return electric_constraint(
        state=state,
        t=t,
        grid=grid,
        geom=geom,
        sources=sources,
        numerics=numerics,
        physical=physical,
        include_axion_coupling=include_axion_coupling,
    )