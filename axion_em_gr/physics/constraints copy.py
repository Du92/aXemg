"""
Constraint diagnostics.

Supports:
- 1D 3+1 constraints from previous phases,
- 2D flat Cartesian constraints for Phase 7.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.derivatives import (
    covariant_divergence_vector_1d,
    divergence_flat,
    gradient_scalar_flat,
    partial_derivative,
)
from axion_em_gr.core.grid import Grid
from axion_em_gr.core.parameters import NumericalParameters, PhysicalParameters
from axion_em_gr.core.state import State
from axion_em_gr.geometry.base_metric import GeometryFields
from axion_em_gr.physics.sources import SourceModel


def magnetic_constraint_flat_nd(
    state: State,
    grid: Grid,
    numerics: NumericalParameters,
) -> np.ndarray:
    """
    Flat Cartesian magnetic constraint:

        C_B = ∂_i B^i.
    """
    if state.B is None:
        raise ValueError("Magnetic constraint requires B field.")

    return divergence_flat(
        state.B,
        grid=grid,
        order=numerics.derivative_order,
    )


def electric_constraint_flat_nd(
    state: State,
    t: float,
    grid: Grid,
    geom: GeometryFields,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters | None = None,
    include_axion_coupling: bool = False,
) -> np.ndarray:
    """
    Flat Cartesian electric Gauss constraint:

        C_E = ∂_i E^i - rho + g B^i ∂_i a.
    """
    if state.E is None:
        raise ValueError("Electric constraint requires E field.")

    div_E = divergence_flat(
        state.E,
        grid=grid,
        order=numerics.derivative_order,
    )

    rho = sources.rho(
        state=state,
        t=t,
        grid=grid,
        geom=geom,
    )

    constraint = div_E - rho

    if include_axion_coupling:
        if physical is None:
            raise ValueError("Physical parameters are required for axion constraint.")

        if state.B is None:
            raise ValueError("Axion Gauss constraint requires B field.")

        grad_a = gradient_scalar_flat(
            state.a,
            grid=grid,
            order=numerics.derivative_order,
        )

        B_dot_grad_a = np.zeros_like(state.a)

        for i in range(grid.ndim):
            B_dot_grad_a += state.B[i] * grad_a[i]

        constraint += physical.g_agamma * B_dot_grad_a

    return constraint


def magnetic_constraint_3p1_1d(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    numerics: NumericalParameters,
) -> np.ndarray:
    """
    Compute C_B = D_i B^i in 1D.
    """
    if grid.ndim != 1:
        raise NotImplementedError("magnetic_constraint_3p1_1d only supports 1D.")

    if state.B is None:
        raise ValueError("Magnetic constraint requires B field.")

    return covariant_divergence_vector_1d(
        state.B,
        grid=grid,
        geom=geom,
        order=numerics.derivative_order,
    )


def electric_constraint_3p1_1d(
    state: State,
    t: float,
    grid: Grid,
    geom: GeometryFields,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters | None = None,
    include_axion_coupling: bool = False,
) -> np.ndarray:
    """
    Compute electric Gauss constraint in 1D 3+1 form.
    """
    if grid.ndim != 1:
        raise NotImplementedError("electric_constraint_3p1_1d only supports 1D.")

    if state.E is None:
        raise ValueError("Electric constraint requires E field.")

    div_E = covariant_divergence_vector_1d(
        state.E,
        grid=grid,
        geom=geom,
        order=numerics.derivative_order,
    )

    rho = sources.rho(
        state=state,
        t=t,
        grid=grid,
        geom=geom,
    )

    constraint = div_E - rho

    if include_axion_coupling:
        if physical is None:
            raise ValueError("Physical parameters are required for axion constraint.")

        if state.B is None:
            raise ValueError("Axion Gauss constraint requires B field.")

        da_dx = partial_derivative(
            state.a,
            grid=grid,
            axis=0,
            order=numerics.derivative_order,
        )

        constraint += physical.g_agamma * state.B[0] * da_dx

    return constraint


def magnetic_constraint(
    state: State,
    grid: Grid,
    geom: GeometryFields,
    numerics: NumericalParameters,
) -> np.ndarray:
    """
    Dispatch magnetic constraint.
    """
    if grid.ndim == 1:
        return magnetic_constraint_3p1_1d(
            state=state,
            grid=grid,
            geom=geom,
            numerics=numerics,
        )

    if grid.ndim == 2:
        return magnetic_constraint_flat_nd(
            state=state,
            grid=grid,
            numerics=numerics,
        )

    raise NotImplementedError("Magnetic constraint supports 1D and 2D.")


def electric_constraint(
    state: State,
    t: float,
    grid: Grid,
    geom: GeometryFields,
    sources: SourceModel,
    numerics: NumericalParameters,
    physical: PhysicalParameters | None = None,
    include_axion_coupling: bool = False,
) -> np.ndarray:
    """
    Dispatch electric constraint.
    """
    if grid.ndim == 1:
        return electric_constraint_3p1_1d(
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
        return electric_constraint_flat_nd(
            state=state,
            t=t,
            grid=grid,
            geom=geom,
            sources=sources,
            numerics=numerics,
            physical=physical,
            include_axion_coupling=include_axion_coupling,
        )

    raise NotImplementedError("Electric constraint supports 1D and 2D.")


def constraint_norms(
    constraint: np.ndarray,
    grid: Grid,
) -> tuple[float, float]:
    """
    Return L2 and Linf norms of a constraint over the interior domain.
    """
    interior = grid.interior_slices

    if grid.ndim == 1:
        volume_element = grid.dx[0]
    elif grid.ndim == 2:
        volume_element = grid.dx[0] * grid.dx[1]
    else:
        raise NotImplementedError("constraint_norms supports 1D and 2D.")

    c = constraint[interior]

    l2 = float(np.sqrt(np.sum(c**2) * volume_element))
    linf = float(np.max(np.abs(c)))

    return l2, linf


# Backward-compatible aliases.
magnetic_constraint_flat_1d = magnetic_constraint_3p1_1d
electric_constraint_flat_1d = electric_constraint_3p1_1d