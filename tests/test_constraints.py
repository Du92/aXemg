"""
Tests for magnetic and electric constraints.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.boundary import PeriodicBoundary
from axion_em_gr.core.state import State
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.physics.constraints import (
    constraint_norms,
    electric_constraint,
    magnetic_constraint,
)
from axion_em_gr.physics.sources import VacuumSources


def test_magnetic_constraint_uniform_B_is_zero(grid_2d_periodic, numerics):
    grid = grid_2d_periodic
    geom = FlatMetric().evaluate(0.0, grid)

    a = np.zeros(grid.shape_full)
    Pi = np.zeros(grid.shape_full)

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    B[0] = 1.0
    B[1] = -0.5
    B[2] = 0.2

    state = State(a=a, Pi=Pi, E=E, B=B)

    C_B = magnetic_constraint(
        state=state,
        grid=grid,
        geom=geom,
        numerics=numerics,
    )

    l2, linf = constraint_norms(C_B, grid)

    assert l2 < 1.0e-12
    assert linf < 1.0e-12


def test_electric_constraint_uniform_E_no_axion_is_zero(
    grid_2d_periodic,
    numerics,
    physical,
):
    grid = grid_2d_periodic
    geom = FlatMetric().evaluate(0.0, grid)

    a = np.zeros(grid.shape_full)
    Pi = np.zeros(grid.shape_full)

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    E[0] = 1.0
    E[1] = 0.3
    B[2] = 1.0

    state = State(a=a, Pi=Pi, E=E, B=B)

    C_E = electric_constraint(
        state=state,
        t=0.0,
        grid=grid,
        geom=geom,
        sources=VacuumSources(),
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
    )

    l2, linf = constraint_norms(C_E, grid)

    assert l2 < 1.0e-12
    assert linf < 1.0e-12


def test_electric_constraint_detects_axion_gradient(
    grid_2d_periodic,
    numerics,
    physical,
):
    grid = grid_2d_periodic
    geom = FlatMetric().evaluate(0.0, grid)

    X, Y = grid.coordinates_2d()

    a = np.sin(X)
    Pi = np.zeros(grid.shape_full)

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    B[0] = 1.0
    B[1] = 0.0
    B[2] = 0.0

    for arr in [a, Pi, E[0], E[1], E[2], B[0], B[1], B[2]]:
        PeriodicBoundary().apply_array(arr, grid)

    state = State(a=a, Pi=Pi, E=E, B=B)

    C_E = electric_constraint(
        state=state,
        t=0.0,
        grid=grid,
        geom=geom,
        sources=VacuumSources(),
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
    )

    expected = physical.g_agamma * np.cos(X)

    interior = grid.interior_slices
    error = np.max(np.abs(C_E[interior] - expected[interior]))

    assert error < 1.0e-2