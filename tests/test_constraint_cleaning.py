"""
Tests for electric constraint cleaning.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.state import State
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.physics.constraint_cleaning import (
    clean_electric_constraint_curved,
    clean_electric_constraint_flat,
)
from axion_em_gr.physics.sources import VacuumSources


def _make_inconsistent_state_2d(grid, g_agamma: float):
    X, Y = grid.coordinates_2d()

    a = np.exp(-0.5 * (X**2 + Y**2) / 10.0**2)
    Pi = np.zeros(grid.shape_full)

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    # In-plane B makes g B^i partial_i a contribute.
    B[0] = 1.0
    B[1] = 0.5
    B[2] = 0.0

    # E initially zero, so Gauss constraint is not solved.
    E[0] = 0.0
    E[1] = 0.0
    E[2] = 0.0

    return State(a=a, Pi=Pi, E=E, B=B)


def test_flat_constraint_cleaning_reduces_l2_constraint(
    grid_2d_compact,
    numerics,
    physical,
):
    grid = grid_2d_compact
    geom = FlatMetric().evaluate(0.0, grid)

    state = _make_inconsistent_state_2d(grid, physical.g_agamma)

    cleaned, report = clean_electric_constraint_flat(
        state=state,
        t=0.0,
        grid=grid,
        geom=geom,
        sources=VacuumSources(),
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
        poisson_solver="jacobi",
        poisson_boundary="dirichlet",
        dirichlet_value=0.0,
        max_iterations=20_000,
        tolerance=1.0e-6,
        omega=2.0 / 3.0,
    )

    assert report.poisson_converged
    assert report.l2_constraint_after < report.l2_constraint_before

    # Demand a meaningful reduction, not exact discrete projection.
    reduction = report.l2_constraint_after / report.l2_constraint_before
    assert reduction < 0.75


def test_curved_constraint_cleaning_reduces_l2_constraint(
    grid_2d_compact,
    numerics,
    physical,
    compact_metric_2d,
):
    grid = grid_2d_compact
    geom = compact_metric_2d.evaluate(0.0, grid)

    state = _make_inconsistent_state_2d(grid, physical.g_agamma)

    cleaned, report = clean_electric_constraint_curved(
        state=state,
        t=0.0,
        grid=grid,
        geom=geom,
        sources=VacuumSources(),
        numerics=numerics,
        physical=physical,
        include_axion_coupling=True,
        poisson_method="jacobi",
        poisson_boundary="dirichlet",
        dirichlet_value=0.0,
        max_iterations=30_000,
        tolerance=1.0e-6,
        omega=2.0 / 3.0,
    )

    assert report.poisson_converged
    assert report.l2_constraint_after < report.l2_constraint_before

    # Curved cleaning uses a metric-compatible Poisson solver, but the
    # correction and diagnostic still use separate discrete operators. Require
    # reduction, not machine-precision annihilation.
    reduction = report.l2_constraint_after / report.l2_constraint_before
    assert reduction < 0.9