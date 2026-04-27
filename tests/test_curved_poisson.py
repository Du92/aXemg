"""
Tests for curved Poisson solver.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.boundary import DirichletBoundary
from axion_em_gr.geometry.flat import FlatMetric
from axion_em_gr.physics.curved_poisson import (
    curved_poisson_residual_diagonal_2d,
    solve_curved_poisson_diagonal_2d,
)


def test_curved_poisson_flat_dirichlet_manufactured_solution(grid_2d_compact):
    """
    Manufactured solution on flat metric:

        phi = sin(pi xi) sin(pi eta)

    We build the RHS from the same discrete operator and verify that the solver
    approximately recovers the manufactured solution.
    """
    grid = grid_2d_compact
    geom = FlatMetric().evaluate(0.0, grid)

    X, Y = grid.coordinates_2d()
    (xmin, xmax), (ymin, ymax) = grid.bounds

    xi = (X - xmin) / (xmax - xmin)
    eta = (Y - ymin) / (ymax - ymin)

    phi_exact = np.sin(np.pi * xi) * np.sin(np.pi * eta)

    DirichletBoundary(value=0.0).apply_array(phi_exact, grid)

    from axion_em_gr.physics.curved_poisson import curved_laplacian_diagonal_2d

    rhs = curved_laplacian_diagonal_2d(
        phi=phi_exact,
        grid=grid,
        geom=geom,
    )

    phi, report = solve_curved_poisson_diagonal_2d(
        rhs_full=rhs,
        grid=grid,
        geom=geom,
        method="jacobi",
        boundary="dirichlet",
        dirichlet_value=0.0,
        max_iterations=25_000,
        tolerance=5.0e-6,
        omega=2.0 / 3.0,
    )

    interior = grid.interior_slices

    error = np.max(np.abs(phi[interior] - phi_exact[interior]))

    assert report.residual_linf < 1.0e-5
    assert error < 8.0e-3


def test_curved_poisson_residual_after_solve_is_small(grid_2d_compact, compact_metric_2d):
    grid = grid_2d_compact
    geom = compact_metric_2d.evaluate(0.0, grid)

    X, Y = grid.coordinates_2d()

    rhs = np.exp(-0.5 * (X**2 + Y**2) / 15.0**2)

    phi, report = solve_curved_poisson_diagonal_2d(
        rhs_full=rhs,
        grid=grid,
        geom=geom,
        method="jacobi",
        boundary="dirichlet",
        dirichlet_value=0.0,
        max_iterations=35_000,
        tolerance=5.0e-5,
        omega=2.0 / 3.0,
    )

    residual = curved_poisson_residual_diagonal_2d(
        phi=phi,
        rhs=rhs,
        grid=grid,
        geom=geom,
    )

    interior = grid.interior_slices
    residual_linf = np.max(np.abs(residual[interior]))

    assert residual_linf < 7.0e-5