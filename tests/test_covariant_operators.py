"""
Tests for covariant spatial operators.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.boundary import PeriodicBoundary, OutflowBoundary
from axion_em_gr.core.covariant_derivatives import (
    divergence_vector_covariant,
    scalar_gradient_covariant,
    scalar_laplacian_covariant,
)
from axion_em_gr.geometry.flat import FlatMetric

from tests.helpers import core_slices


def test_covariant_gradient_scalar_2d(grid_2d_periodic):
    grid = grid_2d_periodic
    X, Y = grid.coordinates_2d()

    f = np.sin(X) * np.cos(Y)

    PeriodicBoundary().apply_array(f, grid)

    grad = scalar_gradient_covariant(
        scalar=f,
        grid=grid,
        order=2,
    )

    expected_x = np.cos(X) * np.cos(Y)
    expected_y = -np.sin(X) * np.sin(Y)

    core = core_slices(grid, margin=4)

    err_x = np.max(np.abs(grad[0][core] - expected_x[core]))
    err_y = np.max(np.abs(grad[1][core] - expected_y[core]))

    assert err_x < 8.0e-3
    assert err_y < 8.0e-3


def test_flat_covariant_laplacian_matches_analytic_2d(grid_2d_periodic):
    """
    In flat space, D_iD^i f should reduce to the ordinary Laplacian.

    We check the deep interior because scalar_laplacian_covariant is a
    composed operator and the largest errors are located near ghost-zone
    interfaces.
    """
    grid = grid_2d_periodic
    metric = FlatMetric()
    geom = metric.evaluate(0.0, grid)

    X, Y = grid.coordinates_2d()

    f = np.sin(X) * np.cos(Y)
    expected = -2.0 * np.sin(X) * np.cos(Y)

    PeriodicBoundary().apply_array(f, grid)

    lap = scalar_laplacian_covariant(
        scalar=f,
        grid=grid,
        geom=geom,
        order=2,
    )

    core = core_slices(grid, margin=6)
    error = np.max(np.abs(lap[core] - expected[core]))

    assert error < 3.0e-2


def test_covariant_divergence_flat_2d(grid_2d_periodic):
    grid = grid_2d_periodic
    metric = FlatMetric()
    geom = metric.evaluate(0.0, grid)

    X, Y = grid.coordinates_2d()

    V = grid.zeros_vector()
    V[0] = np.sin(X)
    V[1] = np.cos(Y)
    V[2] = 0.0

    for i in range(3):
        PeriodicBoundary().apply_array(V[i], grid)

    expected = np.cos(X) - np.sin(Y)

    div = divergence_vector_covariant(
        vector_up=V,
        grid=grid,
        geom=geom,
        order=2,
    )

    core = core_slices(grid, margin=4)
    error = np.max(np.abs(div[core] - expected[core]))

    assert error < 1.5e-2


def test_curved_laplacian_constant_is_zero(grid_2d_compact, compact_metric_2d):
    grid = grid_2d_compact
    geom = compact_metric_2d.evaluate(0.0, grid)

    f = np.ones(grid.shape_full)

    OutflowBoundary().apply_array(f, grid)

    lap = scalar_laplacian_covariant(
        scalar=f,
        grid=grid,
        geom=geom,
        order=2,
    )

    interior = grid.interior_slices
    error = np.max(np.abs(lap[interior]))

    assert error < 1.0e-10