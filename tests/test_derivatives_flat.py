"""
Tests for flat finite-difference derivatives.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.derivatives import partial_derivative
from axion_em_gr.core.boundary import PeriodicBoundary


def test_partial_derivative_1d_sine(grid_1d_periodic):
    grid = grid_1d_periodic
    x = grid.coordinates_1d()

    f = np.sin(x)
    expected = np.cos(x)

    PeriodicBoundary().apply_array(f, grid)

    df = partial_derivative(
        f,
        grid=grid,
        axis=0,
        order=2,
    )

    interior = grid.interior_slices

    error = np.max(np.abs(df[interior] - expected[interior]))

    assert error < 5.0e-3


def test_partial_derivative_2d_sine_x(grid_2d_periodic):
    grid = grid_2d_periodic
    X, Y = grid.coordinates_2d()

    f = np.sin(X) * np.cos(Y)
    expected_dx = np.cos(X) * np.cos(Y)

    PeriodicBoundary().apply_array(f, grid)

    df_dx = partial_derivative(
        f,
        grid=grid,
        axis=0,
        order=2,
    )

    interior = grid.interior_slices
    error = np.max(np.abs(df_dx[interior] - expected_dx[interior]))

    assert error < 8.0e-3


def test_partial_derivative_2d_sine_y(grid_2d_periodic):
    grid = grid_2d_periodic
    X, Y = grid.coordinates_2d()

    f = np.sin(X) * np.cos(Y)
    expected_dy = -np.sin(X) * np.sin(Y)

    PeriodicBoundary().apply_array(f, grid)

    df_dy = partial_derivative(
        f,
        grid=grid,
        axis=1,
        order=2,
    )

    interior = grid.interior_slices
    error = np.max(np.abs(df_dy[interior] - expected_dy[interior]))

    assert error < 8.0e-3