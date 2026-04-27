"""
Utilities for extracting 1D slices from 2D fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axion_em_gr.core.grid import Grid


@dataclass
class Slice1D:
    """
    A one-dimensional slice extracted from a 2D field.
    """

    coordinate: np.ndarray
    values: np.ndarray
    fixed_axis: str
    fixed_value: float
    index: int


def nearest_index(values: np.ndarray, target: float) -> int:
    """
    Return index of the coordinate closest to target.
    """
    return int(np.argmin(np.abs(values - target)))


def extract_x_slice(
    grid: Grid,
    field: np.ndarray,
    y_value: float = 0.0,
) -> Slice1D:
    """
    Extract field(x, y=y_value).

    Returns an x-directed slice.
    """
    if grid.ndim != 2:
        raise ValueError("extract_x_slice requires a 2D grid.")

    x = grid.axis_coordinates(axis=0)
    y = grid.axis_coordinates(axis=1)

    g = grid.nghost
    nx, ny = grid.shape

    j = nearest_index(y[g:g + ny], y_value) + g

    x_int = x[g:g + nx]
    values = field[g:g + nx, j]

    return Slice1D(
        coordinate=x_int,
        values=values,
        fixed_axis="y",
        fixed_value=float(y[j]),
        index=j,
    )


def extract_y_slice(
    grid: Grid,
    field: np.ndarray,
    x_value: float = 0.0,
) -> Slice1D:
    """
    Extract field(x=x_value, y).

    Returns a y-directed slice.
    """
    if grid.ndim != 2:
        raise ValueError("extract_y_slice requires a 2D grid.")

    x = grid.axis_coordinates(axis=0)
    y = grid.axis_coordinates(axis=1)

    g = grid.nghost
    nx, ny = grid.shape

    i = nearest_index(x[g:g + nx], x_value) + g

    y_int = y[g:g + ny]
    values = field[i, g:g + ny]

    return Slice1D(
        coordinate=y_int,
        values=values,
        fixed_axis="x",
        fixed_value=float(x[i]),
        index=i,
    )