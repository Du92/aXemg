"""
Grid infrastructure for finite-difference evolutions.

Supports 1D and 2D Cartesian grids with ghost zones.
The design remains compatible with future 3D extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Grid:
    """
    Cartesian numerical grid with ghost zones.

    Parameters
    ----------
    ndim:
        Number of spatial dimensions.
    shape:
        Number of interior points in each spatial direction.
        Example 1D: (Nx,)
        Example 2D: (Nx, Ny)
    bounds:
        Physical bounds of the interior domain.
        Example 1D: ((xmin, xmax),)
        Example 2D: ((xmin, xmax), (ymin, ymax))
    nghost:
        Number of ghost cells on each side.
    """

    ndim: int
    shape: Tuple[int, ...]
    bounds: Tuple[Tuple[float, float], ...]
    nghost: int = 3

    def __post_init__(self) -> None:
        if self.ndim != len(self.shape):
            raise ValueError("ndim must match len(shape).")

        if self.ndim != len(self.bounds):
            raise ValueError("ndim must match len(bounds).")

        if self.ndim < 1 or self.ndim > 3:
            raise ValueError("Only ndim=1, 2, or 3 are supported.")

        if self.nghost < 1:
            raise ValueError("nghost must be at least 1.")

        for n in self.shape:
            if n <= 0:
                raise ValueError("All grid dimensions must be positive.")

        for xmin, xmax in self.bounds:
            if xmax <= xmin:
                raise ValueError("Each bound must satisfy xmax > xmin.")

    @property
    def dx(self) -> Tuple[float, ...]:
        """
        Grid spacings for the interior domain.

        We use endpoint=False convention, convenient for periodic grids.
        """
        return tuple(
            (xmax - xmin) / n
            for n, (xmin, xmax) in zip(self.shape, self.bounds)
        )

    @property
    def shape_full(self) -> Tuple[int, ...]:
        """
        Full array shape including ghost zones.
        """
        return tuple(n + 2 * self.nghost for n in self.shape)

    @property
    def interior_slices(self) -> Tuple[slice, ...]:
        """
        Slices selecting the physical interior region.
        """
        g = self.nghost
        return tuple(slice(g, g + n) for n in self.shape)

    def axis_coordinates(self, axis: int) -> np.ndarray:
        """
        Return full coordinate array along a single axis, including ghost zones.
        """
        if axis < 0 or axis >= self.ndim:
            raise ValueError("Invalid axis.")

        xmin, xmax = self.bounds[axis]
        dx = self.dx[axis]
        n = self.shape[axis]
        g = self.nghost

        interior = xmin + dx * np.arange(n)
        left_ghosts = interior[0] - dx * np.arange(g, 0, -1)
        right_ghosts = interior[-1] + dx * np.arange(1, g + 1)

        return np.concatenate([left_ghosts, interior, right_ghosts])

    def coordinates_1d(self) -> np.ndarray:
        """
        Return the full 1D coordinate array including ghost zones.
        """
        if self.ndim != 1:
            raise NotImplementedError("coordinates_1d is only valid for ndim=1.")

        return self.axis_coordinates(axis=0)

    def coordinates_2d(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return full 2D coordinate arrays X, Y including ghost zones.

        Shapes:

            X.shape == grid.shape_full
            Y.shape == grid.shape_full
        """
        if self.ndim != 2:
            raise NotImplementedError("coordinates_2d is only valid for ndim=2.")

        x = self.axis_coordinates(axis=0)
        y = self.axis_coordinates(axis=1)

        return np.meshgrid(x, y, indexing="ij")

    def interior_view(self, array: np.ndarray) -> np.ndarray:
        """
        Return the interior view of a scalar grid array.
        """
        return array[self.interior_slices]

    def zeros_scalar(self) -> np.ndarray:
        """
        Allocate a scalar field on the full grid.
        """
        return np.zeros(self.shape_full, dtype=float)

    def zeros_vector(self) -> np.ndarray:
        """
        Allocate a 3-component vector field on the full grid.

        Even in 1D or 2D, vectors keep three components.
        """
        return np.zeros((3, *self.shape_full), dtype=float)