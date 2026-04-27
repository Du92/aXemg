"""
Ghost-zone utilities.

This module centralizes low-level ghost-zone filling operations for 1D and 2D
Cartesian grids.

The goal is to avoid duplicating slicing logic inside each boundary condition.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.grid import Grid


class GhostZoneManager:
    """
    Helper class for filling ghost zones of scalar arrays.

    It supports:
    - periodic filling,
    - constant Dirichlet filling,
    - zero-gradient / outflow filling,
    - linear extrapolation filling.

    Vector fields are handled by applying these scalar operations component by
    component in the BoundaryCondition classes.
    """

    def __init__(self, grid: Grid) -> None:
        self.grid = grid

    @property
    def g(self) -> int:
        return self.grid.nghost

    def apply_periodic(self, array: np.ndarray) -> None:
        """
        Periodic ghost-zone fill.
        """
        if self.grid.ndim == 1:
            self._periodic_1d(array)
        elif self.grid.ndim == 2:
            self._periodic_2d(array)
        else:
            raise NotImplementedError("Periodic ghost fill supports 1D and 2D.")

    def apply_dirichlet(self, array: np.ndarray, value: float = 0.0) -> None:
        """
        Fill ghost zones with a constant value.
        """
        if self.grid.ndim == 1:
            self._dirichlet_1d(array, value)
        elif self.grid.ndim == 2:
            self._dirichlet_2d(array, value)
        else:
            raise NotImplementedError("Dirichlet ghost fill supports 1D and 2D.")

    def apply_neumann(self, array: np.ndarray) -> None:
        """
        Zero-gradient ghost-zone fill.

        This is also the simplest outflow approximation.
        """
        if self.grid.ndim == 1:
            self._neumann_1d(array)
        elif self.grid.ndim == 2:
            self._neumann_2d(array)
        else:
            raise NotImplementedError("Neumann ghost fill supports 1D and 2D.")

    def apply_outflow(self, array: np.ndarray) -> None:
        """
        Alias for zero-gradient outflow.
        """
        self.apply_neumann(array)

    def apply_linear_extrapolation(self, array: np.ndarray) -> None:
        """
        Linear extrapolation into ghost zones.

        This can reduce reflections for smooth outgoing profiles compared to
        pure zero-gradient boundaries, but it can also be less robust if the
        solution becomes noisy.
        """
        if self.grid.ndim == 1:
            self._linear_extrapolation_1d(array)
        elif self.grid.ndim == 2:
            self._linear_extrapolation_2d(array)
        else:
            raise NotImplementedError(
                "Linear extrapolation ghost fill supports 1D and 2D."
            )

    # ------------------------------------------------------------------
    # 1D implementations
    # ------------------------------------------------------------------

    def _periodic_1d(self, array: np.ndarray) -> None:
        g = self.g
        n = self.grid.shape[0]

        array[:g] = array[n:n + g]
        array[g + n:] = array[g:g + g]

    def _dirichlet_1d(self, array: np.ndarray, value: float) -> None:
        g = self.g
        n = self.grid.shape[0]

        array[:g] = value
        array[g + n:] = value

    def _neumann_1d(self, array: np.ndarray) -> None:
        g = self.g
        n = self.grid.shape[0]

        array[:g] = array[g]
        array[g + n:] = array[g + n - 1]

    def _linear_extrapolation_1d(self, array: np.ndarray) -> None:
        g = self.g
        n = self.grid.shape[0]

        left0 = array[g]
        left1 = array[g + 1]
        right0 = array[g + n - 1]
        right1 = array[g + n - 2]

        left_slope = left0 - left1
        right_slope = right0 - right1

        for k in range(1, g + 1):
            array[g - k] = left0 + k * left_slope
            array[g + n - 1 + k] = right0 + k * right_slope

    # ------------------------------------------------------------------
    # 2D implementations
    # ------------------------------------------------------------------

    def _periodic_2d(self, array: np.ndarray) -> None:
        g = self.g
        nx, ny = self.grid.shape

        # x ghost zones
        array[:g, :] = array[nx:nx + g, :]
        array[g + nx:, :] = array[g:g + g, :]

        # y ghost zones
        array[:, :g] = array[:, ny:ny + g]
        array[:, g + ny:] = array[:, g:g + g]

    def _dirichlet_2d(self, array: np.ndarray, value: float) -> None:
        g = self.g
        nx, ny = self.grid.shape

        array[:g, :] = value
        array[g + nx:, :] = value
        array[:, :g] = value
        array[:, g + ny:] = value

    def _neumann_2d(self, array: np.ndarray) -> None:
        g = self.g
        nx, ny = self.grid.shape

        # x boundaries
        array[:g, :] = array[g:g + 1, :]
        array[g + nx:, :] = array[g + nx - 1:g + nx, :]

        # y boundaries
        array[:, :g] = array[:, g:g + 1]
        array[:, g + ny:] = array[:, g + ny - 1:g + ny]

    def _linear_extrapolation_2d(self, array: np.ndarray) -> None:
        g = self.g
        nx, ny = self.grid.shape

        # First extrapolate in x.
        left0 = array[g, :].copy()
        left1 = array[g + 1, :].copy()
        right0 = array[g + nx - 1, :].copy()
        right1 = array[g + nx - 2, :].copy()

        left_slope = left0 - left1
        right_slope = right0 - right1

        for k in range(1, g + 1):
            array[g - k, :] = left0 + k * left_slope
            array[g + nx - 1 + k, :] = right0 + k * right_slope

        # Then extrapolate in y, including corners already filled in x.
        bottom0 = array[:, g].copy()
        bottom1 = array[:, g + 1].copy()
        top0 = array[:, g + ny - 1].copy()
        top1 = array[:, g + ny - 2].copy()

        bottom_slope = bottom0 - bottom1
        top_slope = top0 - top1

        for k in range(1, g + 1):
            array[:, g - k] = bottom0 + k * bottom_slope
            array[:, g + ny - 1 + k] = top0 + k * top_slope