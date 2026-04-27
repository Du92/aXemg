"""
Source models for charge density and spatial current.

The Maxwell sector uses the decomposition

    J^mu = rho n^mu + j^mu,

where rho is the Eulerian charge density and j^i is the spatial current.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State
from axion_em_gr.geometry.base_metric import GeometryFields


class SourceModel:
    """
    Base class for electromagnetic sources.
    """

    def charge_density(
        self,
        t: float,
        grid: Grid,
        state: State,
        geom: GeometryFields,
    ) -> np.ndarray:
        """
        Return rho(t, x^i).
        """
        raise NotImplementedError

    def current_density(
        self,
        t: float,
        grid: Grid,
        state: State,
        geom: GeometryFields,
    ) -> np.ndarray:
        """
        Return j^i(t, x^j).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    def rho(
        self,
        t: float,
        grid: Grid,
        state: State,
        geom: GeometryFields,
    ) -> np.ndarray:
        return self.charge_density(t=t, grid=grid, state=state, geom=geom)

    def current(
        self,
        t: float,
        grid: Grid,
        state: State,
        geom: GeometryFields,
    ) -> np.ndarray:
        return self.current_density(t=t, grid=grid, state=state, geom=geom)


@dataclass
class VacuumSources(SourceModel):
    """
    Vacuum source model:

        rho = 0,
        j^i = 0.
    """

    def charge_density(
        self,
        t: float,
        grid: Grid,
        state: State,
        geom: GeometryFields,
    ) -> np.ndarray:
        return np.zeros(grid.shape_full, dtype=float)

    def current_density(
        self,
        t: float,
        grid: Grid,
        state: State,
        geom: GeometryFields,
    ) -> np.ndarray:
        return grid.zeros_vector()


@dataclass
class ConstantChargeCurrentSources(SourceModel):
    """
    Constant charge density and spatial current.

    Useful for tests.
    """

    rho0: float = 0.0
    j0: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def charge_density(
        self,
        t: float,
        grid: Grid,
        state: State,
        geom: GeometryFields,
    ) -> np.ndarray:
        rho = np.zeros(grid.shape_full, dtype=float)
        rho[...] = self.rho0
        return rho

    def current_density(
        self,
        t: float,
        grid: Grid,
        state: State,
        geom: GeometryFields,
    ) -> np.ndarray:
        j = grid.zeros_vector()

        for i in range(3):
            j[i] = self.j0[i]

        return j


@dataclass
class GaussianChargeSource2D(SourceModel):
    """
    Static Gaussian charge density in 2D with zero current.

    This is optional and mainly useful for later source tests.
    """

    amplitude: float = 1.0
    center: tuple[float, float] = (0.0, 0.0)
    width: tuple[float, float] = (10.0, 10.0)

    def charge_density(
        self,
        t: float,
        grid: Grid,
        state: State,
        geom: GeometryFields,
    ) -> np.ndarray:
        if grid.ndim != 2:
            raise NotImplementedError("GaussianChargeSource2D requires a 2D grid.")

        X, Y = grid.coordinates_2d()
        x0, y0 = self.center
        sx, sy = self.width

        rho = self.amplitude * np.exp(
            -0.5 * (((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2)
        )

        return rho

    def current_density(
        self,
        t: float,
        grid: Grid,
        state: State,
        geom: GeometryFields,
    ) -> np.ndarray:
        return grid.zeros_vector()