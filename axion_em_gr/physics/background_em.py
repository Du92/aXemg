"""
Prescribed time-dependent electromagnetic backgrounds.

Phase 17 introduces external EM fields such as a rotating magnetic dipole.
These fields are not evolved self-consistently by Maxwell's equations. They
are imposed as a background for the axion equation and diagnostics.

This is useful for reduced neutron-star-inspired scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axion_em_gr.core.grid import Grid


class BackgroundEMField:
    """
    Base class for prescribed electromagnetic backgrounds.
    """

    def evaluate(self, t: float, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
        """
        Return E^i and B^i arrays with shape (3, *grid.shape_full).
        """
        raise NotImplementedError


@dataclass
class NoBackgroundEM(BackgroundEMField):
    """
    Null background.
    """

    def evaluate(self, t: float, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
        return grid.zeros_vector(), grid.zeros_vector()


@dataclass
class UniformMagneticBackground(BackgroundEMField):
    """
    Uniform magnetic field with optional uniform electric field.
    """

    B0: tuple[float, float, float] = (0.0, 0.0, 1.0)
    E0: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def evaluate(self, t: float, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
        E = grid.zeros_vector()
        B = grid.zeros_vector()

        for i in range(3):
            E[i] = self.E0[i]
            B[i] = self.B0[i]

        return E, B


@dataclass
class RotatingDipoleBackground2D(BackgroundEMField):
    """
    Prescribed rotating magnetic dipole evaluated on a 2D Cartesian grid.

    The computational grid is interpreted as either:
        plane = "xy"  -> z = plane_offset
        plane = "xz"  -> y = plane_offset

    The magnetic dipole moment is

        mu(t) = mu0 [
            sin(chi) cos(Omega t + phase0),
            sin(chi) sin(Omega t + phase0),
            cos(chi)
        ].

    The magnetic field is approximated by the flat-space dipole formula

        B = B_scale * [3 (mu . rhat) rhat - mu] / r_eff^3.

    To avoid singularities at the origin, use either:
        - softening_radius, so r_eff = sqrt(r^2 + eps^2),
        - star_radius, inside which the field is regularized.

    Optionally, an induced electric field is included through

        E = - v_rot x B,

    where

        v_rot = Omega zhat x r.

    This is only a reduced toy prescription, not a full Deutsch solution and
    not a force-free magnetosphere.
    """

    mu0: float = 1.0
    omega: float = 0.2
    inclination: float = 0.5
    phase0: float = 0.0

    center: tuple[float, float] = (0.0, 0.0)
    plane: str = "xy"
    plane_offset: float = 0.0

    B_scale: float = 1.0
    softening_radius: float = 2.0
    star_radius: float = 5.0

    include_induced_E: bool = True
    electric_scale: float = 1.0

    light_cylinder_limit: bool = True
    max_velocity: float = 0.95

    def _coordinates_3d(self, grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if grid.ndim != 2:
            raise NotImplementedError("RotatingDipoleBackground2D requires a 2D grid.")

        if self.plane not in ("xy", "xz"):
            raise ValueError("plane must be 'xy' or 'xz'.")

        X, Y = grid.coordinates_2d()
        x0, y0 = self.center

        if self.plane == "xy":
            x = X - x0
            y = Y - y0
            z = np.zeros_like(X) + self.plane_offset
            return x, y, z

        # plane == "xz"
        x = X - x0
        y = np.zeros_like(X) + self.plane_offset
        z = Y - y0
        return x, y, z

    def _dipole_moment(self, t: float) -> np.ndarray:
        phase = self.omega * t + self.phase0
        chi = self.inclination

        return self.mu0 * np.array(
            [
                np.sin(chi) * np.cos(phase),
                np.sin(chi) * np.sin(phase),
                np.cos(chi),
            ],
            dtype=float,
        )

    def evaluate(self, t: float, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
        x, y, z = self._coordinates_3d(grid)
        mu = self._dipole_moment(t)

        r2 = x**2 + y**2 + z**2
        eps2 = self.softening_radius**2

        r_eff = np.sqrt(r2 + eps2)

        # Unit-like vector regularized by r_eff.
        rx = x / r_eff
        ry = y / r_eff
        rz = z / r_eff

        mu_dot_rhat = mu[0] * rx + mu[1] * ry + mu[2] * rz

        inv_r3 = 1.0 / np.maximum(r_eff**3, self.softening_radius**3)

        B = grid.zeros_vector()

        B[0] = self.B_scale * (3.0 * mu_dot_rhat * rx - mu[0]) * inv_r3
        B[1] = self.B_scale * (3.0 * mu_dot_rhat * ry - mu[1]) * inv_r3
        B[2] = self.B_scale * (3.0 * mu_dot_rhat * rz - mu[2]) * inv_r3

        # Optional regularization inside star_radius.
        if self.star_radius > 0.0:
            inside = np.sqrt(r2) < self.star_radius

            # Use a smooth suppression of the singular behaviour.
            # This keeps a finite interior field for visualization.
            suppression = np.ones_like(r_eff)
            suppression[inside] = (
                np.sqrt(r2[inside] + self.softening_radius**2)
                / self.star_radius
            ) ** 3

            for i in range(3):
                B[i] *= suppression

        E = grid.zeros_vector()

        if self.include_induced_E:
            # v = Omega zhat x r = (-Omega y, Omega x, 0)
            vx = -self.omega * y
            vy = +self.omega * x
            vz = np.zeros_like(x)

            if self.light_cylinder_limit:
                vmag = np.sqrt(vx**2 + vy**2 + vz**2)
                factor = np.minimum(1.0, self.max_velocity / np.maximum(vmag, 1.0e-14))
                vx *= factor
                vy *= factor
                vz *= factor

            # E = - v x B
            E[0] = -self.electric_scale * (vy * B[2] - vz * B[1])
            E[1] = -self.electric_scale * (vz * B[0] - vx * B[2])
            E[2] = -self.electric_scale * (vx * B[1] - vy * B[0])

        return E, B