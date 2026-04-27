"""
Schwarzschild-like and compact-object metrics.

Phase 18 introduces isotropic Schwarzschild metrics and smooth compact-object
toy metrics.

The Schwarzschild isotropic line element is

    ds^2 = -alpha^2 dt^2 + psi^4 (dx^2 + dy^2 + dz^2),

with

    psi  = 1 + M/(2r),
    alpha = (1 - M/(2r))/(1 + M/(2r)).

The spatial metric is conformally flat:

    gamma_ij = psi^4 delta_ij,
    gamma^ij = psi^-4 delta^ij,
    sqrt(gamma) = psi^6.

These metrics are static, with beta^i = 0 and K = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.geometry.base_metric import BaseMetric, GeometryFields


def _allocate_geometry_arrays(grid: Grid):
    """
    Allocate standard geometry arrays.
    """
    scalar_shape = grid.shape_full
    vector_shape = (3, *scalar_shape)
    tensor_shape = (3, 3, *scalar_shape)

    lapse = np.ones(scalar_shape, dtype=float)
    shift = np.zeros(vector_shape, dtype=float)
    gamma_down = np.zeros(tensor_shape, dtype=float)
    gamma_up = np.zeros(tensor_shape, dtype=float)
    sqrt_gamma = np.ones(scalar_shape, dtype=float)
    K = np.zeros(scalar_shape, dtype=float)

    return lapse, shift, gamma_down, gamma_up, sqrt_gamma, K


def _fill_conformally_flat_metric(
    gamma_down: np.ndarray,
    gamma_up: np.ndarray,
    psi: np.ndarray,
) -> None:
    """
    Fill gamma_ij = psi^4 delta_ij and gamma^ij = psi^-4 delta^ij.
    """
    psi4 = psi**4
    inv_psi4 = psi**(-4)

    for i in range(3):
        gamma_down[i, i] = psi4
        gamma_up[i, i] = inv_psi4


def _schwarzschild_isotropic_factors(
    r: np.ndarray,
    mass: float,
    lapse_floor: float = 1.0e-4,
    horizon_buffer: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return lapse alpha and conformal factor psi for isotropic Schwarzschild.

    The isotropic horizon is at r = M/2. We enforce a small floor outside it
    to avoid division by zero or negative lapse if the grid approaches the
    horizon.
    """
    if mass < 0.0:
        raise ValueError("mass must be non-negative.")

    if mass == 0.0:
        alpha = np.ones_like(r)
        psi = np.ones_like(r)
        return alpha, psi

    r_horizon = 0.5 * mass
    r_min = r_horizon * (1.0 + horizon_buffer)

    r_safe = np.maximum(r, r_min)

    q = mass / (2.0 * r_safe)

    psi = 1.0 + q
    alpha = (1.0 - q) / (1.0 + q)

    alpha = np.maximum(alpha, lapse_floor)

    return alpha, psi


@dataclass
class SchwarzschildIsotropicMetric1D(BaseMetric):
    """
    1D isotropic Schwarzschild metric.

    This is intended for radial-line or Cartesian-line reduced simulations.

    Parameters
    ----------
    mass:
        Schwarzschild mass M.
    center:
        Coordinate center. For a radial domain r in [r_min, r_max], use
        center=0 and use_absolute_radius=False.
    use_absolute_radius:
        If True, use r = |x-center|. Useful for line cuts through the object.
        If False, use r = x-center. Useful when x itself is a radial coordinate.
    radial_floor:
        Optional extra floor for r.
    lapse_floor:
        Minimum lapse value.
    horizon_buffer:
        Keeps r slightly outside the isotropic horizon r=M/2.
    """

    mass: float = 1.0
    center: float = 0.0
    use_absolute_radius: bool = False
    radial_floor: float = 1.0e-6
    lapse_floor: float = 1.0e-4
    horizon_buffer: float = 1.0e-3

    def evaluate(self, t: float, grid: Grid) -> GeometryFields:
        if grid.ndim != 1:
            raise NotImplementedError("SchwarzschildIsotropicMetric1D requires 1D.")

        x = grid.coordinates_1d()

        if self.use_absolute_radius:
            r = np.abs(x - self.center)
        else:
            r = x - self.center

        r = np.maximum(r, self.radial_floor)

        lapse, shift, gamma_down, gamma_up, sqrt_gamma, K = (
            _allocate_geometry_arrays(grid)
        )

        alpha, psi = _schwarzschild_isotropic_factors(
            r=r,
            mass=self.mass,
            lapse_floor=self.lapse_floor,
            horizon_buffer=self.horizon_buffer,
        )

        lapse[...] = alpha
        sqrt_gamma[...] = psi**6

        _fill_conformally_flat_metric(
            gamma_down=gamma_down,
            gamma_up=gamma_up,
            psi=psi,
        )

        return GeometryFields(
            lapse=lapse,
            shift=shift,
            gamma_down=gamma_down,
            gamma_up=gamma_up,
            sqrt_gamma=sqrt_gamma,
            K=K,
        )


@dataclass
class SchwarzschildIsotropicMetric2D(BaseMetric):
    """
    2D slice of isotropic Schwarzschild.

    The grid is interpreted as either:
        plane = "xy" -> z = plane_offset
        plane = "xz" -> y = plane_offset

    The metric remains the 3D spatial conformally flat Schwarzschild metric
    evaluated on the 2D slice.

    Important:
        In the current code, the 2D RHS still uses flat derivative operators.
        This metric affects lapse, index lowering/raising and E_i B^i, but
        full 2D covariant derivatives require the later curved-2D operator
        phase.
    """

    mass: float = 1.0
    center: tuple[float, float] = (0.0, 0.0)
    plane: str = "xy"
    plane_offset: float = 0.0
    radial_floor: float = 1.0e-6
    lapse_floor: float = 1.0e-4
    horizon_buffer: float = 1.0e-3

    def evaluate(self, t: float, grid: Grid) -> GeometryFields:
        if grid.ndim != 2:
            raise NotImplementedError("SchwarzschildIsotropicMetric2D requires 2D.")

        if self.plane not in ("xy", "xz"):
            raise ValueError("plane must be 'xy' or 'xz'.")

        X, Y = grid.coordinates_2d()
        x0, y0 = self.center

        if self.plane == "xy":
            x = X - x0
            y = Y - y0
            z = np.zeros_like(X) + self.plane_offset
        else:
            x = X - x0
            y = np.zeros_like(X) + self.plane_offset
            z = Y - y0

        r = np.sqrt(x**2 + y**2 + z**2)
        r = np.maximum(r, self.radial_floor)

        lapse, shift, gamma_down, gamma_up, sqrt_gamma, K = (
            _allocate_geometry_arrays(grid)
        )

        alpha, psi = _schwarzschild_isotropic_factors(
            r=r,
            mass=self.mass,
            lapse_floor=self.lapse_floor,
            horizon_buffer=self.horizon_buffer,
        )

        lapse[...] = alpha
        sqrt_gamma[...] = psi**6

        _fill_conformally_flat_metric(
            gamma_down=gamma_down,
            gamma_up=gamma_up,
            psi=psi,
        )

        return GeometryFields(
            lapse=lapse,
            shift=shift,
            gamma_down=gamma_down,
            gamma_up=gamma_up,
            sqrt_gamma=sqrt_gamma,
            K=K,
        )


@dataclass
class SmoothCompactObjectMetric2D(BaseMetric):
    """
    Smooth conformally flat compact-object toy metric.

    This avoids a black-hole horizon and is useful as a neutron-star-like
    effective metric.

    We use

        psi = 1 + A / sqrt(r^2 + R^2),

    and a smooth lapse

        alpha = sqrt(1 - 2 C exp[-r^2/(2R^2)]),

    clipped by lapse_floor.

    This is not an exact Einstein-equation solution. It is a controlled toy
    metric that produces a central gravitational redshift and spatial
    curvature without singular behaviour.
    """

    conformal_amplitude: float = 1.0
    compactness: float = 0.2
    radius: float = 10.0
    center: tuple[float, float] = (0.0, 0.0)
    plane: str = "xy"
    plane_offset: float = 0.0
    lapse_floor: float = 0.2

    def evaluate(self, t: float, grid: Grid) -> GeometryFields:
        if grid.ndim != 2:
            raise NotImplementedError("SmoothCompactObjectMetric2D requires 2D.")

        if self.plane not in ("xy", "xz"):
            raise ValueError("plane must be 'xy' or 'xz'.")

        X, Y = grid.coordinates_2d()
        x0, y0 = self.center

        if self.plane == "xy":
            x = X - x0
            y = Y - y0
            z = np.zeros_like(X) + self.plane_offset
        else:
            x = X - x0
            y = np.zeros_like(X) + self.plane_offset
            z = Y - y0

        r2 = x**2 + y**2 + z**2
        R = self.radius

        psi = 1.0 + self.conformal_amplitude / np.sqrt(r2 + R**2)

        lapse_arg = 1.0 - 2.0 * self.compactness * np.exp(-0.5 * r2 / R**2)
        lapse_arg = np.maximum(lapse_arg, self.lapse_floor**2)
        alpha = np.sqrt(lapse_arg)

        lapse, shift, gamma_down, gamma_up, sqrt_gamma, K = (
            _allocate_geometry_arrays(grid)
        )

        lapse[...] = alpha
        sqrt_gamma[...] = psi**6

        _fill_conformally_flat_metric(
            gamma_down=gamma_down,
            gamma_up=gamma_up,
            psi=psi,
        )

        return GeometryFields(
            lapse=lapse,
            shift=shift,
            gamma_down=gamma_down,
            gamma_up=gamma_up,
            sqrt_gamma=sqrt_gamma,
            K=K,
        )