"""
Neutron-star-inspired 2D reduced initial data.

These setups provide an axion cloud/halo around a compact object, while the
electromagnetic field may be prescribed separately as a rotating dipole
background.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State


def axion_cloud_around_compact_object_2d(
    grid: Grid,
    axion_amplitude: float = 1.0,
    cloud_radius: float = 15.0,
    cloud_width: float = 8.0,
    center: tuple[float, float] = (0.0, 0.0),
    axion_background: float = 0.0,
    angular_modulation: float = 0.0,
    azimuthal_mode: int = 1,
    Pi_amplitude: float = 0.0,
    Pi_width: float | None = None,
) -> State:
    """
    2D axion cloud/halo around a compact object.

    The default profile is a ring-like Gaussian:

        a(r) = A exp[-(r-r0)^2/(2 sigma^2)].

    Optional angular modulation:

        a -> a [1 + eps cos(m phi)].

    This is useful for toy models of axion structures around neutron stars.
    """
    if grid.ndim != 2:
        raise NotImplementedError("This setup requires a 2D grid.")

    X, Y = grid.coordinates_2d()
    x0, y0 = center

    x = X - x0
    y = Y - y0

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    profile = axion_amplitude * np.exp(
        -0.5 * ((r - cloud_radius) / cloud_width) ** 2
    )

    if angular_modulation != 0.0:
        profile *= 1.0 + angular_modulation * np.cos(azimuthal_mode * phi)

    a = axion_background + profile

    if Pi_width is None:
        Pi_width = cloud_width

    Pi = Pi_amplitude * np.exp(
        -0.5 * ((r - cloud_radius) / Pi_width) ** 2
    )

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    return State(a=a, Pi=Pi, E=E, B=B)


def gaussian_axion_core_2d(
    grid: Grid,
    axion_amplitude: float = 1.0,
    center: tuple[float, float] = (0.0, 0.0),
    width: float = 10.0,
    Pi_amplitude: float = 0.0,
) -> State:
    """
    Simple centered Gaussian axion cloud.
    """
    if grid.ndim != 2:
        raise NotImplementedError("This setup requires a 2D grid.")

    X, Y = grid.coordinates_2d()
    x0, y0 = center

    r2 = (X - x0) ** 2 + (Y - y0) ** 2

    a = axion_amplitude * np.exp(-0.5 * r2 / width**2)
    Pi = Pi_amplitude * np.exp(-0.5 * r2 / width**2)

    E = grid.zeros_vector()
    B = grid.zeros_vector()

    return State(a=a, Pi=Pi, E=E, B=B)