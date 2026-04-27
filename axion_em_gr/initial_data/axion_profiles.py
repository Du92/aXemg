"""
Initial data for the axion field.
"""

from __future__ import annotations

import numpy as np

from axion_em_gr.core.grid import Grid
from axion_em_gr.core.state import State


def gaussian_axion_packet(
    grid: Grid,
    amplitude: float = 1.0,
    center: float = 50.0,
    width: float = 5.0,
    momentum_amplitude: float = 0.0,
) -> State:
    """
    Gaussian initial data for the axion field.

    a(x, 0) = A exp[-(x - x0)^2 / (2 sigma^2)]
    Pi(x, 0) = constant or zero by default.

    This does not create a purely right-moving packet. It is simply a
    localized initial field configuration, useful for basic testing.
    """
    if grid.ndim != 1:
        raise NotImplementedError("Phase 1 supports only 1D initial data.")

    x = grid.coordinates_1d()

    a = amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    Pi = momentum_amplitude * np.ones_like(a)

    return State(a=a, Pi=Pi)


def sinusoidal_axion_mode(
    grid: Grid,
    amplitude: float = 1.0,
    mode_number: int = 1,
    mass: float = 0.0,
) -> State:
    """
    Standing-wave initial data useful for convergence and dispersion tests.

    For a massless field in flat space:

        a(x, 0) = A sin(k x)
        Pi(x, 0) = 0

    For a massive field, the exact angular frequency is:

        omega^2 = k^2 + m^2

    but Pi=0 still gives a standing oscillator.
    """
    if grid.ndim != 1:
        raise NotImplementedError("Phase 1 supports only 1D initial data.")

    x = grid.coordinates_1d()
    xmin, xmax = grid.bounds[0]
    L = xmax - xmin
    k = 2.0 * np.pi * mode_number / L

    a = amplitude * np.sin(k * (x - xmin))
    Pi = np.zeros_like(a)

    return State(a=a, Pi=Pi)
